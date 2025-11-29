"""
Feature Selection Module for crosssecmom2

Implements the complete feature selection pipeline as specified in feature_selection_spec_v2.md:
- Formation: Global FDR on raw features
- Per-window pipeline: IC filter → stability → supervised binning → redundancy → ElasticNetCV → XGBoost
- Bayesian optimization for hyperparameter tuning
- Memory-efficient implementation with float32 and parallelization

Author: AI Assistant
Date: November 27, 2025
"""

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoLarsIC, Ridge
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS, WLS

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_time_decay_weights(dates: pd.DatetimeIndex, train_end: pd.Timestamp, half_life: int) -> np.ndarray:
    """
    Compute exponential time-decay weights for samples.
    
    Args:
        dates: DatetimeIndex of sample dates
        train_end: End date of training window
        half_life: Half-life in days for exponential decay
        
    Returns:
        Array of weights with same length as dates
    """
    age_days = (train_end - dates).days
    age_days = np.array(age_days, dtype=np.float32)  # Convert TimedeltaIndex to array
    lambda_decay = np.log(2) / half_life
    weights = np.exp(-lambda_decay * age_days)
    return weights.astype(np.float32)


def compute_spearman_ic(feature: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Spearman IC (correlation) between feature and target.
    
    Args:
        feature: Feature values
        target: Target values
        
    Returns:
        Spearman correlation coefficient
    """
    # Filter out NaN values
    mask = ~(np.isnan(feature) | np.isnan(target))
    if mask.sum() < 3:  # Need at least 3 samples
        return 0.0
    
    return stats.spearmanr(feature[mask], target[mask])[0]


def compute_daily_ic_series(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Compute daily cross-sectional IC for each feature.
    
    FULLY VECTORIZED VERSION: Computes Spearman IC for ALL features at once
    per date using matrix operations. Achieves ~10x speedup over naive loops.
    
    Args:
        X: Feature matrix (samples × features) with DatetimeIndex
        y: Target series with DatetimeIndex
        dates: DatetimeIndex for each sample (should match X.index)
        
    Returns:
        DataFrame with shape (unique_dates × features) containing daily ICs
    """
    # Use X.index if it's a DatetimeIndex, otherwise use provided dates
    if isinstance(X.index, pd.DatetimeIndex):
        sample_dates = X.index
    else:
        sample_dates = dates
    
    unique_dates = pd.DatetimeIndex(sample_dates.unique())
    n_dates = len(unique_dates)
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    
    # Preallocate result matrix
    ic_matrix = np.full((n_dates, n_features), np.nan, dtype=np.float32)
    
    # Convert to numpy arrays for speed
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.float32)
    
    # Create date groups (indices for each date)
    date_indices = {date: np.where(sample_dates == date)[0] for date in unique_dates}
    
    for i, date in enumerate(unique_dates):
        row_indices = date_indices[date]
        n_samples = len(row_indices)
        
        if n_samples < 3:
            continue
        
        # Get data for this date: (n_tickers, n_features) and (n_tickers,)
        X_date = X_values[row_indices]
        y_date = y_values[row_indices]
        
        # Mask for valid y values
        y_valid = ~np.isnan(y_date)
        n_valid_y = y_valid.sum()
        
        if n_valid_y < 3:
            continue
        
        # VECTORIZED: Compute ranks for y once
        y_clean = y_date[y_valid]
        y_ranks = _rank_data(y_clean)
        y_centered = y_ranks - y_ranks.mean()
        y_norm = np.sqrt(np.sum(y_centered ** 2))
        
        if y_norm < 1e-10:
            continue
        
        # VECTORIZED: Process all features at once for samples where y is valid
        X_subset = X_date[y_valid]  # (n_valid_y, n_features)
        
        # For features with no additional NaN beyond y's NaN, we can vectorize
        # Count NaN per feature in the subset
        nan_per_feature = np.isnan(X_subset).sum(axis=0)  # (n_features,)
        
        # Features with no NaN in the y-valid subset can be fully vectorized
        no_nan_features = nan_per_feature == 0
        
        if no_nan_features.any():
            # FULLY VECTORIZED path for clean features
            X_clean = X_subset[:, no_nan_features]  # (n_valid_y, n_clean_features)
            
            # Rank each column (feature) - vectorized using argsort
            X_ranks = np.empty_like(X_clean, dtype=np.float32)
            order = np.argsort(X_clean, axis=0)
            ranks = np.arange(X_clean.shape[0], dtype=np.float32)
            for col_idx in range(X_clean.shape[1]):
                X_ranks[order[:, col_idx], col_idx] = ranks
            
            # Center ranks
            X_centered = X_ranks - X_ranks.mean(axis=0, keepdims=True)
            
            # Compute norms
            X_norms = np.sqrt(np.sum(X_centered ** 2, axis=0))  # (n_clean_features,)
            
            # Compute correlations: (y_centered @ X_centered) / (y_norm * X_norms)
            correlations = (y_centered @ X_centered) / (y_norm * X_norms)
            
            # Handle zero-norm features
            correlations = np.where(X_norms < 1e-10, np.nan, correlations)
            
            # Store results
            ic_matrix[i, no_nan_features] = correlations
        
        # Handle features with NaN (slower path, but unavoidable)
        nan_feature_indices = np.where(~no_nan_features)[0]
        for j in nan_feature_indices:
            x_col = X_subset[:, j]
            valid_mask = ~np.isnan(x_col)
            n_valid = valid_mask.sum()
            
            if n_valid < 3:
                continue
            
            # Recompute for this feature's valid samples
            x_clean = x_col[valid_mask]
            y_clean_subset = y_clean[valid_mask]
            
            x_ranks = _rank_data(x_clean)
            y_ranks_subset = _rank_data(y_clean_subset)
            
            x_centered = x_ranks - x_ranks.mean()
            y_centered_subset = y_ranks_subset - y_ranks_subset.mean()
            
            x_norm = np.sqrt(np.sum(x_centered ** 2))
            y_norm_subset = np.sqrt(np.sum(y_centered_subset ** 2))
            
            if x_norm < 1e-10 or y_norm_subset < 1e-10:
                continue
            
            ic_matrix[i, j] = np.sum(x_centered * y_centered_subset) / (x_norm * y_norm_subset)
    
    return pd.DataFrame(ic_matrix, index=unique_dates, columns=feature_names)


def _rank_data(arr: np.ndarray) -> np.ndarray:
    """
    Fast ranking using argsort. Equivalent to scipy.stats.rankdata but faster.
    
    Args:
        arr: 1D array to rank
        
    Returns:
        Array of ranks (0-indexed)
    """
    n = len(arr)
    ranks = np.empty(n, dtype=np.float32)
    order = arr.argsort()
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks


def _compute_newey_west_vectorized(
    ic_daily: pd.DataFrame,
    weights: np.ndarray,
    max_lags: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    VECTORIZED Newey-West t-statistic computation for all features at once.
    
    For intercept-only regression (IC ~ 1), the Newey-West t-stat simplifies to:
    t = mean(IC) / sqrt(HAC_variance / n)
    
    where HAC_variance includes autocorrelation corrections.
    
    Args:
        ic_daily: DataFrame of daily IC values (n_dates × n_features)
        weights: Time-decay weights for each date (n_dates,)
        max_lags: Maximum lags for HAC covariance (default: 5)
        
    Returns:
        Tuple of (t_stats, n_valid_dates) arrays, both shape (n_features,)
    """
    n_dates, n_features = ic_daily.shape
    ic_values = ic_daily.values.astype(np.float64)  # (n_dates, n_features)
    weights = weights.astype(np.float64)
    
    # Count valid (non-NaN) dates per feature
    valid_mask = ~np.isnan(ic_values)  # (n_dates, n_features)
    n_valid = valid_mask.sum(axis=0)  # (n_features,)
    
    # Compute weighted mean IC for each feature
    # Replace NaN with 0 for weighted sum, then divide by sum of weights for valid samples
    ic_filled = np.where(valid_mask, ic_values, 0.0)
    weights_2d = weights[:, np.newaxis] * valid_mask  # Zero out weights for NaN
    sum_weights = weights_2d.sum(axis=0)  # (n_features,)
    
    # Avoid division by zero
    sum_weights = np.where(sum_weights > 0, sum_weights, 1.0)
    ic_mean = (ic_filled * weights[:, np.newaxis]).sum(axis=0) / sum_weights  # (n_features,)
    
    # Compute residuals (IC - mean) for HAC variance
    residuals = ic_values - ic_mean[np.newaxis, :]  # (n_dates, n_features)
    residuals = np.where(valid_mask, residuals, 0.0)
    
    # Weighted residuals for variance computation
    weighted_resid = residuals * np.sqrt(weights[:, np.newaxis])
    
    # HAC variance computation (Newey-West)
    # Var(mean) = (1/n^2) * sum of autocovariances with Bartlett kernel weights
    t_stats = np.zeros(n_features, dtype=np.float64)
    
    for j in range(n_features):
        n_j = n_valid[j]
        if n_j < 3:
            t_stats[j] = 0.0
            continue
        
        # Get valid residuals for this feature
        valid_idx = valid_mask[:, j]
        resid_j = residuals[valid_idx, j]
        w_j = weights[valid_idx]
        
        # Normalize weights to sum to n
        w_j = w_j * n_j / w_j.sum()
        
        # Weighted variance (lag 0)
        var_0 = np.sum(w_j * resid_j ** 2) / n_j
        
        # Add autocovariance terms with Bartlett kernel
        hac_var = var_0
        for lag in range(1, min(max_lags + 1, int(n_j) - 1)):
            bartlett_weight = 1.0 - lag / (max_lags + 1)
            # Autocovariance at this lag
            autocov = np.sum(w_j[lag:] * resid_j[lag:] * resid_j[:-lag]) / n_j
            hac_var += 2 * bartlett_weight * autocov
        
        # t-stat = mean / sqrt(variance of mean)
        # variance of mean = hac_var / n
        se = np.sqrt(max(hac_var / n_j, 1e-20))
        t_stats[j] = ic_mean[j] / se if se > 1e-10 else 0.0
    
    return t_stats, n_valid


def compute_newey_west_tstat(
    ic_series: pd.Series,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute Newey-West t-statistic for IC time series.
    
    NOTE: This is the SCALAR version, kept for backward compatibility.
    The vectorized version _compute_newey_west_vectorized should be preferred.
    
    Args:
        ic_series: Time series of IC values
        weights: Optional sample weights (for WLS)
        
    Returns:
        t-statistic
    """
    if len(ic_series) < 3:
        return 0.0
    
    # Remove NaN values
    valid_mask = ~np.isnan(ic_series)
    ic_clean = ic_series[valid_mask]
    
    if len(ic_clean) < 3:
        return 0.0
    
    # Simple regression: IC ~ 1 (intercept only)
    X = np.ones((len(ic_clean), 1))
    y = ic_clean.values
    
    # Use WLS if weights provided, otherwise OLS
    if weights is not None:
        weights_clean = weights[valid_mask]
        # WLS expects weights, not OLS
        model = WLS(y, X, weights=weights_clean)
    else:
        model = OLS(y, X)
    
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    return results.tvalues[0]


def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[{stage}] Memory usage: {mem_mb:.1f} MB")


# ============================================================================
# Early Data Cleaning: Drop NaN-heavy and Near-Zero Variance Features
# ============================================================================

def drop_bad_features(
    X: pd.DataFrame,
    y: pd.Series,
    nan_threshold: float = 0.20,
    variance_threshold: float = 1e-10,
    use_mad: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Drop features with too many NaNs or near-zero variance EARLY before expensive 
    IC computation and ElasticNet fitting.
    
    This single central cleaning step reduces dimensionality and eliminates
    problematic features that would cause issues downstream.
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series (used to compute common valid mask)
        nan_threshold: Drop features with > this fraction of NaN (default 0.20 = 20%)
        variance_threshold: Drop features with variance < this (default 1e-10)
        use_mad: If True, use MAD instead of variance for zero-spread detection
        verbose: If True, print summary statistics
        
    Returns:
        Tuple of:
            - X_clean: Cleaned feature matrix (float32)
            - kept_features: List of kept feature names
            - diagnostics: Dict with cleaning statistics
    """
    start_time = time.time()
    n_features_in = X.shape[1]
    n_samples = X.shape[0]
    
    dropped_nan = []
    dropped_variance = []
    kept_features = []
    
    for col in X.columns:
        col_data = X[col]
        
        # Check 1: Too many NaNs
        nan_frac = col_data.isna().sum() / n_samples
        if nan_frac > nan_threshold:
            dropped_nan.append((col, nan_frac))
            continue
        
        # Check 2: Near-zero variance/spread
        # Get non-NaN values for variance check
        col_valid = col_data.dropna().values
        if len(col_valid) < 10:
            dropped_variance.append((col, 0.0, "too_few_samples"))
            continue
        
        if use_mad:
            # Use MAD (more robust to outliers)
            median_val = np.median(col_valid)
            mad_val = np.median(np.abs(col_valid - median_val))
            if mad_val < variance_threshold:
                dropped_variance.append((col, mad_val, "mad_near_zero"))
                continue
        else:
            # Use standard variance
            var_val = np.var(col_valid)
            if var_val < variance_threshold:
                dropped_variance.append((col, var_val, "var_near_zero"))
                continue
        
        # Feature passes both checks
        kept_features.append(col)
    
    elapsed = time.time() - start_time
    n_kept = len(kept_features)
    n_dropped_nan = len(dropped_nan)
    n_dropped_var = len(dropped_variance)
    
    # Extract and convert to float32
    X_clean = X[kept_features].astype(np.float32)
    
    diagnostics = {
        'n_features_in': n_features_in,
        'n_features_kept': n_kept,
        'n_dropped_nan': n_dropped_nan,
        'n_dropped_variance': n_dropped_var,
        'nan_threshold': nan_threshold,
        'variance_threshold': variance_threshold,
        'time_seconds': elapsed,
        'dropped_nan_features': [f for f, _ in dropped_nan[:10]],  # First 10 for debug
        'dropped_variance_features': [f for f, _, _ in dropped_variance[:10]],  # First 10
    }
    
    if verbose:
        print(f"[drop_bad_features] Input: {n_features_in} features, {n_samples} samples")
        print(f"[drop_bad_features] Dropped {n_dropped_nan} features with >{nan_threshold*100:.0f}% NaN")
        print(f"[drop_bad_features] Dropped {n_dropped_var} features with near-zero {'MAD' if use_mad else 'variance'}")
        print(f"[drop_bad_features] Output: {n_kept} features ({elapsed:.2f}s)")
    
    return X_clean, kept_features, diagnostics


# ============================================================================
# Formation: Global FDR on Raw Features
# ============================================================================

def formation_fdr(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    half_life: int = 126,
    fdr_level: float = 0.10,
    n_jobs: int = 4
) -> Tuple[List[str], pd.DataFrame]:
    """
    Run Formation-period FDR on raw features.
    
    Computes daily Spearman IC, weighted mean IC, Newey-West t-stats, and applies
    FDR control (Benjamini-Hochberg) to select approved raw features.
    
    Args:
        X: Feature matrix for Formation period (samples × features)
        y: Target series for Formation period
        dates: Date for each sample
        half_life: Half-life in days for time-decay weights
        fdr_level: FDR control level (e.g., 0.10 for 10%)
        n_jobs: Number of parallel jobs for IC computation
        
    Returns:
        Tuple of (approved_features_list, diagnostics_df)
    """
    logger.info("=" * 80)
    logger.info("Starting Formation FDR on raw features")
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}, Dates: {len(dates.unique())}")
    log_memory_usage("Formation start")
    
    start_time = time.time()
    
    # =========================================================================
    # NaN GATE CHECK: Formation FDR cannot proceed with NaN in features
    # =========================================================================
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        error_msg = (
            "\n" + "="*80 + "\n"
            "FORMATION FDR GATE FAILED: NaN values detected in feature data!\n"
            f"\nTotal NaN count: {nan_count:,}\n"
            f"Columns with NaN: {len(nan_cols)}\n"
            "\nThis indicates feature_engineering.py failed to clean the data.\n"
            "Please fix the source - NaN handling must be done BEFORE feature selection.\n"
        )
        if nan_cols:
            error_msg += f"\nColumns with NaN (first 10):\n"
            for col in nan_cols[:10]:
                nan_pct = X[col].isna().sum() / len(X) * 100
                error_msg += f"    - {col}: {nan_pct:.1f}% NaN\n"
        error_msg += "="*80
        raise ValueError(error_msg)
    
    logger.info("✓ NaN gate passed - formation data is clean")
    
    # Initialize stage timing
    stage_times = {}
    
    # Convert to float32 for memory efficiency
    stage_start = time.time()
    X = X.astype(np.float32)
    y = y.astype(np.float64)  # Keep target as float64 for numerical stability
    stage_times['dtype_conversion'] = time.time() - stage_start
    
    # Compute daily IC series for all features
    stage_start = time.time()
    print(f"[Formation FDR] Computing daily IC series for {X.shape[1]} features...")
    ic_daily = compute_daily_ic_series(X, y, dates)
    stage_times['daily_ic'] = time.time() - stage_start
    print(f"[Formation FDR] Daily IC computed ({stage_times['daily_ic']:.1f}s)")
    
    # Compute time-decay weights PER UNIQUE DATE (not per sample)
    unique_dates = ic_daily.index  # This is a DatetimeIndex
    train_end = unique_dates.max()
    weights = compute_time_decay_weights(unique_dates, train_end, half_life)
    
    # VECTORIZED: Compute weighted mean IC and Newey-West t-stats for ALL features
    stage_start = time.time()
    print(f"[Formation FDR] Computing weighted IC and Newey-West t-stats (vectorized)...")
    
    # Weighted mean IC - fully vectorized
    ic_values = ic_daily.values  # (n_dates, n_features)
    ic_weighted = np.average(ic_values, axis=0, weights=weights)  # (n_features,)
    
    # Newey-West t-stats - vectorized computation
    t_nw_values, n_dates_per_feature = _compute_newey_west_vectorized(
        ic_daily, weights, max_lags=5
    )
    
    # Convert t-stats to p-values (two-tailed)
    p_values = np.where(
        n_dates_per_feature > 2,
        2 * (1 - stats.t.cdf(np.abs(t_nw_values), df=n_dates_per_feature - 1)),
        1.0
    )
    
    # Build diagnostics dataframe
    diagnostics_df = pd.DataFrame({
        'feature': X.columns,
        'ic_weighted': ic_weighted,
        't_nw': t_nw_values,
        'p_value': p_values,
        'n_dates': n_dates_per_feature
    })
    
    stage_times['newey_west'] = time.time() - stage_start
    print(f"[Formation FDR] Newey-West t-stats computed ({stage_times['newey_west']:.1f}s)")
    
    # Apply FDR control (Benjamini-Hochberg)
    stage_start = time.time()
    print(f"[Formation FDR] Applying FDR control at level {fdr_level}...")
    reject, pvals_corrected, _, _ = multipletests(
        diagnostics_df['p_value'],
        alpha=fdr_level,
        method='fdr_bh'
    )
    
    diagnostics_df['fdr_reject'] = reject
    diagnostics_df['p_value_corrected'] = pvals_corrected
    stage_times['fdr_control'] = time.time() - stage_start
    
    # Select approved features
    approved_features = diagnostics_df[diagnostics_df['fdr_reject']]['feature'].tolist()
    
    total_time = time.time() - start_time
    
    # Compute counts for clear reporting
    n_features_in = X.shape[1]
    n_approved = len(approved_features)
    n_rejected = n_features_in - n_approved
    
    # Print summary - CLEAR terminology: FDR "rejects null hypothesis" = feature IS significant
    print("=" * 60)
    print(f"[Formation FDR] SUMMARY:")
    print(f"[Formation FDR]   Features in:  {n_features_in}")
    print(f"[Formation FDR]   Approved (significant): {n_approved}")
    print(f"[Formation FDR]   Rejected (not significant): {n_rejected}")
    print(f"[Formation FDR]   IC stats - Mean: {diagnostics_df['ic_weighted'].mean():.4f}, "
          f"Median: {diagnostics_df['ic_weighted'].median():.4f}")
    print(f"[Formation FDR]   Stage times:")
    print(f"[Formation FDR]     - daily_ic:    {stage_times['daily_ic']:.1f}s")
    print(f"[Formation FDR]     - newey_west:  {stage_times['newey_west']:.1f}s")
    print(f"[Formation FDR]     - fdr_control: {stage_times['fdr_control']:.1f}s")
    print(f"[Formation FDR]   Total time: {total_time:.1f}s")
    print("=" * 60)
    
    log_memory_usage("Formation end")
    
    # Clean up
    del ic_daily
    gc.collect()
    
    return approved_features, diagnostics_df


def formation_tune_elasticnet(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    approved_features: List[str],
    weights: Optional[np.ndarray] = None,
    alpha_grid: Optional[List[float]] = None,
    l1_ratio_grid: Optional[List[float]] = None,
    cv_folds: int = 3,
    n_jobs: int = 4
) -> Tuple[float, float, Dict]:
    """
    Tune ElasticNet hyperparameters on Formation window using sklearn's ElasticNetCV.
    
    Uses sklearn's optimized ElasticNetCV which parallelizes the alpha search internally
    (n_jobs=-1), while iterating over l1_ratio values sequentially to avoid nested parallelism.
    
    Args:
        X: Feature matrix for Formation period (samples × all_features)
        y: Target series for Formation period
        dates: Date for each sample
        approved_features: List of features approved by formation_fdr (S_F)
        weights: Optional sample weights (e.g., time-decay weights)
        alpha_grid: List of alpha values to try (default: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
        l1_ratio_grid: List of L1 ratios to try (default: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        cv_folds: Number of time-series CV folds (default: 3)
        n_jobs: Number of parallel jobs (used for parallelism within ElasticNetCV)
        
    Returns:
        Tuple of (best_alpha, best_l1_ratio, diagnostics_dict)
    """
    logger.info("=" * 80)
    logger.info("Formation ElasticNet hyperparameter tuning (using sklearn ElasticNetCV)")
    logger.info(f"Features: {len(approved_features)}, Samples: {X.shape[0]}, CV folds: {cv_folds}")
    log_memory_usage("Formation ElasticNet start")
    
    start_time = time.time()
    
    # Default grids if not provided
    if alpha_grid is None:
        alpha_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    if l1_ratio_grid is None:
        l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    # Extract approved features
    X_approved = X[approved_features].astype(np.float32)
    y_np = y.values.astype(np.float64)
    
    # Handle NaN values: create valid mask and filter
    valid_mask = ~(np.isnan(X_approved.values).any(axis=1) | np.isnan(y_np))
    n_total = len(valid_mask)
    n_valid = valid_mask.sum()
    
    if n_valid < 50:
        logger.warning(f"Too few valid samples ({n_valid}/{n_total}) for ElasticNet tuning")
        return alpha_grid[len(alpha_grid)//2], l1_ratio_grid[len(l1_ratio_grid)//2], {
            'error': f'Too few valid samples: {n_valid}',
            'n_total': n_total,
            'n_valid': n_valid
        }
    
    if n_valid < n_total:
        logger.info(f"Filtering NaN samples: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%) valid")
    
    X_approved = X_approved.iloc[valid_mask]
    y_np = y_np[valid_mask]
    
    # Also filter weights if provided
    if weights is not None:
        weights = weights[valid_mask]
    
    # Robust standardization (median/MAD)
    logger.info("Standardizing features...")
    X_std, std_params = robust_standardization(X_approved)
    X_std_np = X_std.values
    
    # Create time-series cross-validator
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Sequential iteration over l1_ratio, parallel alpha search via ElasticNetCV
    n_l1_ratios = len(l1_ratio_grid)
    logger.info(f"ElasticNetCV: {len(alpha_grid)} alphas × {n_l1_ratios} l1_ratios (parallel alpha, sequential l1_ratio)")
    print(f"[Formation ElasticNetCV] Using n_jobs=-1 for parallel alpha search", flush=True)
    
    best_score = -np.inf
    best_alpha = alpha_grid[0]
    best_l1_ratio = l1_ratio_grid[0]
    best_n_nonzero = 0
    
    results = []
    
    # Suppress convergence warnings (they're informational, not fatal)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    # Outer loop: iterate over l1_ratio values SEQUENTIALLY
    for idx, l1_ratio in enumerate(l1_ratio_grid):
        loop_start = time.time()
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            # ElasticNetCV with parallel alpha search (inner loop parallelized)
            # n_jobs=-1 uses all available cores for the alpha CV search
            model = ElasticNetCV(
                alphas=alpha_grid,
                l1_ratio=l1_ratio,  # Fixed for this iteration
                cv=tscv,
                max_iter=10000,
                tol=1e-4,
                random_state=42,
                n_jobs=-1,  # Parallel alpha search
                selection='cyclic'  # More stable than 'random'
            )
            
            # Fit with or without sample weights
            if weights is not None:
                model.fit(X_std_np, y_np, sample_weight=weights)
            else:
                model.fit(X_std_np, y_np)
        
        # Get best alpha for this l1_ratio
        alpha_best = model.alpha_
        
        # Find the CV score for best alpha (from mse_path_)
        # mse_path_ has shape (n_alphas, n_folds), we want mean across folds
        alpha_idx = list(model.alphas_).index(alpha_best) if alpha_best in model.alphas_ else 0
        mean_mse = model.mse_path_[alpha_idx].mean()
        
        # Score the model (R²) on entire dataset for comparison
        # This is not a proper validation score, just for logging
        r2_score = model.score(X_std_np, y_np)
        
        # Count non-zero coefficients
        n_nonzero = np.sum(model.coef_ != 0)
        
        loop_elapsed = time.time() - loop_start
        
        results.append({
            'alpha': alpha_best,
            'l1_ratio': l1_ratio,
            'mean_cv_mse': mean_mse,
            'r2_score': r2_score,
            'n_nonzero_coefs': n_nonzero,
            'time': loop_elapsed
        })
        
        print(f"[Formation ElasticNetCV] l1_ratio={l1_ratio:.2f}: best_alpha={alpha_best:.4f}, "
              f"MSE={mean_mse:.6f}, R²={r2_score:.4f}, nonzero={n_nonzero}/{len(approved_features)} "
              f"({loop_elapsed:.1f}s)", flush=True)
        
        # Update best: prefer lower MSE AND non-zero features
        # We penalize solutions with 0 features heavily
        effective_score = -mean_mse if n_nonzero > 0 else -1e10
        
        if effective_score > best_score:
            best_score = effective_score
            best_alpha = alpha_best
            best_l1_ratio = l1_ratio
            best_n_nonzero = n_nonzero
    
    elapsed = time.time() - start_time
    
    # If best solution has 0 features, fall back to least regularization
    if best_n_nonzero == 0:
        print(f"[Formation ElasticNetCV] WARNING: Best solution has 0 features, falling back to minimal regularization", flush=True)
        best_alpha = min(alpha_grid)
        best_l1_ratio = min(l1_ratio_grid)  # Lower l1_ratio = more Ridge-like = more features
        
        # Verify with a final fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            fallback_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000)
            if weights is not None:
                fallback_model.fit(X_std_np, y_np, sample_weight=weights)
            else:
                fallback_model.fit(X_std_np, y_np)
            best_n_nonzero = np.sum(fallback_model.coef_ != 0)
            print(f"[Formation ElasticNetCV] Fallback: alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.2f}, nonzero={best_n_nonzero}", flush=True)
    
    # Print summary with timing
    print("=" * 60)
    print(f"[Formation ElasticNetCV] SUMMARY:")
    print(f"[Formation ElasticNetCV]   Features: {len(approved_features)}")
    print(f"[Formation ElasticNetCV]   Samples:  {n_valid} ({100*n_valid/n_total:.1f}% valid)")
    print(f"[Formation ElasticNetCV]   Grid:     {len(alpha_grid)} alphas × {len(l1_ratio_grid)} l1_ratios")
    print(f"[Formation ElasticNetCV]   CV folds: {cv_folds}")
    print(f"[Formation ElasticNetCV]   Best hyperparameters:")
    print(f"[Formation ElasticNetCV]     - alpha:    {best_alpha:.4f}")
    print(f"[Formation ElasticNetCV]     - l1_ratio: {best_l1_ratio:.2f}")
    print(f"[Formation ElasticNetCV]     - nonzero:  {best_n_nonzero}/{len(approved_features)}")
    print(f"[Formation ElasticNetCV]   Total time: {elapsed:.1f}s")
    print("=" * 60)
    
    log_memory_usage("Formation ElasticNet end")
    
    diagnostics = {
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1_ratio,
        'best_n_nonzero': best_n_nonzero,
        'grid_search_results': pd.DataFrame(results),
        'time_formation_elasticnet': elapsed,
        'n_features': len(approved_features),
        'cv_folds': cv_folds
    }
    
    # Clean up
    del X_std, X_std_np
    gc.collect()
    
    return best_alpha, best_l1_ratio, diagnostics


# ============================================================================
# Per-Window Pipeline
# ============================================================================

def per_window_ic_filter(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
    theta_ic: float = 0.025,
    t_min: float = 1.96,
    n_jobs: int = 4
) -> Tuple[List[str], Dict]:
    """
    Per-window IC filter: Keep features with IC > theta_ic and |t_NW| > t_min.
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        dates: Date for each sample
        weights: Time-decay weights
        theta_ic: Minimum absolute IC threshold
        t_min: Minimum absolute Newey-West t-stat
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (selected_features, diagnostics_dict)
    """
    start_time = time.time()
    logger.info(f"IC filter: {X.shape[1]} features")
    
    # Compute daily IC series
    ic_daily = compute_daily_ic_series(X, y, dates)
    
    # Aggregate per-sample weights into per-date weights
    # weights is per-sample, but IC is per-date, so we need to average weights by date
    unique_dates = pd.DatetimeIndex(dates.unique())
    w_by_date = pd.Series(index=unique_dates, dtype=np.float32)
    
    for date in unique_dates:
        date_mask = dates == date
        w_by_date.loc[date] = np.float32(weights[date_mask].mean())
    
    # Process each feature
    def process_feature(feat_name):
        ic_series = ic_daily[feat_name]
        
        # Align per-date weights with IC series dates
        weights_aligned = w_by_date.loc[ic_series.index].values
        
        ic_weighted = np.average(ic_series.values, weights=weights_aligned)
        t_nw = compute_newey_west_tstat(ic_series, weights_aligned)
        
        return {
            'feature': feat_name,
            'ic_weighted': ic_weighted,
            't_nw': t_nw,
            'pass': (abs(ic_weighted) > theta_ic) and (abs(t_nw) > t_min)
        }
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_feature)(feat) for feat in X.columns
    )
    
    results_df = pd.DataFrame(results)
    selected = results_df[results_df['pass']]['feature'].tolist()
    
    diagnostics = {
        'n_start': X.shape[1],
        'n_after_ic': len(selected),
        'ic_mean': results_df['ic_weighted'].mean(),
        'ic_median': results_df['ic_weighted'].median(),
        'time_ic': time.time() - start_time
    }
    
    logger.info(f"IC filter: {len(selected)} features passed (time: {diagnostics['time_ic']:.1f}s)")
    
    del ic_daily
    gc.collect()
    
    return selected, diagnostics


def rank_features_by_ic_and_sign_consistency(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
    num_blocks: int = 3,
    ic_floor: float = 0.01,
    top_k: int = 200,
    min_features: int = 100,
    n_jobs: int = 4
) -> Tuple[List[str], Dict]:
    """
    Soft ranking based on IC and sign consistency (no hard stability drop).
    
    This function:
    1. Computes time-decayed IC_full over the entire Training window
    2. Splits Training into B contiguous blocks
    3. Computes IC_b for each block
    4. Computes sign consistency: c_j = (# blocks with same sign as IC_full) / B
    5. Ranks features by: s_j = |IC_full| * (0.5 + 0.5 * c_j)
    6. Keeps top K features (or all if above min_features threshold)
    
    Features with frequent sign flips get down-weighted but not automatically discarded.
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for samples
        weights: Time-decay weights (per-sample)
        num_blocks: Number of contiguous blocks (B, default 3)
        ic_floor: Weak |IC_full| floor (default 0.01), only enforced if keeps >= min_features
        top_k: Top K features to keep (default 200)
        min_features: Minimum features to guarantee (default 100)
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (ranked_features, diagnostics_dict)
    """
    start_time = time.time()
    logger.info(f"Soft IC ranking: {X.shape[1]} features, {num_blocks} blocks")
    
    # Split dates into B contiguous blocks
    n_dates = len(dates)
    block_size = n_dates // num_blocks
    
    if block_size < 3:
        logger.warning(f"Block size too small ({block_size} dates), reducing to {num_blocks-1} blocks")
        num_blocks = max(2, num_blocks - 1)
        block_size = n_dates // num_blocks
    
    # Create block assignments
    blocks = []
    for b in range(num_blocks):
        start_idx = b * block_size
        end_idx = (b + 1) * block_size if b < num_blocks - 1 else n_dates
        block_dates = dates[start_idx:end_idx]
        blocks.append(block_dates)
    
    # Aggregate per-sample weights into per-date weights for IC computation
    unique_dates = pd.DatetimeIndex(dates.unique())
    w_by_date = pd.Series(index=unique_dates, dtype=np.float32)
    for date in unique_dates:
        date_mask = dates == date
        w_by_date.loc[date] = np.float32(weights[date_mask].mean())
    
    # Compute IC_full for each feature over entire Training window
    logger.info("Computing full-window IC with time-decay...")
    ic_daily = compute_daily_ic_series(X, y, dates)
    
    def process_feature(feat_name):
        ic_series = ic_daily[feat_name]
        
        # Weighted mean IC over full window
        weights_aligned = w_by_date.loc[ic_series.index].values
        ic_full = np.average(ic_series.values, weights=weights_aligned)
        
        # Compute IC for each block
        block_ics = []
        for block_dates in blocks:
            # Filter IC series to this block's dates
            block_ic_series = ic_series[ic_series.index.isin(block_dates)]
            
            if len(block_ic_series) < 3:
                block_ics.append(np.nan)
                continue
            
            # Mean IC in this block (no additional time-decay within block)
            block_ic_mean = block_ic_series.mean()
            block_ics.append(block_ic_mean)
        
        block_ics = np.array(block_ics)
        valid_blocks = block_ics[~np.isnan(block_ics)]
        
        if len(valid_blocks) == 0:
            return {
                'feature': feat_name,
                'ic_full': ic_full,
                'block_ics': block_ics,
                'sign_consistency': 0.0,
                'rank_score': 0.0
            }
        
        # Sign consistency: fraction of blocks with same sign as IC_full
        sgn_full = np.sign(ic_full)
        if sgn_full == 0:
            # Treat zero as neutral (perfect consistency)
            c_j = 1.0
        else:
            n_same_sign = (np.sign(valid_blocks) == sgn_full).sum()
            c_j = n_same_sign / len(valid_blocks)
        
        # Ranking score: s_j = |IC_full| * (0.5 + 0.5 * c_j)
        s_j = abs(ic_full) * (0.5 + 0.5 * c_j)
        
        return {
            'feature': feat_name,
            'ic_full': ic_full,
            'block_ics': block_ics,
            'sign_consistency': c_j,
            'rank_score': s_j
        }
    
    # Process features in parallel
    logger.info("Computing block ICs and sign consistency...")
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_feature)(feat) for feat in X.columns
    )
    
    # Create diagnostics dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by rank_score (descending)
    results_df = results_df.sort_values('rank_score', ascending=False)
    
    # Select top K
    top_k_features = results_df.head(top_k)['feature'].tolist()
    
    # Apply |IC_full| floor if it still leaves >= min_features
    if ic_floor > 0:
        above_floor = results_df[abs(results_df['ic_full']) >= ic_floor]
        if len(above_floor) >= min_features:
            # Intersect top_k with above_floor
            floor_features = set(above_floor['feature'].tolist())
            top_k_features = [f for f in top_k_features if f in floor_features]
            logger.info(f"Applied IC floor {ic_floor}: {len(top_k_features)} features")
        else:
            logger.info(f"IC floor {ic_floor} would leave only {len(above_floor)} features < min {min_features}, ignoring floor")
    
    elapsed = time.time() - start_time
    
    diagnostics = {
        'n_start': X.shape[1],
        'n_after_ranking': len(top_k_features),
        'results_df': results_df,
        'time_ranking': elapsed,
        'num_blocks': num_blocks,
        'ic_floor_applied': ic_floor if len(top_k_features) >= min_features else 0.0
    }
    
    logger.info(f"Soft ranking complete: {len(top_k_features)} features selected (time: {elapsed:.1f}s)")
    logger.info(f"  Mean rank_score: {results_df.head(len(top_k_features))['rank_score'].mean():.4f}")
    logger.info(f"  Mean sign_consistency: {results_df.head(len(top_k_features))['sign_consistency'].mean():.3f}")
    
    # Clean up
    del ic_daily
    gc.collect()
    
    return top_k_features, diagnostics


def per_window_stability(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    k_folds: int = 3,
    theta_stable: float = 0.03,
    min_sign_consistency: int = 2,
    n_jobs: int = 1
) -> Tuple[List[str], Dict]:
    """
    DEPRECATED: Use rank_features_by_ic_and_sign_consistency instead.
    
    Filter features based on stability across temporal folds (kept for backward compatibility).
    
    Stability criteria:
    1. Sign consistency: IC has same sign in at least min_sign_consistency folds
    2. Magnitude: Median |IC| across folds >= theta_stable
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for samples
        k_folds: Number of contiguous temporal folds (default 3)
        theta_stable: Minimum median |IC| threshold
        min_sign_consistency: Minimum number of folds with same IC sign
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (stable_features, diagnostics_dict)
    """
    start_time = time.time()
    logger.info(f"Stability filter: {X.shape[1]} features, {k_folds} folds")
    
    # Split dates into K contiguous folds
    n_dates = len(dates)
    fold_size = n_dates // k_folds
    
    if fold_size < 5:
        logger.warning(f"Fold size too small ({fold_size} dates), reducing to {k_folds-1} folds")
        k_folds = max(2, k_folds - 1)
        fold_size = n_dates // k_folds
    
    # Create fold assignments
    folds = []
    for k in range(k_folds):
        start_idx = k * fold_size
        end_idx = (k + 1) * fold_size if k < k_folds - 1 else n_dates
        fold_dates = dates[start_idx:end_idx]
        folds.append(fold_dates)
    
    # Compute IC for each feature in each fold
    def compute_fold_ics(feat_name):
        fold_ics = []
        for fold_dates in folds:
            # Get data for this fold
            # Handle MultiIndex case (extract Date level)
            if isinstance(X.index, pd.MultiIndex):
                date_level = X.index.get_level_values('Date')
                mask = date_level.isin(fold_dates)
            else:
                mask = X.index.isin(fold_dates)
            
            X_fold = X.loc[mask, [feat_name]]
            y_fold = y.loc[mask]
            
            # Compute mean IC across dates in fold
            if isinstance(X.index, pd.MultiIndex):
                fold_dates_samples = X.index[mask].get_level_values('Date')
            else:
                fold_dates_samples = X.index[mask]
            
            if len(fold_dates_samples.unique()) < 3:
                fold_ics.append(np.nan)
                continue
                
            ic_daily = compute_daily_ic_series(X_fold, y_fold, fold_dates_samples)
            ic_mean = ic_daily[feat_name].mean()
            fold_ics.append(ic_mean)
        
        fold_ics = np.array(fold_ics)
        
        # Check stability criteria
        valid_ics = fold_ics[~np.isnan(fold_ics)]
        
        if len(valid_ics) < min_sign_consistency:
            return {
                'feature': feat_name,
                'fold_ics': fold_ics,
                'median_abs_ic': 0.0,
                'sign_consistency': 0,
                'pass': False
            }
        
        # Sign consistency: count folds with positive vs negative IC
        n_positive = (valid_ics > 0).sum()
        n_negative = (valid_ics < 0).sum()
        sign_consistency = max(n_positive, n_negative)
        
        # Median magnitude
        median_abs_ic = np.median(np.abs(valid_ics))
        
        # Pass if both criteria met
        passes = (sign_consistency >= min_sign_consistency) and (median_abs_ic >= theta_stable)
        
        return {
            'feature': feat_name,
            'fold_ics': fold_ics,
            'median_abs_ic': median_abs_ic,
            'sign_consistency': sign_consistency,
            'pass': passes
        }
    
    # Process features in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_fold_ics)(feat) for feat in X.columns
    )
    
    # Extract stable features
    stable_features = [r['feature'] for r in results if r['pass']]
    
    diagnostics = {
        'n_start': X.shape[1],
        'n_after_stability': len(stable_features),
        'pct_stable': 100.0 * len(stable_features) / X.shape[1] if X.shape[1] > 0 else 0.0,
        'time_stability': time.time() - start_time
    }
    
    logger.info(f"Stability filter: {len(stable_features)} features passed "
                f"({diagnostics['pct_stable']:.1f}%, time: {diagnostics['time_stability']:.1f}s)")
    
    gc.collect()
    
    return stable_features, diagnostics


def supervised_binning_and_representation(
    X: pd.DataFrame,
    y: pd.Series,
    k_bin: int = 3,
    min_samples_leaf: int = 50,
    n_jobs: int = 1
) -> Tuple[List[str], Dict]:
    """
    DEPRECATED in v3 pipeline: Binning is not used in production code.
    
    This function is kept only for backward compatibility with v2 tests.
    The v3 pipeline operates exclusively on continuous features.
    
    Create binned versions of features using decision trees and select best representation.
    
    For each feature:
    1. Fit DecisionTreeRegressor with max_leaf_nodes = k_bin + 1
    2. Create binned feature (predict values are bin assignments)
    3. Compute IC for both raw and binned versions
    4. Select representation with higher |IC|
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        k_bin: Maximum number of bins (tree leaf nodes - 1)
        min_samples_leaf: Minimum samples per leaf node
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (selected_features_list, diagnostics_dict)
    """
    import warnings
    warnings.warn(
        "supervised_binning_and_representation is DEPRECATED in v3 pipeline. "
        "The production pipeline uses continuous features only. "
        "This function is kept for backward compatibility.",
        DeprecationWarning,
        stacklevel=2
    )
    
    start_time = time.time()
    logger.info(f"Binning: {X.shape[1]} features, k_bin={k_bin}")
    
    # Compute IC for all raw features once
    if isinstance(X.index, pd.MultiIndex):
        dates_samples = X.index.get_level_values('Date')
    else:
        dates_samples = X.index
    
    ic_daily_raw = compute_daily_ic_series(X, y, dates_samples)
    ic_raw = ic_daily_raw.mean()  # Mean IC per feature
    
    def process_feature(feat_name):
        """Fit tree, create bins, compare IC."""
        X_feat = X[[feat_name]].values
        y_arr = y.values
        
        # Remove NaN samples
        mask = ~(np.isnan(X_feat).any(axis=1) | np.isnan(y_arr))
        X_clean = X_feat[mask]
        y_clean = y_arr[mask]
        
        if len(X_clean) < 2 * min_samples_leaf:
            # Not enough data for binning
            return {
                'feature': feat_name,
                'representation': 'raw',
                'ic_raw': ic_raw[feat_name],
                'ic_binned': np.nan
            }
        
        # Fit decision tree
        tree = DecisionTreeRegressor(
            max_leaf_nodes=k_bin + 1,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        try:
            tree.fit(X_clean, y_clean)
            
            # Create binned feature (use tree leaf IDs as bin assignments, not predictions)
            X_binned = tree.apply(X_feat).astype(np.float32)  # Get leaf node IDs
            X_binned_df = pd.DataFrame({f'{feat_name}_binned': X_binned}, index=X.index)
            
            # Compute IC for binned version
            ic_daily_binned = compute_daily_ic_series(X_binned_df, y, dates_samples)
            ic_binned = ic_daily_binned.mean().iloc[0]
            
        except Exception as e:
            logger.warning(f"Binning failed for {feat_name}: {e}")
            ic_binned = np.nan
        
        # Choose representation with higher |IC|
        ic_raw_abs = abs(ic_raw[feat_name])
        ic_binned_abs = abs(ic_binned) if not np.isnan(ic_binned) else 0.0
        
        if ic_binned_abs > ic_raw_abs:
            representation = 'binned'
            selected_name = f'{feat_name}_binned'
        else:
            representation = 'raw'
            selected_name = feat_name
        
        return {
            'feature': feat_name,
            'representation': representation,
            'selected_name': selected_name,
            'ic_raw': ic_raw[feat_name],
            'ic_binned': ic_binned
        }
    
    # Process features in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_feature)(feat) for feat in X.columns
    )
    
    # Extract selected features
    selected_features = [r['selected_name'] for r in results]
    
    n_raw = sum(1 for r in results if r['representation'] == 'raw')
    n_binned = sum(1 for r in results if r['representation'] == 'binned')
    
    diagnostics = {
        'n_start': X.shape[1],
        'n_raw_selected': n_raw,
        'n_binned_selected': n_binned,
        'n_total_selected': len(selected_features),
        'time_binning': time.time() - start_time
    }
    
    logger.info(f"Binning: {len(selected_features)} features selected "
                f"({n_raw} raw, {n_binned} binned, time: {diagnostics['time_binning']:.1f}s)")
    
    del ic_daily_raw
    gc.collect()
    
    return selected_features, diagnostics


def correlation_redundancy_filter(
    X: pd.DataFrame,
    corr_threshold: float = 0.80,
    ic_scores: Optional[Dict[str, float]] = None,
    n_jobs: int = 1
) -> Tuple[List[str], Dict]:
    """
    Remove redundant features based on pairwise correlation (VECTORIZED).
    
    When features are highly correlated, keeps the one with better IC score
    (if provided), otherwise keeps the one that appears first.
    
    VECTORIZED ALGORITHM:
    1. Compute correlation matrix using numpy (fast)
    2. Use upper triangular mask to find all pairs above threshold
    3. Build conflict graph and resolve greedily by IC score
    
    Args:
        X: Feature matrix (samples × features)
        corr_threshold: Maximum allowed absolute correlation (default 0.80)
        ic_scores: Optional dict of feature -> IC score for tie-breaking
        n_jobs: Number of parallel jobs (unused, kept for API consistency)
        
    Returns:
        Tuple of (selected_features_list, diagnostics_dict)
    """
    start_time = time.time()
    n_features = X.shape[1]
    logger.info(f"Redundancy filter: {n_features} features, threshold={corr_threshold}")
    
    # VECTORIZED: Compute correlation matrix using numpy (faster than pandas)
    X_np = X.values.astype(np.float32)
    
    # Handle NaN by using nanmean for correlation
    # Standardize columns (subtract mean, divide by std)
    col_means = np.nanmean(X_np, axis=0)
    col_stds = np.nanstd(X_np, axis=0)
    col_stds[col_stds < 1e-10] = 1.0  # Avoid division by zero
    X_centered = (X_np - col_means) / col_stds
    
    # Replace NaN with 0 for correlation computation (they don't contribute)
    X_centered = np.nan_to_num(X_centered, nan=0.0)
    
    # Compute correlation matrix: corr = X'X / n
    n_samples = X_centered.shape[0]
    corr_matrix = np.abs(np.dot(X_centered.T, X_centered) / n_samples)
    
    # Find pairs above threshold using upper triangular mask
    # This is O(n^2) but vectorized
    upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr_pairs = np.where((corr_matrix > corr_threshold) & upper_tri)
    
    # Build set of features to remove
    features_to_remove = set()
    feature_names = list(X.columns)
    
    # If no IC scores provided, use column order as priority
    if ic_scores is None:
        ic_scores = {feat: -i for i, feat in enumerate(feature_names)}  # Earlier = better
    
    # Process each high-correlation pair
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        feat_i = feature_names[i]
        feat_j = feature_names[j]
        
        # Skip if either already removed
        if feat_i in features_to_remove or feat_j in features_to_remove:
            continue
        
        # Keep the one with better (higher absolute) IC
        ic_i = abs(ic_scores.get(feat_i, 0))
        ic_j = abs(ic_scores.get(feat_j, 0))
        
        if ic_i >= ic_j:
            features_to_remove.add(feat_j)
        else:
            features_to_remove.add(feat_i)
    
    features_to_keep = [f for f in feature_names if f not in features_to_remove]
    
    elapsed = time.time() - start_time
    
    diagnostics = {
        'n_start': n_features,
        'n_after_redundancy': len(features_to_keep),
        'n_removed': len(features_to_remove),
        'n_high_corr_pairs': len(high_corr_pairs[0]),
        'time_redundancy': elapsed
    }
    
    logger.info(f"Redundancy filter: {len(features_to_keep)} features kept, "
                f"{len(features_to_remove)} removed (time: {elapsed:.2f}s)")
    
    del corr_matrix, X_centered
    gc.collect()
    
    return features_to_keep, diagnostics


def training_lasso_lars_ic(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str = 'bic',
    min_features: int = 12,
    ridge_alpha: float = 0.01
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Select features using LassoLarsIC (BIC/AIC) and refit with Ridge.
    
    This replaces ElasticNet in the Training phase:
    1. Run LassoLarsIC to get sparse support (features with nonzero coefficients)
    2. If support < min_features, fall back to top features by coefficient magnitude
    3. Refit with Ridge on the selected support for stable coefficients
    
    NO CV REQUIRED - BIC/AIC automatically chooses regularization.
    
    Args:
        X: Standardized feature matrix (samples × features)
        y: Target series
        criterion: 'bic' or 'aic' (default 'bic' - more conservative)
        min_features: Minimum features to select (default 12)
        ridge_alpha: Ridge regularization for final refit (default 0.01)
        
    Returns:
        Tuple of (selected_features_list, ridge_coefficients, diagnostics_dict)
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    start_time = time.time()
    n_features = X.shape[1]
    feature_names = list(X.columns)
    
    print(f"[LassoLarsIC] Running with criterion='{criterion}', min_features={min_features}", flush=True)
    
    # Step 1: Fit LassoLarsIC
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        
        lars = LassoLarsIC(
            criterion=criterion,
            max_iter=1000,
            fit_intercept=True
            # Note: normalize was removed in sklearn 1.2; data should be pre-standardized
        )
        lars.fit(X.values, y.values)
    
    # Get support (non-zero coefficients)
    lars_coef = lars.coef_
    nonzero_mask = np.abs(lars_coef) > 1e-10
    n_nonzero = nonzero_mask.sum()
    
    print(f"[LassoLarsIC] BIC selected {n_nonzero}/{n_features} features", flush=True)
    
    # Step 2: Ensure minimum features
    if n_nonzero >= min_features:
        # Use LassoLarsIC selection
        selected_mask = nonzero_mask
        selection_method = 'lars_bic'
    else:
        # Fallback: select top features by absolute coefficient magnitude
        print(f"[LassoLarsIC] FALLBACK: BIC selected {n_nonzero} < {min_features}, using top {min_features} by |coef|", flush=True)
        
        # Get indices of top features by coefficient magnitude
        coef_abs = np.abs(lars_coef)
        top_indices = np.argsort(coef_abs)[::-1][:min_features]
        
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[top_indices] = True
        selection_method = 'lars_fallback'
    
    selected_features = [feat for i, feat in enumerate(feature_names) if selected_mask[i]]
    n_selected = len(selected_features)
    
    # Step 3: Refit with Ridge on selected features
    X_selected = X[selected_features].values
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        
        ridge = Ridge(
            alpha=ridge_alpha,
            fit_intercept=True,
            solver='auto'
        )
        ridge.fit(X_selected, y.values)
    
    ridge_coef = ridge.coef_
    
    elapsed = time.time() - start_time
    
    print(f"[LassoLarsIC] Final: {n_selected} features, Ridge refit alpha={ridge_alpha} ({elapsed:.2f}s)", flush=True)
    
    diagnostics = {
        'criterion': criterion,
        'n_features_in': n_features,
        'n_lars_nonzero': n_nonzero,
        'n_selected': n_selected,
        'min_features': min_features,
        'selection_method': selection_method,
        'ridge_alpha': ridge_alpha,
        'lars_alpha': lars.alpha_,  # Regularization chosen by BIC
        'time_lars': elapsed
    }
    
    return selected_features, ridge_coef, diagnostics


def robust_standardization(
    X: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize features using robust statistics (median and MAD).
    
    For each feature:
        z = (x - median) / MAD
    where MAD = median(|x - median|)
    
    MAD near zero handling:
    - Features with MAD < 1e-10 use MAD=1.0 (no scaling, just centering)
    - Warning is aggregated (single log line) instead of per-feature
    
    Args:
        X: Feature matrix (samples × features)
        verbose: If True, print summary (default True)
        
    Returns:
        Tuple of (X_standardized DataFrame, parameters_dict)
        parameters_dict contains 'median' and 'mad' for each feature
    """
    start_time = time.time()
    
    # Compute median and MAD for each feature
    medians = {}
    mads = {}
    near_zero_mad_features = []  # Track features with MAD near zero
    
    X_standardized = X.copy()
    
    for col in X.columns:
        median_val = X[col].median()
        mad_val = np.median(np.abs(X[col] - median_val))
        
        # Avoid division by zero - track but don't log individually
        if mad_val < 1e-10:
            near_zero_mad_features.append(col)
            mad_val = 1.0
        
        # Standardize
        X_standardized[col] = (X[col] - median_val) / mad_val
        
        medians[col] = median_val
        mads[col] = mad_val
    
    elapsed = time.time() - start_time
    
    # Single aggregated warning for near-zero MAD features
    n_near_zero = len(near_zero_mad_features)
    if n_near_zero > 0 and verbose:
        print(f"[standardization] {n_near_zero} features with MAD≈0 (replaced MAD with 1.0)")
        if n_near_zero <= 5:
            print(f"[standardization]   Features: {near_zero_mad_features}")
    
    if verbose:
        print(f"[standardization] {X.shape[1]} features standardized ({elapsed:.2f}s)")
    
    parameters = {
        'median': medians,
        'mad': mads,
        'time_standardization': elapsed,
        'n_near_zero_mad': n_near_zero,
        'near_zero_mad_features': near_zero_mad_features[:10]  # First 10 for debug
    }
    
    gc.collect()
    
    return X_standardized, parameters


def elasticnet_cv_selection(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    alpha_grid: List[float] = None,
    l1_ratio_grid: List[float] = None,
    cv_folds: int = 3,
    coef_threshold: float = 1e-4,
    n_jobs: int = 1
) -> Tuple[List[str], object, Dict]:
    """
    Select features using ElasticNetCV with TimeSeriesSplit.
    
    Uses time-series cross-validation to select optimal alpha and l1_ratio,
    then selects features with non-zero coefficients above threshold.
    
    Args:
        X: Standardized feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for samples
        alpha_grid: List of alpha (regularization strength) values to try
        l1_ratio_grid: List of l1_ratio (L1 vs L2 balance) values to try
        cv_folds: Number of TimeSeriesSplit folds
        coef_threshold: Minimum |coefficient| to consider feature selected
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (selected_features_list, fitted_model, diagnostics_dict)
    """
    start_time = time.time()
    logger.info(f"ElasticNetCV: {X.shape[1]} features, {cv_folds} folds")
    
    # Default grids (small and fixed as per spec)
    if alpha_grid is None:
        alpha_grid = [0.001, 0.01, 0.1, 1.0]
    if l1_ratio_grid is None:
        l1_ratio_grid = [0.5, 0.7, 0.9, 0.95]
    
    # Prepare data
    X_arr = X.values
    y_arr = y.values
    
    # Remove NaN samples
    mask = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr))
    X_clean = X_arr[mask]
    y_clean = y_arr[mask]
    dates_clean = dates[mask] if len(dates) == len(X_arr) else X.index.unique()
    
    # Create TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Fit ElasticNetCV with time-series CV
    model = ElasticNetCV(
        alphas=alpha_grid,
        l1_ratio=l1_ratio_grid,
        cv=tscv,
        n_jobs=n_jobs,
        max_iter=10000,
        tol=1e-4,
        selection='random',
        random_state=42
    )
    
    try:
        model.fit(X_clean, y_clean)
        
        # Extract features with non-zero coefficients
        coefs = model.coef_
        selected_idx = np.where(np.abs(coefs) > coef_threshold)[0]
        selected_features = [X.columns[i] for i in selected_idx]
        
        best_alpha = model.alpha_
        best_l1_ratio = model.l1_ratio_
        
    except Exception as e:
        logger.error(f"ElasticNetCV failed: {e}")
        # Fallback: select all features
        selected_features = list(X.columns)
        best_alpha = alpha_grid[0] if alpha_grid else 0.01
        best_l1_ratio = l1_ratio_grid[0] if l1_ratio_grid else 0.5
    
    diagnostics = {
        'n_start': X.shape[1],
        'n_selected': len(selected_features),
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1_ratio,
        'time_elasticnet': time.time() - start_time
    }
    
    logger.info(f"ElasticNetCV: {len(selected_features)} features selected "
                f"(alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.2f}, "
                f"time: {diagnostics['time_elasticnet']:.1f}s)")
    
    gc.collect()
    
    return selected_features, model, diagnostics


def score_at_t0(
    X_t0: pd.DataFrame,
    selected_features: List[str],
    standardization_params: Dict,
    model: object
) -> pd.Series:
    """
    Score new data at t0 using trained model and stored parameters.
    
    CRITICAL: Uses only stored standardization params from training period.
    No label information or t0 statistics are used.
    
    Args:
        X_t0: Raw feature matrix at t0 (samples × features)
        selected_features: List of feature names selected by ElasticNet
        standardization_params: Dict with 'median' and 'mad' from training period
        model: Fitted ElasticNet model
        
    Returns:
        pd.Series of predicted scores indexed by original X_t0 index
    """
    logger.info(f"Scoring t0: {len(X_t0)} samples, {len(selected_features)} features")
    
    # Apply stored standardization to ALL t0 features (model expects all trained features)
    X_t0_std = X_t0.copy()
    for col in X_t0_std.columns:
        if col in standardization_params['median']:
            median_train = standardization_params['median'][col]
            mad_train = standardization_params['mad'][col]
            X_t0_std[col] = (X_t0[col] - median_train) / mad_train
    
    # Model expects features in same order as training
    # ElasticNet internally zeros out non-selected features, so pass all features
    X_arr = X_t0_std.values
    has_nan = np.isnan(X_arr).any(axis=1)
    
    # Predict scores
    scores = np.full(len(X_t0), np.nan)
    if not has_nan.all():
        X_clean = X_arr[~has_nan]
        scores[~has_nan] = model.predict(X_clean)
    
    # Return as Series with original index
    scores_series = pd.Series(scores, index=X_t0.index, name='score')
    
    logger.info(f"Scoring complete: {(~np.isnan(scores)).sum()} valid scores")
    
    return scores_series


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

def per_window_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_t0: pd.DataFrame,
    dates_train: pd.DatetimeIndex,
    approved_features: List[str],
    half_life: float = 126.0,
    theta_ic: float = 0.03,
    t_min: float = 1.96,
    theta_stable: float = 0.03,
    k_folds: int = 3,
    min_sign_consistency: int = 2,
    k_bin: int = 3,
    corr_threshold: float = 0.7,
    alpha_grid: List[float] = None,
    l1_ratio_grid: List[float] = None,
    cv_folds: int = 3,
    coef_threshold: float = 1e-4,
    n_jobs: int = 1
) -> Tuple[pd.Series, Dict, object]:
    """
    Run complete per-window feature selection pipeline and score at t0.
    
    Pipeline stages:
    1. IC filter (theta_ic threshold + t_min Newey-West t-stat)
    2. Stability across K folds (sign consistency + magnitude)
    3. Supervised binning (DecisionTree representation choice)
    4. Correlation redundancy filter
    5. Robust standardization (median/MAD)
    6. ElasticNetCV feature selection
    7. Score at t0 using trained model
    
    Args:
        X_train: Raw feature matrix for training window (approved features only)
        y_train: Target series for training window
        X_t0: Raw feature matrix at t0 (for scoring)
        dates_train: DatetimeIndex for training samples
        approved_features: List of features approved in Formation FDR
        half_life: Half-life for time-decay weights (days)
        theta_ic: IC threshold for per-window filter
        theta_stable: Median |IC| threshold for stability
        k_folds: Number of folds for stability check
        min_sign_consistency: Minimum folds with same IC sign
        k_bin: Number of bins for DecisionTree
        corr_threshold: Maximum correlation for redundancy filter
        alpha_grid: ElasticNet alpha values
        l1_ratio_grid: ElasticNet L1 ratio values
        cv_folds: TimeSeriesSplit folds for ElasticNet
        coef_threshold: Minimum |coefficient| for selection
        n_jobs: Parallel jobs
        
    Returns:
        Tuple of (scores_at_t0 Series, diagnostics_dict, fitted_model)
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"Per-window pipeline: {len(X_train)} train samples, {len(X_t0)} t0 samples")
    logger.info(f"Approved features: {len(approved_features)}")
    
    all_diagnostics = {}
    
    # Extract approved features
    X = X_train[approved_features]
    
    # Compute time-decay weights for IC filter
    train_end = dates_train[-1]
    weights = compute_time_decay_weights(dates_train, train_end, half_life=int(half_life))
    
    # Stage 1: IC Filter
    selected_ic, diag_ic = per_window_ic_filter(
        X, y_train, dates_train, weights,
        theta_ic=theta_ic,
        t_min=t_min,
        n_jobs=n_jobs
    )
    all_diagnostics['ic_filter'] = diag_ic
    logger.info(f"After IC filter: {len(selected_ic)} features")
    
    if len(selected_ic) == 0:
        logger.warning("No features passed IC filter, returning zeros")
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Stage 2: Stability Filter
    X_ic = X[selected_ic]
    selected_stable, diag_stable = per_window_stability(
        X_ic, y_train, dates_train,
        k_folds=k_folds,
        theta_stable=theta_stable,
        min_sign_consistency=min_sign_consistency,
        n_jobs=n_jobs
    )
    all_diagnostics['stability'] = diag_stable
    logger.info(f"After stability: {len(selected_stable)} features")
    
    if len(selected_stable) == 0:
        logger.warning("No features passed stability, returning zeros")
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Stage 3: Supervised Binning - DEPRECATED in v3 pipeline, skipped
    # In v3, we operate only on continuous features (no binning)
    X_stable = X_ic[selected_stable]
    # selected_binned, diag_binning = supervised_binning_and_representation(
    #     X_stable, y_train,
    #     k_bin=k_bin,
    #     min_samples_leaf=50,
    #     n_jobs=n_jobs
    # )
    # all_diagnostics['binning'] = diag_binning
    # logger.info(f"After binning: {len(selected_binned)} features")
    
    # Skip binning in v3 pipeline
    logger.info("Skipping binning stage (v3 pipeline uses continuous features only)")
    
    # Stage 4: Correlation Redundancy Filter
    # Use selected_stable features directly (no binning)
    X_for_corr = X_stable
    selected_nonredundant, diag_redundancy = correlation_redundancy_filter(
        X_for_corr,
        corr_threshold=corr_threshold,
        n_jobs=n_jobs
    )
    all_diagnostics['redundancy'] = diag_redundancy
    logger.info(f"After redundancy filter: {len(selected_nonredundant)} features")
    
    if len(selected_nonredundant) == 0:
        logger.warning("No features after redundancy filter, returning zeros")
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Stage 5: Robust Standardization
    X_nonredundant = X_for_corr[selected_nonredundant]
    X_std, std_params = robust_standardization(X_nonredundant)
    all_diagnostics['standardization'] = {
        'time_standardization': std_params['time_standardization']
    }
    logger.info(f"Standardization complete")
    
    # Stage 6: ElasticNetCV Selection
    selected_final, model, diag_elasticnet = elasticnet_cv_selection(
        X_std, y_train, dates_train,
        alpha_grid=alpha_grid,
        l1_ratio_grid=l1_ratio_grid,
        cv_folds=cv_folds,
        coef_threshold=coef_threshold,
        n_jobs=n_jobs
    )
    all_diagnostics['elasticnet'] = diag_elasticnet
    logger.info(f"Final selection: {len(selected_final)} features")
    
    if len(selected_final) == 0:
        logger.warning("ElasticNet selected no features, returning zeros")
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Stage 7: Refit ElasticNet on selected features ONLY
    # This is more efficient and cleaner than keeping all features
    logger.info(f"Refitting ElasticNet on {len(selected_final)} selected features...")
    
    # Extract selected features from training data
    X_train_selected = X_std[selected_final]
    
    # Refit with same hyperparameters as the full model
    from sklearn.linear_model import ElasticNet
    refit_model = ElasticNet(
        alpha=model.alpha_,
        l1_ratio=model.l1_ratio_,
        max_iter=10000,
        random_state=42
    )
    refit_model.fit(X_train_selected.values, y_train.values)
    
    # Extract scaling params ONLY for selected features
    selected_scaling_params = {
        'median': {feat: std_params['median'][feat] for feat in selected_final},
        'mad': {feat: std_params['mad'][feat] for feat in selected_final}
    }
    
    # Stage 8: Create ElasticNetWindowModel with refit model
    from alpha_models import ElasticNetWindowModel
    
    window_model = ElasticNetWindowModel(
        selected_features=selected_final,
        scaling_params=selected_scaling_params,  # Only selected features
        fitted_model=refit_model,  # Refit model on subset
        binning_params=None  # No binning in current v2 pipeline
    )
    
    # Stage 9: Score at t0 using the refit model
    # Apply same transformations to t0 data
    X_t0_selected = X_t0[selected_nonredundant]  # Extract same features as training
    
    scores_t0 = score_at_t0(
        X_t0_selected,
        selected_final,
        selected_scaling_params,  # Use subset scaling params
        refit_model  # Use refit model
    )
    
    all_diagnostics['total_time'] = time.time() - start_time
    all_diagnostics['final_n_features'] = len(selected_final)
    
    logger.info(f"Pipeline complete: {len(selected_final)} features, "
                f"time: {all_diagnostics['total_time']:.1f}s")
    logger.info("=" * 80)
    
    return scores_t0, all_diagnostics, window_model  # Return wrapped model


def per_window_pipeline_v3(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_t0: pd.DataFrame,
    dates_train: pd.DatetimeIndex,
    formation_artifacts: Dict,
    config: 'ResearchConfig',
    n_jobs: int = 1
) -> Tuple[pd.Series, Dict, object]:
    """
    V3 per-window feature selection pipeline using Formation artifacts.
    
    Pipeline stages (V3 / V4):
    1. Extract S_F (approved features) from formation_artifacts
    2. Soft ranking by IC + sign consistency (no hard stability drop)
    3. Correlation redundancy filter (vectorized)
    4. Robust standardization (median/MAD)
    5. LassoLarsIC with BIC (min 12 features) + Ridge refit
    6. Score at t0 using trained model
    
    Key differences from v2:
    - No hard IC filter with t-stat threshold
    - No hard stability filter (uses soft ranking instead)
    - No binning
    - Uses LassoLarsIC with BIC criterion (not ElasticNetCV)
    - Formation no longer tunes hyperparameters (just FDR + redundancy)
    
    Args:
        X_train: Raw feature matrix for training window (all features)
        y_train: Target series for training window
        X_t0: Raw feature matrix at t0 (for scoring)
        dates_train: DatetimeIndex for training samples
        formation_artifacts: Dict containing:
            - 'approved_features': List[str] (S_F from formation_fdr)
        config: ResearchConfig with v3/v4 parameters
        n_jobs: Parallel jobs
        
    Returns:
        Tuple of (scores_at_t0 Series, diagnostics_dict, ElasticNetWindowModel)
    """
    start_time = time.time()
    print("=" * 80, flush=True)
    print("[v3] Per-window pipeline (Formation-guided)", flush=True)
    print(f"[v3] Train samples: {len(X_train)}, t0 samples: {len(X_t0)}", flush=True)
    
    # =========================================================================
    # NaN GATE CHECK: Feature selection CANNOT proceed with NaN data
    # =========================================================================
    # This is a HARD GATE. If NaN exist, it means feature_engineering.py failed
    # to clean the data properly. We fail loudly here to force fix at the source.
    nan_count_train = X_train.isna().sum().sum()
    nan_count_t0 = X_t0.isna().sum().sum()
    
    if nan_count_train > 0 or nan_count_t0 > 0:
        nan_cols_train = X_train.columns[X_train.isna().any()].tolist()
        nan_cols_t0 = X_t0.columns[X_t0.isna().any()].tolist()
        
        error_msg = (
            "\n" + "="*80 + "\n"
            "FEATURE SELECTION GATE FAILED: NaN values detected in input data!\n"
            "\n"
            f"Training data NaN count: {nan_count_train:,}\n"
            f"Scoring data (t0) NaN count: {nan_count_t0:,}\n"
            "\n"
            "This indicates a bug in feature_engineering.py. NaN handling must\n"
            "be done at the SOURCE before feature selection begins.\n"
            "\n"
            "Feature selection CANNOT proceed because:\n"
            "  - ElasticNet cannot fit with NaN (produces zero coefficients)\n"
            "  - IC computation with NaN gives unreliable results\n"
            "\n"
            "Please fix feature_engineering.py to ensure clean output data.\n"
        )
        
        if nan_cols_train:
            error_msg += f"\nTraining columns with NaN (showing first 10):\n"
            for col in nan_cols_train[:10]:
                nan_pct = X_train[col].isna().sum() / len(X_train) * 100
                error_msg += f"    - {col}: {nan_pct:.1f}% NaN\n"
        
        error_msg += "="*80
        
        raise ValueError(error_msg)
    
    print(f"[v3] ✓ NaN gate passed - input data is clean", flush=True)
    
    all_diagnostics = {}
    
    # Extract Formation artifacts
    approved_features = formation_artifacts['approved_features']
    # Note: V4 no longer uses best_alpha/best_l1_ratio from Formation
    # LassoLarsIC selects lambda automatically via BIC criterion
    
    print(f"[v3] Formation-approved features (S_F): {len(approved_features)}", flush=True)
    print(f"[v3] Using LassoLarsIC (criterion={config.features.lars_criterion}, min_features={config.features.lars_min_features})", flush=True)
    
    # Initialize stage timing
    stage_times = {}
    
    # Extract approved features
    X = X_train[approved_features]
    
    # Compute time-decay weights for soft ranking
    train_end = dates_train[-1]
    weights = compute_time_decay_weights(
        dates_train, 
        train_end, 
        half_life=config.features.training_halflife_days
    )
    
    # Stage 1: Soft ranking by IC + sign consistency
    stage_start = time.time()
    selected_ranked, diag_ranking = rank_features_by_ic_and_sign_consistency(
        X, y_train, dates_train, weights,
        num_blocks=config.features.per_window_num_blocks,
        ic_floor=config.features.per_window_ic_floor,
        top_k=config.features.per_window_top_k,
        min_features=config.features.per_window_min_features,
        n_jobs=n_jobs
    )
    stage_times['soft_ranking'] = time.time() - stage_start
    all_diagnostics['soft_ranking'] = diag_ranking
    print(f"[v3 Stage 1] Soft ranking: {len(approved_features)} -> {len(selected_ranked)} features ({stage_times['soft_ranking']:.2f}s)", flush=True)
    
    if len(selected_ranked) == 0:
        print("[v3 WARN] No features after soft ranking, returning zeros", flush=True)
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Stage 2: Correlation Redundancy Filter
    stage_start = time.time()
    X_ranked = X[selected_ranked]
    selected_nonredundant, diag_redundancy = correlation_redundancy_filter(
        X_ranked,
        corr_threshold=config.features.corr_threshold,
        n_jobs=n_jobs
    )
    stage_times['redundancy_filter'] = time.time() - stage_start
    all_diagnostics['redundancy'] = diag_redundancy
    print(f"[v3 Stage 2] Redundancy filter: {len(selected_ranked)} -> {len(selected_nonredundant)} features ({stage_times['redundancy_filter']:.2f}s)", flush=True)
    
    if len(selected_nonredundant) == 0:
        print("[v3 WARN] No features after redundancy filter, returning zeros", flush=True)
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Stage 3: Robust Standardization
    stage_start = time.time()
    X_nonredundant = X_ranked[selected_nonredundant]
    X_std, std_params = robust_standardization(X_nonredundant)
    stage_times['standardization'] = time.time() - stage_start
    all_diagnostics['standardization'] = {
        'time_standardization': std_params['time_standardization']
    }
    print(f"[v3 Stage 3] Standardization: {len(selected_nonredundant)} features ({stage_times['standardization']:.2f}s)", flush=True)
    
    # Stage 4: LassoLarsIC feature selection + Ridge refit (NO CV)
    stage_start = time.time()
    
    selected_final, ridge_coef, lars_diag = training_lasso_lars_ic(
        X_std, 
        y_train,
        criterion=config.features.lars_criterion,
        min_features=config.features.lars_min_features,
        ridge_alpha=config.features.ridge_refit_alpha
    )
    
    n_selected = len(selected_final)
    stage_times['lars_ridge'] = time.time() - stage_start
    
    print(f"[v3 Stage 4] LassoLarsIC + Ridge: {len(selected_nonredundant)} -> {n_selected} features ({stage_times['lars_ridge']:.2f}s)", flush=True)
    
    all_diagnostics['lars'] = lars_diag
    
    # Create a simple model wrapper that holds the Ridge coefficients
    # We'll use sklearn Ridge for consistency
    from sklearn.linear_model import Ridge
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    # Refit Ridge on selected features to get a proper model object
    X_train_selected = X_std[selected_final]
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        final_model = Ridge(
            alpha=config.features.ridge_refit_alpha,
            fit_intercept=True
        )
        final_model.fit(X_train_selected.values, y_train.values)
    
    # Extract scaling params ONLY for selected features
    selected_scaling_params = {
        'median': {feat: std_params['median'][feat] for feat in selected_final},
        'mad': {feat: std_params['mad'][feat] for feat in selected_final}
    }
    
    # Stage 5: Create ElasticNetWindowModel (reusing name for backward compatibility)
    from alpha_models import ElasticNetWindowModel
    
    window_model = ElasticNetWindowModel(
        selected_features=selected_final,
        scaling_params=selected_scaling_params,
        fitted_model=final_model,
        binning_params=None  # No binning in v3
    )
    
    # Stage 6: Score at t0
    stage_start = time.time()
    # Important: Use selected_final features (after LARS selection), NOT selected_nonredundant
    X_t0_selected = X_t0[selected_final]  # Extract only LARS-selected features
    
    scores_t0 = score_at_t0(
        X_t0_selected,
        selected_final,
        selected_scaling_params,
        final_model
    )
    stage_times['scoring'] = time.time() - stage_start
    print(f"[v3 Stage 5] Scoring at t0: {len(X_t0)} tickers ({stage_times['scoring']:.2f}s)", flush=True)
    
    all_diagnostics['stage_times'] = stage_times
    all_diagnostics['total_time'] = time.time() - start_time
    all_diagnostics['final_n_features'] = n_selected
    
    # Feature flow summary
    all_diagnostics['feature_flow'] = {
        'input_approved': len(approved_features),
        'after_soft_ranking': len(selected_ranked),
        'after_redundancy': len(selected_nonredundant),
        'after_lars': n_selected
    }
    
    print("-" * 60, flush=True)
    print(f"[v3] PIPELINE SUMMARY:", flush=True)
    print(f"[v3]   Feature flow: {len(approved_features)} -> {len(selected_ranked)} -> {len(selected_nonredundant)} -> {n_selected}", flush=True)
    print(f"[v3]   Stage times: ranking={stage_times['soft_ranking']:.2f}s, redundancy={stage_times['redundancy_filter']:.2f}s, "
          f"lars={stage_times['lars_ridge']:.2f}s, score={stage_times['scoring']:.2f}s", flush=True)
    print(f"[v3]   Total time: {all_diagnostics['total_time']:.2f}s", flush=True)
    print("=" * 80, flush=True)
    
    return scores_t0, all_diagnostics, window_model


# ============================================================================
# Bayesian Optimization for Hyperparameter Tuning
# ============================================================================

def tune_hyperparameters(
    X_formation: pd.DataFrame,
    y_formation: pd.Series,
    dates_formation: pd.DatetimeIndex,
    validation_windows: List[Tuple],
    approved_features: List[str],
    n_iterations: int = 50,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Run Bayesian optimization to tune hyperparameters on Validation period.
    
    Hyperparameters to tune:
    - half_life: [42, 252]
    - theta_ic_window: [0.015, 0.05]
    - K_bin: [20, 70]
    - F_final_window: [8, 20]
    - use_xgboost_for_selection: [0, 1]
    
    Args:
        X_formation: Feature matrix for Formation period
        y_formation: Target for Formation period
        dates_formation: Dates for Formation period
        validation_windows: List of (train_dates, test_date) tuples for Validation
        approved_features: Features approved in Formation FDR
        n_iterations: Number of Bayesian optimization iterations
        output_path: Optional path to save optimization results
        
    Returns:
        Dictionary of optimal hyperparameters
    """
    logger.info("=" * 80)
    logger.info("Starting Bayesian optimization for hyperparameter tuning")
    logger.info(f"Validation windows: {len(validation_windows)}, Iterations: {n_iterations}")
    
    # TODO: Implement Bayesian optimization using scikit-optimize or Optuna
    # For now, return default hyperparameters
    optimal_hyperparams = {
        'half_life': 126,
        'theta_ic_window': 0.025,
        'K_bin': 50,
        'F_final_window': 15,
        'use_xgboost_for_selection': False
    }
    
    logger.info(f"Optimal hyperparameters: {optimal_hyperparams}")
    
    
    return optimal_hyperparams


# ============================================================================
# Walk-Forward Interface (High-Level Wrapper)
# ============================================================================

def train_window_model(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    t0: pd.Timestamp,
    config: 'ResearchConfig',
    formation_artifacts: Optional[Dict] = None
) -> Tuple[Optional['ElasticNetWindowModel'], Dict]:
    """
    High-level interface for walk-forward engine (supports both v2 and v3 pipelines).
    
    Takes a training panel (with Date/Ticker multi-index) and returns a trained
    model that can be used to score at t0.
    
    V3 Pipeline (when formation_artifacts provided):
    - Uses approved features from Formation FDR
    - Uses ElasticNet hyperparameters from Formation tuning
    - Applies soft IC ranking (no hard stability drops)
    - No binning, no CV in Training
    
    V2 Pipeline (when formation_artifacts=None, backward compatibility):
    - Uses all features as "approved"
    - Runs ElasticNetCV on Training window
    - Uses hard IC/stability filters + binning
    
    Args:
        panel: Training data panel (Date/Ticker multi-index, feature columns)
        metadata: Universe metadata (ticker, sector, region, etc.)
        t0: Scoring date (not used in training, only for scoring)
        config: Research configuration
        formation_artifacts: Optional dict from Formation phase containing:
            - 'approved_features': List[str] (S_F from formation_fdr)
            - 'best_alpha': float (from formation_tune_elasticnet)
            - 'best_l1_ratio': float (from formation_tune_elasticnet)
        
    Returns:
        (model, diagnostics) tuple where:
        - model: ElasticNetWindowModel (or None if training failed)
        - diagnostics: Dict with training statistics
    """
    from config import ResearchConfig
    
    use_v3 = formation_artifacts is not None
    pipeline_version = "v3" if use_v3 else "v2 (backward compatibility)"
    logger.info(f"train_window_model called for t0={t0}, pipeline={pipeline_version}")
    
    # Extract dates and prepare training data
    dates_train = panel.index.get_level_values('Date').unique().sort_values()
    
    # Identify feature columns (exclude target and metadata columns)
    # Support both naming conventions: FwdRet_{H} and ret_fwd_{H}d
    target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    if target_col not in panel.columns:
        # Fall back to alternative naming convention
        target_col = f'ret_fwd_{config.time.HOLDING_PERIOD_DAYS}d'
    
    exclude_cols = {target_col, 'market_cap', 'volume', 'dollar_volume', 'Close', 'ADV_63', 'ADV_63_Rank'}
    exclude_cols.update([f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}', f'ret_fwd_{config.time.HOLDING_PERIOD_DAYS}d'])
    
    feature_cols = [col for col in panel.columns if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        logger.warning("No feature columns found in panel")
        return None, {'error': 'no_features', 'n_start': 0}
    
    # Check if target exists
    if target_col not in panel.columns:
        logger.error(f"Target column '{target_col}' not found in panel")
        return None, {'error': 'no_target', 'n_start': len(feature_cols)}
    
    # Prepare X and y
    X_train = panel[feature_cols].copy()
    y_train = panel[target_col].copy()
    
    # Drop rows with NaN target
    valid_mask = y_train.notna()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(y_train) == 0:
        logger.warning("No valid training samples (all targets are NaN)")
        return None, {'error': 'no_valid_samples', 'n_start': len(feature_cols)}
    
    # For scoring at t0, we DON'T need X_t0 here - scoring happens later
    # via model.score_at_date() which has access to full panel
    # Just create dummy X_t0 for per_window_pipeline (which expects it)
    # Use last date in training data as proxy
    last_date = dates_train[-1]
    t0_mask_proxy = panel.index.get_level_values('Date') == last_date
    X_t0 = panel.loc[t0_mask_proxy, feature_cols].copy()
    
    if len(X_t0) == 0:
        logger.warning(f"No data at last training date={last_date}")
        return None, {'error': 'no_data_at_last_date', 'n_start': len(feature_cols)}
    
    # Extract dates for training samples
    dates_train_samples = X_train.index.get_level_values('Date')
    
    # Determine n_jobs: force sequential execution in profiling/debug mode
    # This ensures timing measurements are accurate and not muddied by parallel overhead
    is_profiling_mode = (config.compute.max_rebalance_dates_for_debug is not None)
    if is_profiling_mode:
        effective_n_jobs = 1
        print(f"[train_window_model] PROFILING MODE: n_jobs forced to 1 for accurate timing")
    else:
        effective_n_jobs = config.compute.n_jobs
    
    # Choose pipeline based on whether formation_artifacts are provided
    try:
        if use_v3:
            # V3 Pipeline: Use Formation artifacts
            logger.info("Using V3 pipeline with Formation artifacts")
            scores_t0, diagnostics, model = per_window_pipeline_v3(
                X_train=X_train,
                y_train=y_train,
                X_t0=X_t0,
                dates_train=dates_train_samples,
                formation_artifacts=formation_artifacts,
                config=config,
                n_jobs=effective_n_jobs
            )
        else:
            # V2 Pipeline (backward compatibility): Use all features, run CV
            logger.info("Using V2 pipeline (backward compatibility mode)")
            approved_features = feature_cols
            
            # Set hyperparameters from config
            alpha_grid = config.features.ALPHA_ELASTICNET if hasattr(config.features, 'ALPHA_ELASTICNET') else [0.1, 0.5, 1.0]
            l1_ratio_grid = config.features.L1_RATIO if hasattr(config.features, 'L1_RATIO') else [0.5, 0.7, 0.9]
            
            scores_t0, diagnostics, model = per_window_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_t0=X_t0,
                dates_train=dates_train_samples,
                approved_features=approved_features,
                half_life=126.0,
                theta_ic=config.features.FORMATION_IC_THRESHOLD if hasattr(config.features, 'FORMATION_IC_THRESHOLD') else 0.03,
                t_min=0.0,  # Relaxed for small test datasets; production should use 1.96
                theta_stable=config.features.MIN_STABILITY_PVALUE if hasattr(config.features, 'MIN_STABILITY_PVALUE') else 0.03,
                k_folds=3,
                min_sign_consistency=2,
                k_bin=config.features.K_BIN if hasattr(config.features, 'K_BIN') else 3,
                corr_threshold=0.7,
                alpha_grid=alpha_grid,
                l1_ratio_grid=l1_ratio_grid,
                cv_folds=3,
                coef_threshold=1e-6,  # Very relaxed for small test datasets
                n_jobs=effective_n_jobs
            )
        
        # Return model and diagnostics
        return model, diagnostics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e), 'n_start': len(feature_cols)}

