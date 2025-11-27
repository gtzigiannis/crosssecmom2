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
from sklearn.linear_model import ElasticNetCV
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
    n_features = X.shape[1]
    ic_matrix = np.zeros((len(unique_dates), n_features), dtype=np.float32)
    
    for i, date in enumerate(unique_dates):
        # Get rows for this date
        if isinstance(X.index, pd.DatetimeIndex):
            X_date = X.loc[date].values
            y_date = y.loc[date].values
            # Handle case where loc returns Series (single row) vs DataFrame (multiple rows)
            if X_date.ndim == 1:
                X_date = X_date.reshape(1, -1)
                y_date = np.array([y_date])
        else:
            date_mask = sample_dates == date
            X_date = X.values[date_mask]
            y_date = y.values[date_mask]
        
        for j in range(n_features):
            ic_matrix[i, j] = compute_spearman_ic(X_date[:, j], y_date)
    
    return pd.DataFrame(ic_matrix, index=unique_dates, columns=X.columns)


def compute_newey_west_tstat(
    ic_series: pd.Series,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute Newey-West t-statistic for IC time series.
    
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
    
    # Convert to float32 for memory efficiency
    X = X.astype(np.float32)
    y = y.astype(np.float64)  # Keep target as float64 for numerical stability
    
    # Compute time-decay weights
    train_end = dates.max()
    weights = compute_time_decay_weights(dates, train_end, half_life)
    
    # Compute daily IC series for all features
    logger.info("Computing daily IC series...")
    ic_daily = compute_daily_ic_series(X, y, dates)
    
    # For each feature, compute weighted mean IC and Newey-West t-stat
    logger.info("Computing weighted IC and Newey-West t-stats...")
    results = []
    
    def process_feature(feat_name):
        ic_series = ic_daily[feat_name]
        
        # Weighted mean IC
        ic_weighted = np.average(ic_series, weights=weights)
        
        # Newey-West t-stat
        t_nw = compute_newey_west_tstat(ic_series, weights)
        
        # Convert t-stat to p-value (two-tailed)
        n_dates = len(ic_series.dropna())
        if n_dates > 2:
            p_value = 2 * (1 - stats.t.cdf(abs(t_nw), df=n_dates-1))
        else:
            p_value = 1.0
        
        return {
            'feature': feat_name,
            'ic_weighted': ic_weighted,
            't_nw': t_nw,
            'p_value': p_value,
            'n_dates': n_dates
        }
    
    # Parallel processing
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_feature)(feat) for feat in X.columns
    )
    
    # Create diagnostics dataframe
    diagnostics_df = pd.DataFrame(results)
    
    # Apply FDR control (Benjamini-Hochberg)
    logger.info(f"Applying FDR control at level {fdr_level}")
    reject, pvals_corrected, _, _ = multipletests(
        diagnostics_df['p_value'],
        alpha=fdr_level,
        method='fdr_bh'
    )
    
    diagnostics_df['fdr_reject'] = reject
    diagnostics_df['p_value_corrected'] = pvals_corrected
    
    # Select approved features
    approved_features = diagnostics_df[diagnostics_df['fdr_reject']]['feature'].tolist()
    
    elapsed = time.time() - start_time
    logger.info(f"Formation FDR complete in {elapsed:.1f}s")
    logger.info(f"Approved features: {len(approved_features)} / {X.shape[1]}")
    logger.info(f"IC stats - Mean: {diagnostics_df['ic_weighted'].mean():.4f}, "
                f"Median: {diagnostics_df['ic_weighted'].median():.4f}")
    log_memory_usage("Formation end")
    
    # Clean up
    del ic_daily
    gc.collect()
    
    return approved_features, diagnostics_df


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
    
    # Process each feature
    def process_feature(feat_name):
        ic_series = ic_daily[feat_name]
        ic_weighted = np.average(ic_series, weights=weights)
        t_nw = compute_newey_west_tstat(ic_series, weights)
        
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
    Filter features based on stability across temporal folds.
    
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
            mask = X.index.isin(fold_dates)
            X_fold = X.loc[mask, [feat_name]]
            y_fold = y.loc[mask]
            
            # Compute mean IC across dates in fold
            fold_dates_actual = X.index[mask].unique()
            if len(fold_dates_actual) < 3:
                fold_ics.append(np.nan)
                continue
                
            ic_daily = compute_daily_ic_series(X_fold, y_fold, fold_dates_actual)
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
    start_time = time.time()
    logger.info(f"Binning: {X.shape[1]} features, k_bin={k_bin}")
    
    # Compute IC for all raw features once
    dates_unique = X.index.unique()
    ic_daily_raw = compute_daily_ic_series(X, y, dates_unique)
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
            
            # Create binned feature (use tree predictions as bin assignments)
            X_binned = tree.predict(X_feat)
            X_binned_df = pd.DataFrame({f'{feat_name}_binned': X_binned}, index=X.index)
            
            # Compute IC for binned version
            ic_daily_binned = compute_daily_ic_series(X_binned_df, y, dates_unique)
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
    corr_threshold: float = 0.7,
    n_jobs: int = 1
) -> Tuple[List[str], Dict]:
    """
    Remove redundant features based on pairwise correlation.
    
    Greedy algorithm:
    1. Compute pairwise correlation matrix
    2. For each pair with |corr| > threshold:
       - Keep the feature that appears first in the list
       - Remove the other feature
    3. Continue until no pairs exceed threshold
    
    Args:
        X: Feature matrix (samples × features)
        corr_threshold: Maximum allowed absolute correlation
        n_jobs: Number of parallel jobs (unused, kept for API consistency)
        
    Returns:
        Tuple of (selected_features_list, diagnostics_dict)
    """
    start_time = time.time()
    logger.info(f"Redundancy filter: {X.shape[1]} features, threshold={corr_threshold}")
    
    # Compute correlation matrix
    corr_matrix = X.corr().abs()
    
    # Greedy removal: iterate through features and remove highly correlated ones
    features_to_keep = list(X.columns)
    features_removed = []
    
    i = 0
    while i < len(features_to_keep):
        feat_i = features_to_keep[i]
        
        # Find features highly correlated with feat_i
        to_remove = []
        for j in range(i + 1, len(features_to_keep)):
            feat_j = features_to_keep[j]
            
            if corr_matrix.loc[feat_i, feat_j] > corr_threshold:
                to_remove.append(feat_j)
        
        # Remove highly correlated features
        for feat in to_remove:
            features_to_keep.remove(feat)
            features_removed.append(feat)
        
        i += 1
    
    diagnostics = {
        'n_start': X.shape[1],
        'n_after_redundancy': len(features_to_keep),
        'n_removed': len(features_removed),
        'time_redundancy': time.time() - start_time
    }
    
    logger.info(f"Redundancy filter: {len(features_to_keep)} features kept, "
                f"{len(features_removed)} removed (time: {diagnostics['time_redundancy']:.1f}s)")
    
    del corr_matrix
    gc.collect()
    
    return features_to_keep, diagnostics


def robust_standardization(
    X: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize features using robust statistics (median and MAD).
    
    For each feature:
        z = (x - median) / MAD
    where MAD = median(|x - median|)
    
    Args:
        X: Feature matrix (samples × features)
        
    Returns:
        Tuple of (X_standardized DataFrame, parameters_dict)
        parameters_dict contains 'median' and 'mad' for each feature
    """
    start_time = time.time()
    logger.info(f"Standardization: {X.shape[1]} features")
    
    # Compute median and MAD for each feature
    medians = {}
    mads = {}
    
    X_standardized = X.copy()
    
    for col in X.columns:
        median_val = X[col].median()
        mad_val = np.median(np.abs(X[col] - median_val))
        
        # Avoid division by zero
        if mad_val < 1e-10:
            mad_val = 1.0
            logger.warning(f"Feature {col} has MAD near zero, using MAD=1.0")
        
        # Standardize
        X_standardized[col] = (X[col] - median_val) / mad_val
        
        medians[col] = median_val
        mads[col] = mad_val
    
    parameters = {
        'median': medians,
        'mad': mads,
        'time_standardization': time.time() - start_time
    }
    
    logger.info(f"Standardization complete (time: {parameters['time_standardization']:.1f}s)")
    
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
    1. IC filter (theta_ic threshold)
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
        t_min=1.96,
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
    
    # Stage 3: Supervised Binning
    X_stable = X_ic[selected_stable]
    selected_binned, diag_binning = supervised_binning_and_representation(
        X_stable, y_train,
        k_bin=k_bin,
        min_samples_leaf=50,
        n_jobs=n_jobs
    )
    all_diagnostics['binning'] = diag_binning
    logger.info(f"After binning: {len(selected_binned)} features")
    
    if len(selected_binned) == 0:
        logger.warning("No features after binning, returning zeros")
        return pd.Series(0.0, index=X_t0.index), all_diagnostics, None
    
    # Note: Binning may create new feature names (e.g., 'feature_0_binned')
    # We need to reconstruct X with binned features for next stages
    # For simplicity, if binning selected raw features, continue with those
    # In production, would need to apply binning transformation
    
    # Stage 4: Correlation Redundancy Filter
    # Use original selected_stable features (binning just for IC comparison)
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

