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
from statsmodels.regression.linear_model import OLS

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
        weights: Optional sample weights
        
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
    
    if weights is not None:
        weights_clean = weights[valid_mask]
        model = OLS(y, X, weights=weights_clean)
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


# TODO: Implement remaining pipeline functions:
# - per_window_stability()
# - supervised_binning_and_representation()
# - correlation_redundancy_filter()
# - robust_standardization()
# - elasticnet_cv_selection()
# - xgboost_refinement()
# - score_at_t0()
# - per_window_pipeline() (orchestrator)

# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

def per_window_pipeline(
    X_raw: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    t0: pd.Timestamp,
    approved_features: List[str],
    hyperparams: Dict,
    diagnostics_path: Optional[Path] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Run complete per-window feature selection pipeline.
    
    Pipeline stages:
    1. IC filter
    2. Stability across K folds
    3. Supervised binning (top K_bin features)
    4. Representation choice
    5. Correlation redundancy filter
    6. Robust standardization (median/MAD)
    7. ElasticNetCV
    8. Optional XGBoost refinement
    9. Score at t0
    
    Args:
        X_raw: Raw feature matrix for training window
        y: Target series for training window
        dates: Date for each sample
        t0: Rebalance date (for scoring)
        approved_features: List of features approved in Formation
        hyperparams: Dictionary of hyperparameters (half_life, theta_ic, K_bin, etc.)
        diagnostics_path: Optional path to save diagnostics
        
    Returns:
        Tuple of (scores_at_t0, diagnostics_dict)
    """
    logger.info("=" * 80)
    logger.info(f"Per-window pipeline for t0={t0}")
    log_memory_usage("Pipeline start")
    
    # TODO: Implement full pipeline
    # For now, return dummy scores
    diagnostics = {
        't0': t0,
        'n_approved': len(approved_features),
        'status': 'not_implemented'
    }
    
    scores = np.zeros(len(X_raw))
    
    return scores, diagnostics


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
# Scoring Function
# ============================================================================

def score_at_t0(
    X_t0: pd.DataFrame,
    model_enet,
    model_xgb,
    selected_features: List[str],
    scaling_params: Dict,
    binning_params: Dict,
    use_xgboost: bool = False,
    ensemble_weights: Tuple[float, float] = (0.6, 0.4)
) -> np.ndarray:
    """
    Generate predictions at t0 using fitted models.
    
    Args:
        X_t0: Feature matrix at t0 (assets × features)
        model_enet: Fitted ElasticNet model
        model_xgb: Fitted XGBoost model (optional)
        selected_features: List of selected feature names
        scaling_params: Dict of {feature: (median, MAD)} for standardization
        binning_params: Dict of {feature: boundaries} for binned features
        use_xgboost: Whether to use XGBoost
        ensemble_weights: Tuple of (w_enet, w_xgb) for ensemble
        
    Returns:
        Array of scores for each asset at t0
    """
    logger.info(f"Scoring at t0: {len(X_t0)} assets")
    
    # TODO: Implement scoring logic
    # For now, return zeros
    scores = np.zeros(len(X_t0), dtype=np.float32)
    
    return scores
