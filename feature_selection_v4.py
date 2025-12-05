"""
Feature Selection Module for crosssecmom2

Implements the V3 feature selection pipeline:
- Formation: Global FDR on raw features with interaction screening
- Per-window pipeline: Soft IC ranking → redundancy → LassoCV/LassoLarsIC → Ridge refit
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
from scipy.stats import rankdata as scipy_rankdata  # Fast Spearman approximation
from sklearn.linear_model import ElasticNet, LassoLarsIC, Ridge
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac

logger = logging.getLogger(__name__)


# ============================================================================
# Formation Cache (for reusing Formation computation across windows)
# ============================================================================

class FormationCache:
    """
    Singleton cache for Formation-level statistics.
    
    PERFORMANCE OPTIMIZATION: Caches expensive computations that can be reused
    across walk-forward windows sharing the same formation period.
    
    Caches:
    - Daily IC series for all features (base + interactions)
    - Base feature stats for parent pre-filtering (Stage 0)
    - Per-date group indices for fast slicing
    - FDR-approved feature lists per formation window
    - Timing statistics for performance monitoring
    
    IMPORTANT: No forward-looking bias - all cached data uses only historical
    information available at the time of computation.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.clear()
        return cls._instance
    
    def clear(self):
        """Clear all cached data."""
        self.initialized = False
        self.target = None
        self.panel_df = None
        self.config = None
        self.formation_results = {}  # Keyed by formation window (start, end)
        self.base_feature_stats_cache = {}  # Cache for Stage 0 parent pre-filtering
        self.daily_ic_cache = {}  # Cache for daily IC series per formation window
        self.date_indices_cache = {}  # Cache for per-date row indices
        self.timing_stats = {}  # Performance timing storage
        # NEW: Per-date IC cache for cross-window reuse
        # Key: (date, feature_name) -> IC value
        # This enables reuse across rolling windows that share dates
        self.per_date_ic_cache = {}
        # NEW: Pre-computed daily IC matrix for ALL dates and ALL features
        # Computed once at backtest initialization, then sliced per window
        self.precomputed_daily_ic = None  # DataFrame: (all_dates × all_features)
        self.precomputed_features = None  # List of feature names in precomputed matrix
        logger.info("[FormationCache] Cache cleared")
    
    # -------------------------------------------------------------------------
    # Pre-computed Daily IC (computed ONCE at backtest init, sliced per window)
    # -------------------------------------------------------------------------
    def has_precomputed_ic(self) -> bool:
        """Check if pre-computed daily IC is available."""
        return self.precomputed_daily_ic is not None
    
    def set_precomputed_daily_ic(self, ic_df: pd.DataFrame):
        """
        Store pre-computed daily IC matrix.
        
        Args:
            ic_df: DataFrame with shape (all_dates × all_features)
        """
        self.precomputed_daily_ic = ic_df
        self.precomputed_features = list(ic_df.columns)
        logger.info(f"[FormationCache] Pre-computed daily IC stored: {ic_df.shape[0]} dates × {ic_df.shape[1]} features")
    
    def get_precomputed_ic_slice(self, date_min, date_max, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get a slice of pre-computed daily IC for a specific window.
        
        This is O(1) - just slicing, no computation!
        
        Args:
            date_min: Start date of window
            date_max: End date of window  
            feature_names: Optional list of features (if None, returns all)
            
        Returns:
            DataFrame slice of daily IC for the window
        """
        if self.precomputed_daily_ic is None:
            return None
        
        # Date slice
        mask = (self.precomputed_daily_ic.index >= date_min) & (self.precomputed_daily_ic.index <= date_max)
        ic_slice = self.precomputed_daily_ic.loc[mask]
        
        # Feature slice (if requested)
        if feature_names is not None:
            # Only include features that exist in pre-computed matrix
            available_features = [f for f in feature_names if f in self.precomputed_features]
            if len(available_features) < len(feature_names):
                # Some features are missing (e.g., new interactions not in pre-computed)
                missing = set(feature_names) - set(available_features)
                logger.debug(f"[FormationCache] {len(missing)} features not in pre-computed IC")
            if available_features:
                ic_slice = ic_slice[available_features]
            else:
                return None
        
        return ic_slice
    
    # -------------------------------------------------------------------------
    # Per-Date IC Cache (for cross-window reuse)
    # -------------------------------------------------------------------------
    def get_per_date_ic(self, date, feature_name):
        """Get cached IC for a specific date and feature."""
        key = (date, feature_name)
        return self.per_date_ic_cache.get(key, None)
    
    def set_per_date_ic(self, date, feature_name, ic_value):
        """Cache IC for a specific date and feature."""
        key = (date, feature_name)
        self.per_date_ic_cache[key] = ic_value
    
    def get_per_date_ic_batch(self, dates, feature_names):
        """
        Get cached ICs for multiple dates and features.
        
        Returns:
            dict: {(date, feature_name): ic_value} for cached entries
            set: set of (date, feature_name) tuples that need computation
        """
        cached = {}
        missing = set()
        for date in dates:
            for feat in feature_names:
                key = (date, feat)
                if key in self.per_date_ic_cache:
                    cached[key] = self.per_date_ic_cache[key]
                else:
                    missing.add(key)
        return cached, missing
    
    def set_per_date_ic_batch(self, ic_dict):
        """Cache multiple IC values at once."""
        self.per_date_ic_cache.update(ic_dict)
    
    # -------------------------------------------------------------------------
    # Base Feature Stats (Stage 0 parent pre-filtering)
    # -------------------------------------------------------------------------
    def get_base_feature_stats(self, date_min, date_max):
        """Get cached base feature stats for a formation window."""
        key = (date_min, date_max)
        return self.base_feature_stats_cache.get(key, None)
    
    def set_base_feature_stats(self, date_min, date_max, stats_dict):
        """Cache base feature stats for a formation window."""
        key = (date_min, date_max)
        self.base_feature_stats_cache[key] = stats_dict
        logger.debug(f"[FormationCache] Cached base feature stats for {key}")
    
    # -------------------------------------------------------------------------
    # Daily IC Series Cache
    # -------------------------------------------------------------------------
    def get_daily_ic(self, date_min, date_max, feature_subset=None):
        """
        Get cached daily IC series for a formation window.
        
        Args:
            date_min: Start of formation window
            date_max: End of formation window
            feature_subset: Optional list of features to return (for slicing)
            
        Returns:
            DataFrame of daily IC or None if not cached
        """
        key = (date_min, date_max)
        ic_daily = self.daily_ic_cache.get(key, None)
        if ic_daily is None:
            return None
        if feature_subset is not None:
            # Return only requested features (that exist in cache)
            available_cols = [f for f in feature_subset if f in ic_daily.columns]
            return ic_daily[available_cols] if available_cols else None
        return ic_daily
    
    def set_daily_ic(self, date_min, date_max, ic_daily_df):
        """Cache daily IC series for a formation window."""
        key = (date_min, date_max)
        self.daily_ic_cache[key] = ic_daily_df
        logger.debug(f"[FormationCache] Cached daily IC for {key} ({ic_daily_df.shape[1]} features)")
    
    # -------------------------------------------------------------------------
    # Date Indices Cache (for fast per-date slicing)
    # -------------------------------------------------------------------------
    def get_date_indices(self, dates_key):
        """Get cached date-to-row mapping."""
        return self.date_indices_cache.get(dates_key, None)
    
    def set_date_indices(self, dates_key, date_indices_dict):
        """Cache date-to-row mapping for fast slicing."""
        self.date_indices_cache[dates_key] = date_indices_dict
        logger.debug(f"[FormationCache] Cached date indices for {dates_key}")
    
    # -------------------------------------------------------------------------
    # Formation Results
    # -------------------------------------------------------------------------
    def set_context(self, panel_df: pd.DataFrame, target: str, config):
        """Set the data context for the cache."""
        self.panel_df = panel_df
        self.target = target
        self.config = config
        self.initialized = True
        logger.info(f"[FormationCache] Context set for target={target}")
    
    def get_formation_result(self, formation_start, formation_end):
        """Get cached formation result if available."""
        key = (formation_start, formation_end)
        return self.formation_results.get(key, None)
    
    def set_formation_result(self, formation_start, formation_end, result):
        """Cache a formation result."""
        key = (formation_start, formation_end)
        self.formation_results[key] = result
        logger.debug(f"[FormationCache] Cached formation result for {key}")
    
    # -------------------------------------------------------------------------
    # Timing Statistics
    # -------------------------------------------------------------------------
    def record_timing(self, stage_name: str, elapsed_seconds: float, window_id: str = None):
        """Record timing for a stage (for performance monitoring)."""
        if stage_name not in self.timing_stats:
            self.timing_stats[stage_name] = []
        self.timing_stats[stage_name].append({
            'window_id': window_id,
            'elapsed_seconds': elapsed_seconds
        })
    
    def get_timing_summary(self) -> Dict:
        """Get summary statistics for all timed stages."""
        summary = {}
        for stage_name, timings in self.timing_stats.items():
            elapsed_list = [t['elapsed_seconds'] for t in timings]
            summary[stage_name] = {
                'count': len(elapsed_list),
                'total_seconds': sum(elapsed_list),
                'mean_seconds': np.mean(elapsed_list) if elapsed_list else 0,
                'max_seconds': max(elapsed_list) if elapsed_list else 0,
                'min_seconds': min(elapsed_list) if elapsed_list else 0
            }
        return summary


_formation_cache = FormationCache()


def initialize_formation_cache(panel_df: pd.DataFrame, target: str, config):
    """
    Initialize the formation cache with panel data and target.
    
    This should be called once at the start of a backtest run before
    the walk-forward loop begins.
    
    Args:
        panel_df: Full panel data with (Date, Ticker) MultiIndex
        target: Target column name (e.g., 'y_resid_z_21d')
        config: ResearchConfig object
    """
    _formation_cache.clear()
    _formation_cache.set_context(panel_df, target, config)
    logger.info(f"[FormationCache] Initialized for target={target}")


def get_formation_cache() -> FormationCache:
    """Get the singleton formation cache instance."""
    return _formation_cache


def print_cache_timing_summary():
    """
    Print a summary of all recorded timing statistics from the cache.
    
    This should be called at the end of a backtest to analyze performance.
    """
    cache = get_formation_cache()
    summary = cache.get_timing_summary()
    
    if not summary:
        print("[FormationCache] No timing statistics recorded")
        return
    
    print("\n" + "=" * 70)
    print("FEATURE SELECTION TIMING SUMMARY")
    print("=" * 70)
    
    total_time = 0
    for stage_name, stats in sorted(summary.items()):
        total_time += stats['total_seconds']
        print(f"  {stage_name}:")
        print(f"    Count:   {stats['count']}")
        print(f"    Total:   {stats['total_seconds']:.1f}s")
        print(f"    Mean:    {stats['mean_seconds']:.2f}s")
        print(f"    Min/Max: {stats['min_seconds']:.2f}s / {stats['max_seconds']:.2f}s")
    
    print("-" * 70)
    print(f"  TOTAL TIME (all stages): {total_time:.1f}s")
    print("=" * 70 + "\n")
    
    return summary


def precompute_all_daily_ics(
    panel_df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    PRE-COMPUTE all daily ICs for the ENTIRE backtest period at once.
    
    This function computes cross-sectional Spearman IC for each (date, feature)
    pair ONE TIME. The result is stored in FormationCache and each walk-forward
    window simply SLICES the pre-computed matrix - no re-computation needed!
    
    SCIENTIFIC CORRECTNESS:
    - Each date's IC uses ONLY that date's cross-sectional data
    - No look-ahead bias: IC for date d computed using only data at date d
    - Window-specific aggregation (mean IC, t-stats) computed per window
    
    PERFORMANCE GAIN:
    - Before: ~30-35s per window computing daily IC
    - After: ~0.01s per window (just slicing a DataFrame)
    - For 80 windows: saves ~40+ minutes of redundant computation!
    
    Args:
        panel_df: Full panel data with (Date, Ticker) MultiIndex
        target_column: Name of target column (e.g., 'y_cs_21d')
        feature_columns: List of ALL feature columns to pre-compute
        n_jobs: Number of parallel jobs for IC computation
        verbose: Print progress messages
        
    Returns:
        DataFrame with shape (all_dates × all_features) containing daily ICs
    """
    import time as time_module
    
    if verbose:
        print("=" * 70)
        print("[PRE-COMPUTE] Computing daily IC for ALL dates × ALL features")
        print(f"[PRE-COMPUTE] Features: {len(feature_columns)}")
        print("=" * 70)
    
    start_time = time_module.time()
    
    # Get dates and target
    dates = panel_df.index.get_level_values('Date')
    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    n_dates = len(unique_dates)
    
    if verbose:
        print(f"[PRE-COMPUTE] Dates: {n_dates} ({unique_dates.min().date()} to {unique_dates.max().date()})")
    
    # Extract X and y
    X = panel_df[feature_columns].astype(np.float32)
    y = panel_df[target_column].astype(np.float32)
    
    # Use the parallel feature-batch computation
    # This is the same vectorized computation, but done ONCE for ALL dates
    if verbose:
        print(f"[PRE-COMPUTE] Computing daily IC (parallel, n_jobs={n_jobs})...")
    
    # Use compute_daily_ic_series_parallel which is already optimized
    ic_daily = compute_daily_ic_series_parallel(X, y, dates, n_jobs=n_jobs)
    
    elapsed = time_module.time() - start_time
    
    if verbose:
        print(f"[PRE-COMPUTE] Complete: {ic_daily.shape[0]} dates × {ic_daily.shape[1]} features in {elapsed:.1f}s")
        # Estimate savings
        n_windows = 80  # typical backtest
        estimated_savings = n_windows * 30  # ~30s per window saved
        print(f"[PRE-COMPUTE] Estimated time savings: ~{estimated_savings/60:.0f} minutes for {n_windows} windows")
    
    # Store in cache
    cache = get_formation_cache()
    cache.set_precomputed_daily_ic(ic_daily)
    
    return ic_daily


# ============================================================================
# Helper Functions
# ============================================================================

def compute_time_decay_weights(
    dates: pd.DatetimeIndex, 
    train_end: pd.Timestamp, 
    half_life: int,
    use_decay: bool = True
) -> np.ndarray:
    """
    Compute weights for samples - either exponential time-decay or uniform.
    
    When use_decay=False, returns uniform weights (1.0 for all dates).
    This is the default for feature selection to demonstrate cross-temporal persistence.
    
    Args:
        dates: DatetimeIndex of sample dates
        train_end: End date of training window
        half_life: Half-life in days for exponential decay (only used if use_decay=True)
        use_decay: If True, apply exponential decay. If False, return uniform weights.
        
    Returns:
        Array of weights with same length as dates
    """
    if not use_decay:
        # Uniform weights - all time points receive equal weight
        return np.ones(len(dates), dtype=np.float32)
    
    # Exponential time-decay weights
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


def _compute_ic_for_date(
    X_values: np.ndarray,
    y_values: np.ndarray,
    row_indices: np.ndarray,
    n_features: int
) -> np.ndarray:
    """
    Compute IC for all features for a single date. Helper for parallelization.
    
    Args:
        X_values: Full feature matrix (n_samples, n_features)
        y_values: Full target array (n_samples,)
        row_indices: Indices of samples for this date
        n_features: Number of features
        
    Returns:
        Array of IC values for each feature (n_features,)
    """
    ic_row = np.full(n_features, np.nan, dtype=np.float32)
    n_samples = len(row_indices)
    
    if n_samples < 3:
        return ic_row
    
    # Get data for this date
    X_date = X_values[row_indices]
    y_date = y_values[row_indices]
    
    # Mask for valid y values
    y_valid = ~np.isnan(y_date)
    n_valid_y = y_valid.sum()
    
    if n_valid_y < 3:
        return ic_row
    
    # Compute ranks for y once
    y_clean = y_date[y_valid]
    y_ranks = _rank_data(y_clean)
    y_centered = y_ranks - y_ranks.mean()
    y_norm = np.sqrt(np.sum(y_centered ** 2))
    
    if y_norm < 1e-10:
        return ic_row
    
    # Process all features at once for samples where y is valid
    X_subset = X_date[y_valid]
    
    # Count NaN per feature
    nan_per_feature = np.isnan(X_subset).sum(axis=0)
    no_nan_features = nan_per_feature == 0
    
    if no_nan_features.any():
        # Vectorized path for clean features
        X_clean = X_subset[:, no_nan_features]
        
        # Rank each column
        X_ranks = np.empty_like(X_clean, dtype=np.float32)
        order = np.argsort(X_clean, axis=0)
        ranks = np.arange(X_clean.shape[0], dtype=np.float32)
        for col_idx in range(X_clean.shape[1]):
            X_ranks[order[:, col_idx], col_idx] = ranks
        
        # Center ranks
        X_centered = X_ranks - X_ranks.mean(axis=0, keepdims=True)
        
        # Compute norms
        X_norms = np.sqrt(np.sum(X_centered ** 2, axis=0))
        
        # Compute correlations
        correlations = (y_centered @ X_centered) / (y_norm * X_norms)
        correlations = np.where(X_norms < 1e-10, np.nan, correlations)
        
        ic_row[no_nan_features] = correlations
    
    # Handle features with NaN (slower path)
    nan_feature_indices = np.where(~no_nan_features)[0]
    for j in nan_feature_indices:
        x_col = X_subset[:, j]
        valid_mask = ~np.isnan(x_col)
        n_valid = valid_mask.sum()
        
        if n_valid < 3:
            continue
        
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
        
        ic_row[j] = np.sum(x_centered * y_centered_subset) / (x_norm * y_norm_subset)
    
    return ic_row


def compute_daily_ic_series_parallel(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Compute daily cross-sectional IC for each feature using parallel processing.
    
    PARALLELIZED VERSION: Splits dates into chunks and processes in parallel.
    Achieves significant speedup on multi-core systems.
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for each sample
        n_jobs: Number of parallel jobs (-1 = all cores)
        
    Returns:
        DataFrame with shape (unique_dates × features) containing daily ICs
    """
    import os
    
    # Use X.index if it's a DatetimeIndex, otherwise use provided dates
    if isinstance(X.index, pd.DatetimeIndex):
        sample_dates = X.index
    else:
        sample_dates = dates
    
    unique_dates = pd.DatetimeIndex(sample_dates.unique())
    n_dates = len(unique_dates)
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    
    # Convert to numpy arrays for speed
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.float32)
    
    # Create date groups
    date_indices = {date: np.where(sample_dates == date)[0] for date in unique_dates}
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_compute_ic_for_date)(X_values, y_values, date_indices[date], n_features)
        for date in unique_dates
    )
    
    # Stack results
    ic_matrix = np.vstack(results)
    
    return pd.DataFrame(ic_matrix, index=unique_dates, columns=feature_names)


def _compute_ic_for_feature_batch(
    feature_indices: List[int],
    X_values: np.ndarray,
    y_values: np.ndarray,
    date_indices_list: List[Tuple],  # List of (date, row_indices)
    n_dates: int
) -> np.ndarray:
    """
    Compute IC for a batch of features across all dates.
    
    Helper for parallelization by features (more balanced load than by dates).
    
    Args:
        feature_indices: List of feature column indices to process
        X_values: Full feature matrix (n_samples, n_features)
        y_values: Full target array (n_samples,)
        date_indices_list: List of (date, row_indices) tuples
        n_dates: Number of unique dates
        
    Returns:
        IC matrix of shape (n_dates, len(feature_indices))
    """
    n_batch = len(feature_indices)
    ic_batch = np.full((n_dates, n_batch), np.nan, dtype=np.float32)
    
    for date_idx, (_, row_indices) in enumerate(date_indices_list):
        n_samples = len(row_indices)
        if n_samples < 3:
            continue
        
        # Get y for this date
        y_date = y_values[row_indices]
        y_valid = ~np.isnan(y_date)
        n_valid_y = y_valid.sum()
        
        if n_valid_y < 3:
            continue
        
        # Compute y ranks once for this date
        y_clean = y_date[y_valid]
        y_ranks = scipy_rankdata(y_clean).astype(np.float32) - 1
        y_centered = y_ranks - y_ranks.mean()
        y_norm = np.sqrt(np.sum(y_centered ** 2))
        
        if y_norm < 1e-10:
            continue
        
        # Process each feature in batch
        for batch_idx, feat_idx in enumerate(feature_indices):
            x_date = X_values[row_indices, feat_idx]
            x_subset = x_date[y_valid]
            
            # Check for additional NaN in feature
            x_valid = ~np.isnan(x_subset)
            n_valid = x_valid.sum()
            
            if n_valid < 3:
                continue
            
            if n_valid == n_valid_y:
                # No additional NaN - use precomputed y_ranks
                x_clean = x_subset
                x_ranks = scipy_rankdata(x_clean).astype(np.float32) - 1
                x_centered = x_ranks - x_ranks.mean()
                x_norm = np.sqrt(np.sum(x_centered ** 2))
                
                if x_norm < 1e-10:
                    continue
                
                ic_batch[date_idx, batch_idx] = np.sum(x_centered * y_centered) / (x_norm * y_norm)
            else:
                # Additional NaN - recompute for valid subset
                x_clean = x_subset[x_valid]
                y_clean_subset = y_clean[x_valid]
                
                x_ranks = scipy_rankdata(x_clean).astype(np.float32) - 1
                y_ranks_subset = scipy_rankdata(y_clean_subset).astype(np.float32) - 1
                
                x_centered = x_ranks - x_ranks.mean()
                y_centered_subset = y_ranks_subset - y_ranks_subset.mean()
                
                x_norm = np.sqrt(np.sum(x_centered ** 2))
                y_norm_subset = np.sqrt(np.sum(y_centered_subset ** 2))
                
                if x_norm < 1e-10 or y_norm_subset < 1e-10:
                    continue
                
                ic_batch[date_idx, batch_idx] = np.sum(x_centered * y_centered_subset) / (x_norm * y_norm_subset)
    
    return ic_batch


def compute_daily_ic_parallel_by_features(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    n_jobs: int = -1,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Compute daily cross-sectional IC parallelized BY FEATURES.
    
    This version parallelizes over feature batches instead of dates, which
    provides better load balancing when there are many more features than dates.
    For 11,000+ interactions across 500 dates, this gives ~4-8x speedup.
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for each sample
        n_jobs: Number of parallel jobs (-1 = all cores)
        batch_size: Number of features per batch (default 100)
        
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
    
    # Convert to numpy arrays for speed
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.float32)
    
    # Create date groups as list of tuples (for passing to parallel workers)
    date_indices_list = [(date, np.where(sample_dates == date)[0]) for date in unique_dates]
    
    # Split features into batches
    feature_batches = []
    for start_idx in range(0, n_features, batch_size):
        end_idx = min(start_idx + batch_size, n_features)
        feature_batches.append(list(range(start_idx, end_idx)))
    
    # Parallel computation over feature batches
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_compute_ic_for_feature_batch)(
            batch, X_values, y_values, date_indices_list, n_dates
        )
        for batch in feature_batches
    )
    
    # Concatenate results horizontally
    ic_matrix = np.hstack(results)
    
    return pd.DataFrame(ic_matrix, index=unique_dates, columns=feature_names)


def compute_daily_ic_with_cache(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    n_jobs: int = -1,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Compute daily IC - uses PRE-COMPUTED matrix if available, else per-date cache.
    
    PRIORITY ORDER:
    1. Pre-computed IC matrix (computed once at backtest init) - O(1) slice!
    2. Per-date cache (for features not in pre-computed matrix)
    3. Fresh computation (only for truly new features)
    
    Rolling walk-forward windows share ~95% of their dates. This function
    uses pre-computed values when available for INSTANT retrieval.
    
    Args:
        X: Feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for each sample
        n_jobs: Number of parallel jobs
        use_cache: Whether to use caching
        
    Returns:
        DataFrame with shape (unique_dates × features) containing daily ICs
    """
    cache = get_formation_cache()
    
    # Use X.index if it's a DatetimeIndex, otherwise use provided dates
    if isinstance(X.index, pd.DatetimeIndex):
        sample_dates = X.index
    else:
        sample_dates = dates
    
    unique_dates = pd.DatetimeIndex(sample_dates.unique()).sort_values()
    date_min, date_max = unique_dates.min(), unique_dates.max()
    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    n_dates = len(unique_dates)
    
    # =========================================================================
    # FAST PATH: Use pre-computed IC matrix if available
    # =========================================================================
    if cache.has_precomputed_ic():
        # Try to get slice from pre-computed matrix
        ic_precomputed = cache.get_precomputed_ic_slice(date_min, date_max, feature_names)
        
        if ic_precomputed is not None and len(ic_precomputed.columns) == n_features:
            # Perfect hit - all features available in pre-computed matrix
            # Filter to only the dates in our window
            ic_filtered = ic_precomputed.loc[ic_precomputed.index.isin(unique_dates)]
            print(f"[IC Cache] Using PRE-COMPUTED IC: {len(ic_filtered)} dates × {n_features} features (instant)")
            return ic_filtered
        
        elif ic_precomputed is not None and len(ic_precomputed.columns) > 0:
            # Partial hit - some features in pre-computed, need to compute rest
            precomputed_features = set(ic_precomputed.columns)
            missing_features = [f for f in feature_names if f not in precomputed_features]
            
            print(f"[IC Cache] PRE-COMPUTED: {len(precomputed_features)}/{n_features} features, computing {len(missing_features)} new")
            
            # Filter pre-computed to our dates
            ic_precomputed_filtered = ic_precomputed.loc[ic_precomputed.index.isin(unique_dates)]
            
            if len(missing_features) > 0:
                # Compute IC for missing features only
                X_missing = X[missing_features]
                ic_missing = compute_daily_ic_series_parallel(X_missing, y, sample_dates, n_jobs=n_jobs)
                
                # Merge pre-computed and newly computed
                ic_combined = pd.concat([ic_precomputed_filtered, ic_missing], axis=1)
                # Reorder to match requested feature order
                ic_combined = ic_combined[feature_names]
                return ic_combined
            else:
                return ic_precomputed_filtered[feature_names]
    
    # =========================================================================
    # FALLBACK: No pre-computed matrix - use per-date cache or compute fresh
    # =========================================================================
    if not use_cache:
        return compute_daily_ic_parallel_by_features(X, y, dates, n_jobs)
    
    # Check which (date, feature) pairs are already in per-date cache
    cached_ics, missing_pairs = cache.get_per_date_ic_batch(unique_dates, feature_names)
    
    n_cached = len(cached_ics)
    n_total = n_dates * n_features
    
    if n_cached > 0:
        print(f"[IC Cache] Reusing {n_cached}/{n_total} cached IC values ({100*n_cached/n_total:.1f}%)")
    
    # Preallocate result matrix
    ic_matrix = np.full((n_dates, n_features), np.nan, dtype=np.float32)
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    
    # Fill in cached values
    for (date, feat), ic_val in cached_ics.items():
        ic_matrix[date_to_idx[date], feat_to_idx[feat]] = ic_val
    
    # Identify which dates and features need computation
    if missing_pairs:
        # Group missing pairs by date for efficient computation
        missing_dates = set(d for d, _ in missing_pairs)
        missing_features_by_date = {}
        for d, f in missing_pairs:
            if d not in missing_features_by_date:
                missing_features_by_date[d] = set()
            missing_features_by_date[d].add(f)
        
        # Compute missing values - use feature-parallel version
        # Get subset of data for missing dates
        X_values = X.values.astype(np.float32)
        y_values = y.values.astype(np.float32)
        
        date_indices = {date: np.where(sample_dates == date)[0] for date in missing_dates}
        
        new_ics_to_cache = {}
        
        for date in missing_dates:
            row_indices = date_indices[date]
            features_to_compute = list(missing_features_by_date[date])
            
            # Get feature indices
            feat_indices = [feat_to_idx[f] for f in features_to_compute]
            
            # Compute IC for this date
            ic_row = _compute_ic_for_date(X_values, y_values, row_indices, n_features)
            
            # Store results
            date_idx = date_to_idx[date]
            for feat, f_idx in zip(features_to_compute, feat_indices):
                ic_val = ic_row[f_idx]
                ic_matrix[date_idx, f_idx] = ic_val
                new_ics_to_cache[(date, feat)] = ic_val
        
        # Cache newly computed values
        cache.set_per_date_ic_batch(new_ics_to_cache)
        print(f"[IC Cache] Computed and cached {len(new_ics_to_cache)} new IC values")
    
    return pd.DataFrame(ic_matrix, index=unique_dates, columns=feature_names)


def _rank_data(arr: np.ndarray) -> np.ndarray:
    """
    Fast ranking using scipy's optimized rankdata (Spearman approximation).
    
    Uses scipy.stats.rankdata which is implemented in C and handles ties properly.
    This is faster than pure numpy argsort for moderate-sized arrays.
    
    Args:
        arr: 1D array to rank
        
    Returns:
        Array of ranks (1-indexed, averaged for ties) converted to 0-indexed float32
    """
    # scipy_rankdata returns 1-indexed ranks with 'average' tie-breaking
    # Convert to 0-indexed for consistency with Pearson correlation on ranks
    return (scipy_rankdata(arr) - 1).astype(np.float32)


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


def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[{stage}] Memory usage: {mem_mb:.1f} MB")


# ============================================================================
# INTERACTION FEATURE PRE-SCREENING
# ============================================================================
# Interaction features (multiplicative products of base features from different
# families) undergo a separate, stricter screening process BEFORE being merged
# with base features for the main FDR pipeline. This prevents interaction
# features from dominating the feature pool through noise overfitting.

def formation_interaction_screening(
    X_base: pd.DataFrame,
    X_interaction: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    fdr_level: float = 0.05,
    ic_floor: float = 0.03,
    stability_folds: int = 5,
    min_ic_agreement: float = 0.60,
    max_features: int = 150,
    corr_vs_base_threshold: float = 0.75,
    half_life: int = 126,
    use_time_decay: bool = False,
    n_jobs: int = 4,
    target_column: str = None  # Optional: for cache lookup
) -> Tuple[List[str], pd.DataFrame]:
    """
    Pre-screen interaction features before merging with base features.
    
    This function implements a multi-gate screening process:
    1. IC-FDR Gate: Interaction must pass FDR control (stricter than base)
    2. IC Floor Gate: |IC| must exceed minimum threshold
    3. Stability Gate: IC sign must be consistent across rolling folds
    4. Orthogonality Gate: Must not be too correlated with any base feature
    5. Hard Cap: Limit total interactions to prevent explosion
    
    The goal is to ensure interaction features genuinely add predictive value
    beyond what base features provide, rather than overfitting to noise.
    
    Args:
        X_base: Base feature matrix (samples × base_features)
        X_interaction: Interaction feature matrix (samples × interaction_features)
        y: Target series
        dates: DatetimeIndex for each sample
        fdr_level: FDR threshold for interactions (default 0.05, stricter than base)
        ic_floor: Minimum absolute IC required (default 0.03)
        stability_folds: Number of time-series folds for stability check (default 5)
        min_ic_agreement: Fraction of folds with consistent IC sign (default 0.60)
        max_features: Hard cap on approved interactions (default 150)
        corr_vs_base_threshold: Max correlation with any base feature (default 0.75)
        half_life: Half-life for time-decay weights (default 126 days, only used if use_time_decay=True)
        use_time_decay: If True, apply exponential time decay; if False (default), use uniform weights
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (approved_interactions_list, diagnostics_df)
    """
    start_time = time.time()
    n_base = X_base.shape[1]
    n_interactions = X_interaction.shape[1]
    
    print("=" * 70)
    print("[Interaction Screening] Starting pre-filter for interaction features")
    print(f"[Interaction Screening]   Base features: {n_base}")
    print(f"[Interaction Screening]   Interaction candidates: {n_interactions}")
    print(f"[Interaction Screening]   Thresholds: FDR={fdr_level}, IC_floor={ic_floor}, "
          f"stability={min_ic_agreement:.0%}, max={max_features}")
    print("=" * 70)
    
    if n_interactions == 0:
        print("[Interaction Screening] No interaction features to screen")
        return [], pd.DataFrame()
    
    # Convert to float32 for efficiency
    X_interaction = X_interaction.astype(np.float32)
    X_base = X_base.astype(np.float32)
    y = y.astype(np.float64)
    
    unique_dates = dates.unique()
    n_dates = len(unique_dates)
    
    # =========================================================================
    # STAGE 0: PARENT PRE-FILTERING (new - from fix_selection_v2.md)
    # Drop interactions where BOTH parents are "clearly bad"
    # A parent is "clearly bad" if: p > alpha_parent_loose AND |IC| < IC_min_parent_loose
    # 
    # PERFORMANCE OPTIMIZATION: Cache base feature stats since they're expensive
    # to compute and don't change within a formation window.
    # =========================================================================
    stage_start = time.time()
    print(f"[Interaction Screening] Stage 0: Parent pre-filtering...")
    
    # Parse config parameters for parent filtering
    # TIGHTENED THRESHOLDS: More aggressive filtering to reduce interaction count
    # Changed from: alpha=0.50, ic_min=0.01 (too loose)
    # Changed to: alpha=0.30, ic_min=0.015 (filters ~50% more interactions early)
    alpha_parent_loose = 0.30  # Stricter: 30% FDR threshold for "clearly bad"
    ic_min_parent_loose = 0.015  # Stricter: 1.5% IC threshold for "clearly bad"
    
    # Check cache for base feature stats
    cache = get_formation_cache()
    date_min, date_max = dates.min(), dates.max()
    base_feature_stats = cache.get_base_feature_stats(date_min, date_max)
    
    if base_feature_stats is not None:
        # Fast path: use cached base feature stats
        print(f"[Interaction Screening] Stage 0: Using CACHED base feature stats ({len(base_feature_stats)} features)")
    else:
        # Slow path: compute base feature diagnostics (IC + FDR)
        print(f"[Interaction Screening] Stage 0: Computing base feature stats (will cache)...")
        ic_daily_base = compute_daily_ic_series(X_base, y, dates)
        
        # Compute weights for base feature IC
        unique_dates_idx = ic_daily_base.index
        train_end = unique_dates_idx.max()
        weights_base = compute_time_decay_weights(unique_dates_idx, train_end, half_life, use_decay=use_time_decay)
        
        # Weighted mean IC for base features
        ic_base_weighted = np.average(ic_daily_base.values, axis=0, weights=weights_base)
        
        # Newey-West t-stats for base features
        t_nw_base, n_dates_base = _compute_newey_west_vectorized(ic_daily_base, weights_base, max_lags=5)
        
        # Convert to p-values
        p_values_base = np.where(
            n_dates_base > 2,
            2 * (1 - stats.t.cdf(np.abs(t_nw_base), df=n_dates_base - 1)),
            1.0
        )
        
        # Apply FDR control to base features (for parent quality assessment)
        _, pvals_base_corrected, _, _ = multipletests(p_values_base, alpha=0.10, method='fdr_bh')
        
        # Build base feature lookup: feature_name -> (ic, p_corrected)
        base_feature_stats = {}
        for i, feat in enumerate(X_base.columns):
            base_feature_stats[feat] = {
                'ic': ic_base_weighted[i],
                'p_corrected': pvals_base_corrected[i]
            }
        
        # Cache the computed stats for future windows with same formation period
        cache.set_base_feature_stats(date_min, date_max, base_feature_stats)
        
        # Clean up intermediate variables
        del ic_daily_base
    
    # Helper function to extract parent names from interaction feature name
    def extract_parents(interaction_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract parent feature names from interaction name like 'featA_x_featB'."""
        if '_x_' in interaction_name:
            parts = interaction_name.split('_x_')
            if len(parts) == 2:
                return parts[0], parts[1]
        # Other interaction patterns (_div_, _minus_) - extract similarly
        for pattern in ['_div_', '_minus_']:
            if pattern in interaction_name:
                parts = interaction_name.split(pattern)
                if len(parts) == 2:
                    return parts[0], parts[1]
        return None, None
    
    # Helper function to check if parent is "clearly bad"
    def is_clearly_bad(parent_name: str) -> bool:
        if parent_name not in base_feature_stats:
            return False  # Unknown parent - don't drop
        stats = base_feature_stats[parent_name]
        # "Clearly bad" = high p-value AND low IC
        return stats['p_corrected'] > alpha_parent_loose and abs(stats['ic']) < ic_min_parent_loose
    
    # Filter interactions: drop if BOTH parents are clearly bad
    interactions_to_keep = []
    interactions_dropped = []
    
    for interaction_name in X_interaction.columns:
        parent_a, parent_b = extract_parents(interaction_name)
        
        if parent_a is None or parent_b is None:
            # Can't parse parents - keep the interaction
            interactions_to_keep.append(interaction_name)
            continue
        
        # Check if both parents are clearly bad
        both_bad = is_clearly_bad(parent_a) and is_clearly_bad(parent_b)
        
        if both_bad:
            interactions_dropped.append(interaction_name)
        else:
            interactions_to_keep.append(interaction_name)
    
    n_dropped_parents = len(interactions_dropped)
    n_remaining = len(interactions_to_keep)
    
    stage0_time = time.time() - stage_start
    cache.record_timing('interaction_screening_stage0', stage0_time)
    print(f"[Interaction Screening] Stage 0: Dropped {n_dropped_parents} interactions (both parents bad)")
    print(f"[Interaction Screening] Stage 0: {n_remaining} interactions remain ({stage0_time:.1f}s)")
    
    if n_remaining == 0:
        print("[Interaction Screening] All interactions dropped by parent pre-filter")
        # Return empty result with diagnostics
        diagnostics_df = pd.DataFrame({
            'feature': X_interaction.columns,
            'dropped_by_parent_filter': [True] * n_interactions,
            'approved': [False] * n_interactions
        })
        return [], diagnostics_df
    
    # Update X_interaction to only include surviving features
    X_interaction = X_interaction[interactions_to_keep]
    
    # =========================================================================
    # STAGE 1: Compute daily IC series for remaining interactions
    # Uses PER-DATE CACHING for cross-window reuse (rolling windows share ~95% of dates)
    # Uses FEATURE-PARALLEL computation for better load balancing with many interactions
    # =========================================================================
    stage_start = time.time()
    
    # Use the new feature-parallel version with per-date caching
    # This provides:
    # 1. Parallelization over features (better load balance for 11,000+ interactions)
    # 2. Per-date caching (cross-window reuse since rolling windows share dates)
    print(f"[Interaction Screening] Stage 1: Computing daily IC for {n_remaining} interactions...")
    print(f"[Interaction Screening] Stage 1: Using feature-parallel + per-date cache")
    
    ic_daily = compute_daily_ic_with_cache(X_interaction, y, dates, n_jobs=n_jobs, use_cache=True)
    
    stage1_time = time.time() - stage_start
    cache.record_timing('interaction_screening_stage1', stage1_time)
    print(f"[Interaction Screening] Stage 1 complete ({stage1_time:.1f}s)")
    
    # =========================================================================
    # STAGE 2: IC-FDR with Newey-West t-stats
    # =========================================================================
    stage_start = time.time()
    print(f"[Interaction Screening] Stage 2: Running FDR control at level {fdr_level}...")
    
    # Compute weights (time-decay or uniform based on configuration)
    unique_dates_idx = ic_daily.index
    train_end = unique_dates_idx.max()
    weights = compute_time_decay_weights(unique_dates_idx, train_end, half_life, use_decay=use_time_decay)
    
    # Weighted mean IC
    ic_values = ic_daily.values
    ic_weighted = np.average(ic_values, axis=0, weights=weights)
    
    # Newey-West t-stats
    t_nw_values, n_dates_per_feature = _compute_newey_west_vectorized(
        ic_daily, weights, max_lags=5
    )
    
    # Convert to p-values
    p_values = np.where(
        n_dates_per_feature > 2,
        2 * (1 - stats.t.cdf(np.abs(t_nw_values), df=n_dates_per_feature - 1)),
        1.0
    )
    
    # Apply FDR control
    reject, pvals_corrected, _, _ = multipletests(
        p_values,
        alpha=fdr_level,
        method='fdr_bh'
    )
    
    # Build diagnostics dataframe
    diagnostics_df = pd.DataFrame({
        'feature': X_interaction.columns,
        'ic_weighted': ic_weighted,
        't_nw': t_nw_values,
        'p_value': p_values,
        'p_value_corrected': pvals_corrected,
        'fdr_pass': reject,
        'ic_floor_pass': np.abs(ic_weighted) >= ic_floor
    })
    
    n_fdr_pass = diagnostics_df['fdr_pass'].sum()
    n_ic_floor_pass = (diagnostics_df['fdr_pass'] & diagnostics_df['ic_floor_pass']).sum()
    
    stage2_time = time.time() - stage_start
    print(f"[Interaction Screening] Stage 2: FDR pass={n_fdr_pass}, IC floor pass={n_ic_floor_pass} ({stage2_time:.1f}s)")
    
    # =========================================================================
    # STAGE 3: IC Sign Stability Check (Rolling Time-Series Folds)
    # =========================================================================
    stage_start = time.time()
    print(f"[Interaction Screening] Stage 3: Checking IC sign stability ({stability_folds} folds)...")
    
    # Get features that passed FDR + IC floor
    candidates_mask = diagnostics_df['fdr_pass'] & diagnostics_df['ic_floor_pass']
    candidate_features = diagnostics_df[candidates_mask]['feature'].tolist()
    
    if len(candidate_features) == 0:
        print("[Interaction Screening] No candidates passed FDR + IC floor gates")
        total_time = time.time() - start_time
        diagnostics_df['stability_pass'] = False
        diagnostics_df['orthogonality_pass'] = False
        diagnostics_df['approved'] = False
        return [], diagnostics_df
    
    # Create time-series folds
    sorted_dates = np.sort(unique_dates)
    fold_size = len(sorted_dates) // stability_folds
    
    # Compute IC sign per fold for each candidate
    ic_signs = np.zeros((len(candidate_features), stability_folds), dtype=np.float32)
    
    for fold_idx in range(stability_folds):
        fold_start = fold_idx * fold_size
        if fold_idx == stability_folds - 1:
            # Last fold takes remaining dates
            fold_dates = sorted_dates[fold_start:]
        else:
            fold_dates = sorted_dates[fold_start:fold_start + fold_size]
        
        # Get daily ICs for this fold
        fold_mask = ic_daily.index.isin(fold_dates)
        ic_fold = ic_daily.loc[fold_mask, candidate_features]
        
        # Compute mean IC sign per feature for this fold
        fold_mean_ic = ic_fold.mean(axis=0).values
        ic_signs[:, fold_idx] = np.sign(fold_mean_ic)
    
    # Stability = fraction of folds with same sign as overall IC
    overall_sign = np.sign(diagnostics_df.set_index('feature').loc[candidate_features, 'ic_weighted'].values)
    agreement = (ic_signs == overall_sign[:, np.newaxis]).mean(axis=1)
    
    # Create stability mask
    stability_pass = agreement >= min_ic_agreement
    
    # Add to diagnostics
    diagnostics_df['ic_agreement'] = 0.0
    diagnostics_df.loc[candidates_mask, 'ic_agreement'] = agreement
    diagnostics_df['stability_pass'] = False
    diagnostics_df.loc[candidates_mask, 'stability_pass'] = stability_pass
    
    stable_features = [f for f, passed in zip(candidate_features, stability_pass) if passed]
    n_stable = len(stable_features)
    
    stage3_time = time.time() - stage_start
    print(f"[Interaction Screening] Stage 3: {n_stable} interactions passed stability check ({stage3_time:.1f}s)")
    
    if n_stable == 0:
        print("[Interaction Screening] No candidates passed stability gate")
        total_time = time.time() - start_time
        diagnostics_df['orthogonality_pass'] = False
        diagnostics_df['approved'] = False
        return [], diagnostics_df
    
    # =========================================================================
    # STAGE 4: Orthogonality Check (vs Base Features)
    # =========================================================================
    stage_start = time.time()
    print(f"[Interaction Screening] Stage 4: Checking orthogonality vs {n_base} base features...")
    
    # For each stable interaction, compute max correlation with any base feature
    X_stable = X_interaction[stable_features].values.astype(np.float32)
    X_base_np = X_base.values.astype(np.float32)
    
    # Standardize for correlation computation
    def _standardize(arr):
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        stds[stds < 1e-10] = 1.0
        centered = (arr - means) / stds
        return np.nan_to_num(centered, nan=0.0)
    
    X_stable_std = _standardize(X_stable)
    X_base_std = _standardize(X_base_np)
    
    # Compute correlation matrix: interactions × base features
    n_samples = X_stable_std.shape[0]
    corr_matrix = np.abs(np.dot(X_stable_std.T, X_base_std) / n_samples)  # (n_stable, n_base)
    
    # Max correlation with any base feature
    max_corr_with_base = corr_matrix.max(axis=1)  # (n_stable,)
    
    # Features that are orthogonal enough
    orthogonal_mask = max_corr_with_base < corr_vs_base_threshold
    
    # Add to diagnostics
    diagnostics_df['max_corr_vs_base'] = 0.0
    stable_idx = diagnostics_df['feature'].isin(stable_features)
    diagnostics_df.loc[stable_idx, 'max_corr_vs_base'] = max_corr_with_base
    diagnostics_df['orthogonality_pass'] = False
    diagnostics_df.loc[stable_idx, 'orthogonality_pass'] = orthogonal_mask
    
    orthogonal_features = [f for f, passed in zip(stable_features, orthogonal_mask) if passed]
    n_orthogonal = len(orthogonal_features)
    
    stage4_time = time.time() - stage_start
    print(f"[Interaction Screening] Stage 4: {n_orthogonal} interactions passed orthogonality check ({stage4_time:.1f}s)")
    
    # =========================================================================
    # STAGE 5: Hard Cap + Ranking by IC Magnitude
    # =========================================================================
    stage_start = time.time()
    
    if n_orthogonal > max_features:
        print(f"[Interaction Screening] Stage 5: Applying hard cap ({n_orthogonal} -> {max_features})...")
        
        # Rank by absolute weighted IC
        orthogonal_ic = diagnostics_df.set_index('feature').loc[orthogonal_features, 'ic_weighted'].abs()
        top_features = orthogonal_ic.nlargest(max_features).index.tolist()
        approved_interactions = top_features
    else:
        approved_interactions = orthogonal_features
    
    # Mark final approved features
    diagnostics_df['approved'] = diagnostics_df['feature'].isin(approved_interactions)
    
    # Add dropped interactions back to diagnostics for full reporting
    if interactions_dropped:
        dropped_df = pd.DataFrame({
            'feature': interactions_dropped,
            'dropped_by_parent_filter': True,
            'ic_weighted': np.nan,
            't_nw': np.nan,
            'p_value': np.nan,
            'p_value_corrected': np.nan,
            'fdr_pass': False,
            'ic_floor_pass': False,
            'ic_agreement': np.nan,
            'stability_pass': False,
            'max_corr_vs_base': np.nan,
            'orthogonality_pass': False,
            'approved': False
        })
        diagnostics_df['dropped_by_parent_filter'] = False
        diagnostics_df = pd.concat([diagnostics_df, dropped_df], ignore_index=True)
    
    stage5_time = time.time() - stage_start
    total_time = time.time() - start_time
    
    # =========================================================================
    # Summary
    # =========================================================================
    n_approved = len(approved_interactions)
    
    print("=" * 70)
    print("[Interaction Screening] SUMMARY:")
    print(f"[Interaction Screening]   Input interactions:       {n_interactions}")
    print(f"[Interaction Screening]   After parent filter:      {n_remaining} (dropped {n_dropped_parents})")
    print(f"[Interaction Screening]   After FDR gate:           {n_fdr_pass}")
    print(f"[Interaction Screening]   After IC floor gate:      {n_ic_floor_pass}")
    print(f"[Interaction Screening]   After stability gate:     {n_stable}")
    print(f"[Interaction Screening]   After orthogonality:      {n_orthogonal}")
    print(f"[Interaction Screening]   Final approved:           {n_approved}")
    print(f"[Interaction Screening]   Pass rate: {100 * n_approved / n_interactions:.1f}%")
    print(f"[Interaction Screening]   Total time: {total_time:.1f}s")
    print("=" * 70)
    
    # Clean up
    del ic_daily, corr_matrix, X_stable_std, X_base_std
    
    return approved_interactions, diagnostics_df


# ============================================================================
# Short-Lag Feature Protection (pipeline integration)
# ============================================================================

def identify_short_lag_features(
    features: List[str],
    max_horizon: int = 5,
) -> List[str]:
    """
    Identify features with horizon < max_horizon days (short-lag features).
    
    Args:
        features: List of feature names
        max_horizon: Maximum horizon to be considered "short-lag" (exclusive)
                    Default 5 means features with horizon 1, 2, 3, 4 days
        
    Returns:
        List of short-lag feature names
    """
    short_lag = []
    for feat in features:
        horizon = extract_feature_horizon(feat)
        if horizon is not None and 1 <= horizon < max_horizon:
            short_lag.append(feat)
    return short_lag


def get_short_lag_candidates(
    features: List[str],
    diagnostics_df: pd.DataFrame,
    max_horizon: int = 5,
    min_ic: float = 0.01,
    sort_by: str = 'ic_weighted',
) -> pd.DataFrame:
    """
    Get short-lag feature candidates with their diagnostics, sorted by |IC|.
    
    This is used by pipeline stages (FDR, soft ranking, redundancy) to protect
    short-lag features from being filtered out.
    
    Args:
        features: List of feature names to consider
        diagnostics_df: DataFrame with feature diagnostics (must have 'feature' column)
        max_horizon: Maximum horizon to be considered "short-lag" (exclusive)
        min_ic: Minimum |IC| threshold for short-lag candidates
        sort_by: Column to sort by (default 'ic_weighted', uses absolute value)
        
    Returns:
        DataFrame of short-lag candidates sorted by |IC| descending
    """
    # Identify short-lag features
    short_lag_feats = identify_short_lag_features(features, max_horizon)
    
    if not short_lag_feats:
        return pd.DataFrame()
    
    # Filter diagnostics to short-lag features
    diag = diagnostics_df[diagnostics_df['feature'].isin(short_lag_feats)].copy()
    
    if diag.empty:
        return pd.DataFrame()
    
    # Add horizon column
    diag['horizon'] = diag['feature'].apply(extract_feature_horizon)
    
    # Add absolute IC column for sorting
    if sort_by in diag.columns:
        diag['ic_abs'] = diag[sort_by].abs()
    elif 'ic_weighted' in diag.columns:
        diag['ic_abs'] = diag['ic_weighted'].abs()
    else:
        diag['ic_abs'] = 0.0
    
    # Filter by minimum IC
    diag = diag[diag['ic_abs'] >= min_ic]
    
    # Sort by |IC| descending
    diag = diag.sort_values('ic_abs', ascending=False)
    
    return diag


def protect_short_lag_features(
    approved_features: List[str],
    all_features: List[str],
    diagnostics_df: pd.DataFrame,
    protect_count: int,
    max_horizon: int = 5,
    min_ic: float = 0.01,
    stage_name: str = "unknown",
) -> Tuple[List[str], Dict]:
    """
    Ensure at least `protect_count` short-lag features survive a pipeline stage.
    
    If the current approved_features has fewer than `protect_count` short-lag features,
    this function adds the top short-lag features (by |IC|) from the candidates pool.
    
    Args:
        approved_features: Features currently approved by this stage
        all_features: All features that entered this stage (superset of approved)
        diagnostics_df: DataFrame with diagnostics for all_features
        protect_count: Number of short-lag features to protect
        max_horizon: Maximum horizon for short-lag classification
        min_ic: Minimum |IC| threshold for protection
        stage_name: Name of pipeline stage (for logging)
        
    Returns:
        Tuple of (updated_approved_features, protection_diagnostics)
    """
    # Count short-lag features already in approved set
    approved_short_lag = identify_short_lag_features(approved_features, max_horizon)
    current_count = len(approved_short_lag)
    
    protection_info = {
        'stage': stage_name,
        'max_horizon': max_horizon,
        'protect_count': protect_count,
        'short_lag_in_approved': current_count,
        'short_lag_features': approved_short_lag,
        'protected': [],
        'action': 'none',
    }
    
    # If we already have enough, no action needed
    if current_count >= protect_count:
        protection_info['action'] = f'sufficient ({current_count} >= {protect_count})'
        print(f"[{stage_name}] Short-lag protection: {current_count} features already protected (need {protect_count})")
        return approved_features, protection_info
    
    # Get candidates from all features that entered this stage
    candidates = get_short_lag_candidates(
        features=all_features,
        diagnostics_df=diagnostics_df,
        max_horizon=max_horizon,
        min_ic=min_ic,
    )
    
    if candidates.empty:
        protection_info['action'] = 'no candidates'
        print(f"[{stage_name}] Short-lag protection: No candidates with |IC| >= {min_ic}")
        return approved_features, protection_info
    
    # Find how many we need to add
    need_to_add = protect_count - current_count
    
    # Get top candidates that are NOT already in approved_features
    approved_set = set(approved_features)
    candidates_to_add = []
    
    for _, row in candidates.iterrows():
        feat = row['feature']
        if feat not in approved_set:
            candidates_to_add.append(feat)
            if len(candidates_to_add) >= need_to_add:
                break
    
    # Add protected features to approved list
    if candidates_to_add:
        protection_info['protected'] = candidates_to_add
        protection_info['action'] = f'added {len(candidates_to_add)} features'
        
        updated_approved = list(approved_features) + candidates_to_add
        
        print(f"[{stage_name}] Short-lag protection: Added {len(candidates_to_add)} features "
              f"(had {current_count}, need {protect_count})")
        print(f"[{stage_name}]   Protected: {candidates_to_add}")
        
        return updated_approved, protection_info
    else:
        protection_info['action'] = 'no candidates to add (all already approved)'
        print(f"[{stage_name}] Short-lag protection: All candidates already approved")
        return approved_features, protection_info


# ============================================================================
# Short-Lag Feature Inclusion (DEPRECATED - use protection through pipeline)
# ============================================================================

def extract_feature_horizon(feature_name: str) -> Optional[int]:
    """
    Extract the horizon (in days) from a feature name.
    
    Supported patterns:
    - Close%-21 -> 21 (momentum returns)
    - Close_lag5 -> 5 (lagged prices)
    - Close_std21 -> 21 (rolling std)
    - Close_Mom5 -> 5 (momentum)
    - Close_RSI14 -> 14 (RSI)
    - Close_MA21 -> 21 (moving average)
    - Close_EMA10 -> 10 (exponential MA)
    - Close_skew21 -> 21 (skewness)
    - Close_kurt63 -> 63 (kurtosis)
    - Close_DD20 -> 20 (drawdown)
    - amihud_21 -> 21 (liquidity)
    - garman_klass_vol_21 -> 21 (volatility)
    - obv_slope_21 -> 21 (OBV slope)
    - trend_r2_21 -> 21 (trend R²)
    - Rel20_vs_VT -> 20 (relative performance)
    - pv_corr_21 -> 21 (price-volume correlation)
    - Hurst21 -> 21 (Hurst exponent)
    
    Returns None if no horizon can be extracted.
    """
    import re
    
    # Pattern 1: Close%-X (momentum returns) - e.g., Close%-21, Close%-5
    match = re.match(r'^Close%-(\d+)(?:_|$)', feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 2: Close_lagX - e.g., Close_lag5
    match = re.match(r'^Close_lag(\d+)', feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 3: Close_<stat><number> patterns - e.g., Close_std21, Close_Mom5, Close_RSI14
    # Matches: std, Mom, RSI, MA, EMA, DD, skew, kurt, Hurst, BollLo, BollUp, WilliamsR
    stat_pattern = r'^Close_(?:std|Mom|RSI|MA|EMA|DD|skew|kurt|Hurst|BollLo|BollUp|ATR|WilliamsR)(\d+)'
    match = re.match(stat_pattern, feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 3b: Close_Ret1dZ (1-day return z-score) - special case
    if 'Ret1dZ' in feature_name:
        return 1  # 1-day horizon
    
    # Pattern 4: Rel<number>_vs_ patterns - e.g., Rel20_vs_VT, Rel60_vs_Basket
    match = re.match(r'^Rel(\d+)_vs_', feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 5: Corr<number>_ patterns - e.g., Corr20_VT
    match = re.match(r'^Corr(\d+)_', feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 6: amihud_X, garman_klass_vol_X, parkinson_vol_X, rogers_satchell_vol_X
    volatility_pattern = r'(?:amihud|garman_klass_vol|parkinson_vol|rogers_satchell_vol)_(\d+)'
    match = re.search(volatility_pattern, feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 7: pv_corr_X, obv_slope_X, trend_r2_X, trend_strength_X, up_down_vol_ratio_X
    indicator_pattern = r'(?:pv_corr|obv_slope|trend_r2|trend_strength|up_down_vol_ratio|vol_of_vol)_(\d+)'
    match = re.search(indicator_pattern, feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 8: ADV_X, beta_VT_X, idio_vol_X - liquidity/risk metrics
    risk_pattern = r'(?:ADV|beta_VT|beta_BNDW|idio_vol|downside_beta_VT|r_squared_VT)_(\d+)'
    match = re.search(risk_pattern, feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 9: Generic _Xd suffix - e.g., mom_10d
    match = re.search(r'_(\d+)d$', feature_name)
    if match:
        return int(match.group(1))
    
    # Pattern 10: Trailing number after underscore - e.g., volume_zscore_20, price_zscore_50
    match = re.search(r'_(\d+)$', feature_name)
    if match:
        val = int(match.group(1))
        # Filter out very large numbers that aren't horizons
        if val <= 252:
            return val
    
    return None


def select_short_lag_features(
    approved_features: List[str],
    formation_diagnostics: pd.DataFrame,
    config,
    selected_so_far: List[str],
) -> Tuple[List[str], Dict]:
    """
    Select top features from short-lag buckets to ensure horizon diversity.
    
    This is called AFTER LassoCV selection to add short-lag features that may
    have been filtered out by the sparse selection.
    
    Args:
        approved_features: Features that passed Formation FDR
        formation_diagnostics: DataFrame with ic_weighted, p_value, etc.
        config: ResearchConfig with short_lag_* parameters
        selected_so_far: Features already selected by LassoCV
        
    Returns:
        Tuple of (features_to_add, diagnostics_dict)
    """
    fc = config.features
    
    if not fc.enable_short_lag_inclusion:
        return [], {'enabled': False}
    
    print(f"\n[Short-Lag] Selecting top features from short-lag buckets...")
    
    # Build diagnostics lookup
    diag_lookup = formation_diagnostics.set_index('feature')
    
    # Categorize approved features by horizon
    bucket1_features = []  # [1, bucket_1_max)
    bucket2_features = []  # [bucket_2_min, bucket_2_max]
    
    for feat in approved_features:
        horizon = extract_feature_horizon(feat)
        if horizon is not None:
            # Check IC and p-value thresholds
            if feat in diag_lookup.index:
                row = diag_lookup.loc[feat]
                ic_abs = abs(row['ic_weighted'])
                pval = row['p_value']
                
                if ic_abs >= fc.short_lag_min_ic and pval <= fc.short_lag_min_fdr_pval:
                    if 1 <= horizon < fc.short_lag_bucket_1_max:
                        bucket1_features.append((feat, horizon, ic_abs))
                    elif fc.short_lag_bucket_2_min <= horizon <= fc.short_lag_bucket_2_max:
                        bucket2_features.append((feat, horizon, ic_abs))
    
    # Sort by |IC| descending and select top K from each bucket
    bucket1_features.sort(key=lambda x: -x[2])
    bucket2_features.sort(key=lambda x: -x[2])
    
    bucket1_top = [f[0] for f in bucket1_features[:fc.short_lag_bucket_1_top_k]]
    bucket2_top = [f[0] for f in bucket2_features[:fc.short_lag_bucket_2_top_k]]
    
    # Exclude features already selected by LassoCV
    selected_set = set(selected_so_far)
    bucket1_add = [f for f in bucket1_top if f not in selected_set]
    bucket2_add = [f for f in bucket2_top if f not in selected_set]
    
    features_to_add = bucket1_add + bucket2_add
    
    diagnostics = {
        'enabled': True,
        'bucket1_range': f"[1, {fc.short_lag_bucket_1_max})",
        'bucket1_candidates': len(bucket1_features),
        'bucket1_selected': bucket1_top,
        'bucket1_added': bucket1_add,
        'bucket2_range': f"[{fc.short_lag_bucket_2_min}, {fc.short_lag_bucket_2_max}]",
        'bucket2_candidates': len(bucket2_features),
        'bucket2_selected': bucket2_top,
        'bucket2_added': bucket2_add,
        'total_added': len(features_to_add),
    }
    
    print(f"[Short-Lag]   Bucket 1 (horizon {diagnostics['bucket1_range']}): "
          f"{len(bucket1_features)} candidates -> {len(bucket1_top)} selected -> {len(bucket1_add)} new")
    print(f"[Short-Lag]   Bucket 2 (horizon {diagnostics['bucket2_range']}): "
          f"{len(bucket2_features)} candidates -> {len(bucket2_top)} selected -> {len(bucket2_add)} new")
    print(f"[Short-Lag]   Total features to add: {len(features_to_add)}")
    
    if features_to_add:
        print(f"[Short-Lag]   Adding: {features_to_add}")
    
    return features_to_add, diagnostics


# ============================================================================
# Formation: Global FDR on Raw Features
# ============================================================================

def formation_fdr(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    half_life: int = 126,
    fdr_level: float = 0.10,
    use_time_decay: bool = False,
    n_jobs: int = 4,
    config = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Run Formation-period FDR on raw features.
    
    Computes daily Spearman IC, weighted mean IC, Newey-West t-stats, and applies
    FDR control (Benjamini-Hochberg) to select approved raw features.
    
    If config is provided and enable_short_lag_protection is True, ensures that
    top short-lag features (by |IC|) survive the FDR filtering.
    
    Args:
        X: Feature matrix for Formation period (samples × features)
        y: Target series for Formation period
        dates: Date for each sample
        half_life: Half-life in days for time-decay weights (only used if use_time_decay=True)
        fdr_level: FDR control level (e.g., 0.10 for 10%)
        use_time_decay: If True, apply exponential time decay; if False (default), use uniform weights
        n_jobs: Number of parallel jobs for IC computation
        config: Optional config object with short_lag_protect_fdr parameter
        
    Returns:
        Tuple of (approved_features_list, diagnostics_df)
    """
    logger.info("=" * 80)
    logger.info("Starting Formation FDR on raw features")
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}, Dates: {len(dates.unique())}")
    log_memory_usage("Formation start")
    
    start_time = time.time()
    
    # =========================================================================
    # NaN HANDLING: Drop columns with NaN rather than failing
    # Risk features have inherent warmup periods, so we drop them for early windows
    # =========================================================================
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Dropping {len(nan_cols)} columns with NaN from formation FDR")
        if len(nan_cols) <= 20:
            for col in nan_cols:
                nan_pct = X[col].isna().sum() / len(X) * 100
                logger.info(f"    - {col}: {nan_pct:.1f}% NaN (dropped)")
        else:
            logger.info(f"    Columns dropped (first 10): {nan_cols[:10]}")
            logger.info(f"    ... and {len(nan_cols) - 10} more")
        
        # Drop the columns with NaN
        X = X.drop(columns=nan_cols)
        
        if X.shape[1] == 0:
            raise ValueError(
                "All features have NaN in this formation window. "
                "This typically means the window is too early (insufficient warmup). "
                f"NaN columns: {nan_cols[:10]}"
            )
    
    logger.info(f"✓ NaN check passed - proceeding with {X.shape[1]} features")
    
    # Initialize stage timing
    stage_times = {}
    cache = get_formation_cache()
    date_min, date_max = dates.min(), dates.max()
    
    # Convert to float32 for memory efficiency
    stage_start = time.time()
    X = X.astype(np.float32)
    y = y.astype(np.float64)  # Keep target as float64 for numerical stability
    stage_times['dtype_conversion'] = time.time() - stage_start
    
    # Compute daily IC series for all features
    # Try cache first (for features that were computed in a previous call)
    stage_start = time.time()
    feature_list = X.columns.tolist()
    cached_ic = cache.get_daily_ic(date_min, date_max, feature_subset=feature_list)
    
    if cached_ic is not None and set(cached_ic.columns) == set(feature_list):
        # Fast path: use fully cached IC
        print(f"[Formation FDR] Using CACHED daily IC for {X.shape[1]} features")
        ic_daily = cached_ic[feature_list]  # Ensure column order matches
    else:
        # Slow path: compute fresh (or partial cache + compute missing)
        if cached_ic is not None:
            # Partial cache hit - compute only missing features
            cached_cols = set(cached_ic.columns)
            missing_cols = [f for f in feature_list if f not in cached_cols]
            if missing_cols:
                print(f"[Formation FDR] Computing daily IC for {len(missing_cols)} missing features (cache had {len(cached_cols)})...")
                ic_missing = compute_daily_ic_series(X[missing_cols], y, dates)
                ic_daily = pd.concat([cached_ic, ic_missing], axis=1)[feature_list]
            else:
                ic_daily = cached_ic[feature_list]
        else:
            # Full compute
            print(f"[Formation FDR] Computing daily IC series for {X.shape[1]} features...")
            ic_daily = compute_daily_ic_series(X, y, dates)
        
        # Update cache with newly computed IC (merge with existing if any)
        existing_ic = cache.get_daily_ic(date_min, date_max)
        if existing_ic is not None:
            # Merge: take new columns, keep existing for overlap
            all_cols = list(set(existing_ic.columns.tolist() + ic_daily.columns.tolist()))
            merged_ic = pd.concat([existing_ic, ic_daily], axis=1)
            merged_ic = merged_ic.loc[:, ~merged_ic.columns.duplicated(keep='last')]
            cache.set_daily_ic(date_min, date_max, merged_ic)
        else:
            cache.set_daily_ic(date_min, date_max, ic_daily)
    
    stage_times['daily_ic'] = time.time() - stage_start
    cache.record_timing('formation_fdr_daily_ic', stage_times['daily_ic'])
    print(f"[Formation FDR] Daily IC computed ({stage_times['daily_ic']:.1f}s)")
    
    # Compute weights PER UNIQUE DATE (time-decay or uniform based on configuration)
    unique_dates = ic_daily.index  # This is a DatetimeIndex
    train_end = unique_dates.max()
    weights = compute_time_decay_weights(unique_dates, train_end, half_life, use_decay=use_time_decay)
    
    # Log weighting mode
    weight_mode = "time-decay" if use_time_decay else "uniform"
    print(f"[Formation FDR] Using {weight_mode} weights")
    
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
    
    # =========================================================================
    # DIRECTION CONSISTENCY: Compute fraction of time blocks with same sign as global mean
    # This ensures features have PERSISTENT signal, not just statistical significance.
    # =========================================================================
    stage_start = time.time()
    
    # Get config parameters (with defaults)
    if config is not None and hasattr(config.features, 'direction_consistency_min'):
        fc = config.features
        direction_consistency_min = fc.direction_consistency_min
        ic_min_threshold = fc.ic_min_base
    else:
        direction_consistency_min = 0.60  # Default: 60% of blocks must have same sign
        ic_min_threshold = 0.02           # Default: |IC| >= 2%
    
    # Split dates into blocks (approximately yearly blocks for robustness)
    # Using ~252 trading days per block (1 year)
    block_size_days = 252
    sorted_dates = np.sort(unique_dates)
    n_dates_total = len(sorted_dates)
    n_blocks = max(2, n_dates_total // block_size_days)
    
    # Create block boundaries (vectorized)
    block_boundaries = np.linspace(0, n_dates_total, n_blocks + 1, dtype=int)
    
    print(f"[Formation FDR] Computing direction consistency ({n_blocks} blocks)...")
    
    # VECTORIZED: Compute mean IC per block for all features at once
    n_features = ic_daily.shape[1]
    block_mean_ic = np.zeros((n_blocks, n_features), dtype=np.float32)
    
    # PRE-COMPUTE block masks once (performance optimization)
    block_masks = []
    for b in range(n_blocks):
        start_idx = block_boundaries[b]
        end_idx = block_boundaries[b + 1]
        block_dates = sorted_dates[start_idx:end_idx]
        block_masks.append(ic_daily.index.isin(block_dates))
    
    # Compute block mean IC using pre-computed masks
    for b in range(n_blocks):
        if block_masks[b].sum() > 0:
            block_mean_ic[b, :] = ic_daily.loc[block_masks[b]].mean(axis=0).values
    
    # Compute sign of global mean IC (already in diagnostics_df)
    global_sign = np.sign(ic_weighted)  # (n_features,)
    
    # Compute sign of each block's mean IC
    block_signs = np.sign(block_mean_ic)  # (n_blocks, n_features)
    
    # VECTORIZED direction consistency: fraction of blocks with same sign as global
    # For features with global_sign == 0, treat as perfect consistency
    # For others, count matching signs across blocks using broadcasting
    same_sign_matrix = (block_signs == global_sign[np.newaxis, :])  # (n_blocks, n_features)
    direction_consistency = same_sign_matrix.sum(axis=0).astype(np.float32) / n_blocks
    # Handle global_sign == 0 case: set consistency to 1.0
    direction_consistency[global_sign == 0] = 1.0
    
    diagnostics_df['direction_consistency'] = direction_consistency
    diagnostics_df['n_blocks'] = n_blocks
    
    stage_times['direction_consistency'] = time.time() - stage_start
    print(f"[Formation FDR] Direction consistency computed ({stage_times['direction_consistency']:.1f}s)")
    
    # =========================================================================
    # BLOCK-LEVEL PERSISTENCE FILTER (new requirement from instructions.txt)
    # In addition to pooled FDR + direction consistency, require that a feature
    # shows non-trivial IC in at least m_base blocks with the correct sign.
    # This is a SECOND filter applied on top of the existing pooled FDR filter.
    # =========================================================================
    stage_start = time.time()
    
    # Get config parameters for block persistence filter
    block_persistence_enabled = True
    block_m_base = 2  # Less aggressive: require 2 of 4 blocks (50%) instead of 3 of 4 (75%)
    block_ic_min = 0.01
    block_t_min = 1.5
    
    if config is not None and hasattr(config.features, 'block_persistence_enabled'):
        fc = config.features
        block_persistence_enabled = getattr(fc, 'block_persistence_enabled', True)
        block_m_base = getattr(fc, 'block_persistence_m_base', 3)
        block_ic_min = getattr(fc, 'block_persistence_ic_min', 0.01)
        block_t_min = getattr(fc, 'block_persistence_t_min', 1.5)

    # Cap block_m_base at the number of available blocks to avoid impossible thresholds
    if block_m_base > n_blocks:
        print(f"[Formation FDR] WARNING: block_m_base={block_m_base} > n_blocks={n_blocks}, capping to {n_blocks}")
        block_m_base = n_blocks

    if block_persistence_enabled:
        print(f"[Formation FDR] Computing block-level persistence (m_base={block_m_base}, IC_min={block_ic_min}, t_min={block_t_min})...")
        # Compute per-block t-statistics for each feature (VECTORIZED)
        # t_block = mean_IC_block / (std_IC_block / sqrt(n_dates_in_block))
        block_t_stats = np.zeros((n_blocks, n_features), dtype=np.float32)
        
        # Use pre-computed block_masks from direction consistency step
        for b in range(n_blocks):
            if block_masks[b].sum() > 2:  # Need at least 3 dates for t-stat
                ic_block = ic_daily.loc[block_masks[b]].values  # (n_dates_in_block, n_features)
                
                # Compute mean and std for each feature
                block_mean = np.nanmean(ic_block, axis=0)
                block_std = np.nanstd(ic_block, axis=0, ddof=1)
                n_dates_block = block_masks[b].sum()
                
                # t-stat = mean / (std / sqrt(n))
                se = block_std / np.sqrt(n_dates_block)
                se[se < 1e-10] = 1e-10  # Avoid division by zero
                block_t_stats[b, :] = block_mean / se
        
        # FULLY VECTORIZED block persistence check:
        # A block passes if: sign matches global AND (|IC| >= threshold OR |t| >= threshold)
        # Using numpy broadcasting instead of nested Python loops
        
        # Condition 1: Sign consistency (n_blocks, n_features)
        sign_matches = (block_signs == global_sign[np.newaxis, :])  # Broadcasting
        
        # Condition 2: Magnitude threshold (n_blocks, n_features)
        ic_passes = np.abs(block_mean_ic) >= block_ic_min
        t_passes = np.abs(block_t_stats) >= block_t_min
        magnitude_passes = ic_passes | t_passes
        
        # Combined: sign matches AND magnitude passes
        block_pass_matrix = sign_matches & magnitude_passes  # (n_blocks, n_features)
        
        # For global_sign == 0 features, only check magnitude (no sign requirement)
        zero_sign_mask = (global_sign == 0)
        block_pass_matrix[:, zero_sign_mask] = magnitude_passes[:, zero_sign_mask]
        
        # Count passing blocks per feature
        block_passes = block_pass_matrix.sum(axis=0).astype(np.int32)  # (n_features,)
        
        diagnostics_df['block_passes'] = block_passes
        diagnostics_df['block_persistence_pass'] = block_passes >= block_m_base
        
        stage_times['block_persistence'] = time.time() - stage_start
        n_block_pass = diagnostics_df['block_persistence_pass'].sum()
        print(f"[Formation FDR] Block persistence: {n_block_pass} features pass (≥{block_m_base} blocks) ({stage_times['block_persistence']:.1f}s)")
    else:
        # Block persistence disabled - all features pass this gate
        diagnostics_df['block_passes'] = n_blocks
        diagnostics_df['block_persistence_pass'] = True
        stage_times['block_persistence'] = 0.0
        print(f"[Formation FDR] Block persistence filter DISABLED")
    
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
    
    # =========================================================================
    # COMBINED FILTER: FDR + IC floor + Direction Consistency + Block Persistence
    # A feature must pass ALL FOUR gates to be approved
    # =========================================================================
    diagnostics_df['ic_floor_pass'] = np.abs(diagnostics_df['ic_weighted']) >= ic_min_threshold
    diagnostics_df['direction_consistency_pass'] = diagnostics_df['direction_consistency'] >= direction_consistency_min
    
    # Combined approval mask (now includes block persistence)
    approval_mask = (
        diagnostics_df['fdr_reject'] & 
        diagnostics_df['ic_floor_pass'] & 
        diagnostics_df['direction_consistency_pass'] &
        diagnostics_df['block_persistence_pass']
    )
    diagnostics_df['approved'] = approval_mask
    
    # Select approved features
    approved_features = diagnostics_df[approval_mask]['feature'].tolist()
    
    total_time = time.time() - start_time
    
    # Compute counts for clear reporting
    n_features_in = X.shape[1]
    n_fdr_pass = diagnostics_df['fdr_reject'].sum()
    n_ic_pass = (diagnostics_df['fdr_reject'] & diagnostics_df['ic_floor_pass']).sum()
    n_direction_pass = (diagnostics_df['fdr_reject'] & diagnostics_df['ic_floor_pass'] & diagnostics_df['direction_consistency_pass']).sum()
    n_block_pass_final = (diagnostics_df['fdr_reject'] & diagnostics_df['ic_floor_pass'] & 
                          diagnostics_df['direction_consistency_pass'] & diagnostics_df['block_persistence_pass']).sum()
    n_approved_raw = len(approved_features)
    n_rejected = n_features_in - n_approved_raw
    
    # =========================================================================
    # SHORT-LAG PROTECTION: Ensure top short-lag features survive FDR
    # =========================================================================
    short_lag_info = None
    if config is not None and hasattr(config.features, 'enable_short_lag_protection'):
        fc = config.features
        if fc.enable_short_lag_protection:
            approved_features, short_lag_info = protect_short_lag_features(
                approved_features=approved_features,
                all_features=list(X.columns),
                diagnostics_df=diagnostics_df,
                protect_count=fc.short_lag_protect_fdr,
                max_horizon=fc.short_lag_max_horizon,
                min_ic=fc.short_lag_min_ic,
                stage_name="Formation FDR",
            )
    
    n_approved = len(approved_features)
    n_protected = n_approved - n_approved_raw if short_lag_info else 0
    
    # Print summary - CLEAR terminology: FDR "rejects null hypothesis" = feature IS significant
    print("=" * 60)
    print(f"[Formation FDR] SUMMARY:")
    print(f"[Formation FDR]   Features in:  {n_features_in}")
    print(f"[Formation FDR]   Gate 1 - FDR significant:     {n_fdr_pass}")
    print(f"[Formation FDR]   Gate 2 - + IC floor (≥{ic_min_threshold:.2f}): {n_ic_pass}")
    print(f"[Formation FDR]   Gate 3 - + Direction (≥{direction_consistency_min:.0%}): {n_direction_pass}")
    if block_persistence_enabled:
        print(f"[Formation FDR]   Gate 4 - + Block persist (≥{block_m_base}): {n_approved_raw}")
    else:
        print(f"[Formation FDR]   Gate 4 - Block persist: DISABLED")
    if n_protected > 0:
        print(f"[Formation FDR]   Protected (short-lag): +{n_protected}")
        print(f"[Formation FDR]   Total approved: {n_approved}")
    print(f"[Formation FDR]   Rejected (all gates): {n_rejected}")
    print(f"[Formation FDR]   Direction consistency - Mean: {diagnostics_df['direction_consistency'].mean():.2%}")
    if block_persistence_enabled:
        print(f"[Formation FDR]   Block passes - Mean: {diagnostics_df['block_passes'].mean():.1f}/{n_blocks}")
    print(f"[Formation FDR]   IC stats - Mean: {diagnostics_df['ic_weighted'].mean():.4f}, "
          f"Median: {diagnostics_df['ic_weighted'].median():.4f}")
    print(f"[Formation FDR]   Stage times:")
    print(f"[Formation FDR]     - daily_ic:              {stage_times['daily_ic']:.1f}s")
    print(f"[Formation FDR]     - newey_west:            {stage_times['newey_west']:.1f}s")
    print(f"[Formation FDR]     - direction_consistency: {stage_times['direction_consistency']:.1f}s")
    if block_persistence_enabled:
        print(f"[Formation FDR]     - block_persistence:     {stage_times['block_persistence']:.1f}s")
    print(f"[Formation FDR]     - fdr_control:           {stage_times['fdr_control']:.1f}s")
    print(f"[Formation FDR]   Total time: {total_time:.1f}s")
    print("=" * 60)
    
    log_memory_usage("Formation end")
    
    # Clean up
    del ic_daily
    
    return approved_features, diagnostics_df


# ============================================================================
# Per-Window Soft Ranking (V3 only)
# ============================================================================

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
    
    return top_k_features, diagnostics


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
    
    return features_to_keep, diagnostics


def bucket_aware_redundancy_filter(
    X: pd.DataFrame,
    corr_within_bucket: float = 0.80,
    corr_cross_bucket: float = 0.90,
    min_per_bucket: int = 5,
    ic_scores: Optional[Dict[str, float]] = None,
    n_jobs: int = 1
) -> Tuple[List[str], Dict]:
    """
    Remove redundant features while preserving diversity across (family, horizon) buckets.
    
    This filter applies different correlation thresholds:
    - Within same bucket: stricter threshold (default 0.80)
    - Across different buckets: more lenient threshold (default 0.90)
    
    Additionally, it preserves at least `min_per_bucket` features per bucket.
    
    Args:
        X: Feature matrix (samples × features)
        corr_within_bucket: Max correlation within same (family, horizon) bucket
        corr_cross_bucket: Max correlation across different buckets  
        min_per_bucket: Minimum features to keep per bucket (default 5)
        ic_scores: Optional dict of feature -> IC score for tie-breaking
        n_jobs: Number of parallel jobs (unused, kept for API consistency)
        
    Returns:
        Tuple of (selected_features_list, diagnostics_dict)
    """
    from feature_engineering import get_feature_bucket, HORIZON_BUCKET_NAMES
    
    start_time = time.time()
    n_features = X.shape[1]
    feature_names = list(X.columns)
    
    print(f"[Bucket Redundancy] Starting with {n_features} features")
    print(f"[Bucket Redundancy] Thresholds: within={corr_within_bucket}, cross={corr_cross_bucket}, min_per_bucket={min_per_bucket}")
    
    # Classify each feature into bucket
    feature_buckets = {f: get_feature_bucket(f) for f in feature_names}
    
    # Group features by bucket for statistics
    bucket_counts = {}
    for f, bucket in feature_buckets.items():
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    
    print(f"[Bucket Redundancy] {len(bucket_counts)} unique buckets found")
    
    # If no IC scores provided, use column order as priority
    if ic_scores is None:
        ic_scores = {feat: -i for i, feat in enumerate(feature_names)}
    
    # VECTORIZED: Compute correlation matrix using numpy
    X_np = X.values.astype(np.float32)
    
    # Standardize columns
    col_means = np.nanmean(X_np, axis=0)
    col_stds = np.nanstd(X_np, axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    X_centered = (X_np - col_means) / col_stds
    X_centered = np.nan_to_num(X_centered, nan=0.0)
    
    # Compute full correlation matrix
    n_samples = X_centered.shape[0]
    corr_matrix = np.abs(np.dot(X_centered.T, X_centered) / n_samples)
    
    # Build removal set with bucket-aware logic
    features_to_remove = set()
    protected_counts = {bucket: 0 for bucket in bucket_counts.keys()}  # Track kept features per bucket
    
    # Sort features by IC (best first) so we prefer keeping high-IC features
    sorted_features = sorted(feature_names, key=lambda f: abs(ic_scores.get(f, 0)), reverse=True)
    
    # Process features in IC order
    features_kept = []
    
    for feat in sorted_features:
        if feat in features_to_remove:
            continue
            
        feat_idx = feature_names.index(feat)
        feat_bucket = feature_buckets[feat]
        
        # Check if this feature is correlated with any already-kept feature
        should_remove = False
        correlated_with = None
        
        for kept_feat in features_kept:
            kept_idx = feature_names.index(kept_feat)
            kept_bucket = feature_buckets[kept_feat]
            
            corr_val = corr_matrix[feat_idx, kept_idx]
            
            # Choose threshold based on bucket match
            if feat_bucket == kept_bucket:
                threshold = corr_within_bucket
            else:
                threshold = corr_cross_bucket
            
            if corr_val > threshold:
                should_remove = True
                correlated_with = kept_feat
                break
        
        if should_remove:
            # Check if bucket needs protection
            current_bucket_count = protected_counts[feat_bucket]
            bucket_features_remaining = sum(
                1 for f in feature_names 
                if f not in features_to_remove and feature_buckets[f] == feat_bucket
            )
            
            # If removing would drop bucket below minimum, keep this one instead
            if bucket_features_remaining <= min_per_bucket:
                # Force keep for diversity - don't remove
                features_kept.append(feat)
                protected_counts[feat_bucket] += 1
            else:
                features_to_remove.add(feat)
        else:
            features_kept.append(feat)
            protected_counts[feat_bucket] += 1
    
    # Final feature list maintains original order
    features_to_keep = [f for f in feature_names if f not in features_to_remove]
    
    elapsed = time.time() - start_time
    
    # Compute bucket-level diagnostics
    final_bucket_counts = {}
    for f in features_to_keep:
        bucket = feature_buckets[f]
        final_bucket_counts[bucket] = final_bucket_counts.get(bucket, 0) + 1
    
    diagnostics = {
        'n_start': n_features,
        'n_after_redundancy': len(features_to_keep),
        'n_removed': len(features_to_remove),
        'n_buckets_start': len(bucket_counts),
        'n_buckets_final': len(final_bucket_counts),
        'bucket_counts_start': bucket_counts,
        'bucket_counts_final': final_bucket_counts,
        'corr_within_bucket': corr_within_bucket,
        'corr_cross_bucket': corr_cross_bucket,
        'min_per_bucket': min_per_bucket,
        'time_redundancy': elapsed
    }
    
    # Print summary by family
    family_summary = {}
    for (family, hb), count in final_bucket_counts.items():
        family_summary[family] = family_summary.get(family, 0) + count
    
    print(f"[Bucket Redundancy] RESULT: {n_features} -> {len(features_to_keep)} features")
    print(f"[Bucket Redundancy] By family after filter:")
    for family, count in sorted(family_summary.items(), key=lambda x: -x[1]):
        print(f"    {family}: {count}")
    print(f"[Bucket Redundancy] Time: {elapsed:.2f}s")
    
    logger.info(f"Bucket redundancy filter: {len(features_to_keep)} features kept, "
                f"{len(features_to_remove)} removed (time: {elapsed:.2f}s)")
    
    del corr_matrix, X_centered
    
    return features_to_keep, diagnostics


def kfold_lasso_cv_selection(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    selection_threshold: float = 0.60,
    min_features: int = 12,
    max_features: int = 50,
    ridge_alpha: float = 0.01,
    n_jobs: int = 4,
    use_block_only: bool = True
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    K-fold time-series CV with LassoCV for robust feature selection.
    
    This implements the Stage B multivariate selection from fix_selection_v2.md:
    
    With use_block_only=True (default, NEW per instructions.txt):
        1. Split data into K contiguous time BLOCKS (no shuffling)
        2. For each block b: fit LassoCV ONLY on that block's data
        3. Compute selection frequency π_j = (# blocks selecting j) / K
        4. Keep features with π_j >= selection_threshold
        5. Refit Ridge on final selected features
        
        This is NON-OVERLAPPING: each block is an independent regime for stability.
    
    With use_block_only=False (legacy leave-one-out):
        1. Split data into K contiguous time folds (no shuffling)
        2. For each fold k: train on K-1 folds, fit LassoCV
        3. Compute selection frequency as before
        
        This is OVERLAPPING: training sets share data across folds.
    
    The non-overlapping approach (use_block_only=True) is preferred because:
    - Each block tests the feature's ability to predict in that specific regime
    - No data leakage between blocks
    - More stringent test of cross-temporal stability
    
    Args:
        X: Standardized feature matrix (samples × features)
        y: Target series
        dates: DatetimeIndex for each sample (for time-series split)
        n_splits: Number of time-series blocks/folds (K, default 5)
        selection_threshold: Min fraction of blocks where feature selected (π_threshold, default 0.60)
        min_features: Minimum features to keep (default 12)
        max_features: Maximum features after frequency filter (default 50)
        ridge_alpha: Ridge regularization for final refit (default 0.01)
        n_jobs: Number of parallel jobs for fold fitting (default 4)
        use_block_only: If True (default), fit on each block's data only (non-overlapping).
                       If False, use leave-one-out (overlapping training sets).
        
    Returns:
        Tuple of (selected_features_list, ridge_coefficients, diagnostics_dict)
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    
    start_time = time.time()
    n_samples = X.shape[0]
    n_features = X.shape[1]
    feature_names = list(X.columns)
    
    mode_str = "block-only (non-overlapping)" if use_block_only else "leave-one-out (overlapping)"
    print(f"[K-Fold LassoCV] Starting {n_splits}-fold time-series CV ({mode_str})...")
    print(f"[K-Fold LassoCV] Features in: {n_features}, Samples: {n_samples}")
    print(f"[K-Fold LassoCV] Selection threshold: π ≥ {selection_threshold:.0%}")
    
    # Create time-series splits (contiguous, preserving order)
    sorted_indices = np.argsort(dates)
    fold_size = n_samples // n_splits
    
    folds = []
    for k in range(n_splits):
        start_idx = k * fold_size
        if k == n_splits - 1:
            # Last fold takes remaining samples
            end_idx = n_samples
        else:
            end_idx = (k + 1) * fold_size
        
        block_indices = sorted_indices[start_idx:end_idx]
        
        if use_block_only:
            # NON-OVERLAPPING: fit only on this block's data
            train_indices = block_indices
        else:
            # OVERLAPPING (legacy): train on all OTHER folds (leave-one-out)
            train_indices = np.concatenate([sorted_indices[:start_idx], sorted_indices[end_idx:]])
        
        folds.append((train_indices, block_indices))
    
    # Track selection counts per feature
    selection_counts = np.zeros(n_features, dtype=np.int32)
    fold_results = []
    
    # Helper function to fit a single fold/block
    def fit_fold(fold_idx, train_idx, block_idx):
        """Fit LassoCV on training data and return selected feature mask."""
        X_train = X.iloc[train_idx].values
        y_train = y.iloc[train_idx].values
        
        # Need minimum samples for LassoCV
        if len(X_train) < 30:
            print(f"[K-Fold LassoCV] Block {fold_idx+1}: skipped (only {len(X_train)} samples)")
            return {
                'fold': fold_idx,
                'n_train': len(train_idx),
                'n_block': len(block_idx),
                'n_selected': 0,
                'alpha': 0.0,
                'selected_mask': np.zeros(n_features, dtype=bool),
                'skipped': True
            }
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            # Adjust inner CV based on sample size
            inner_cv = min(3, len(X_train) // 10)  # At least 10 samples per inner fold
            inner_cv = max(2, inner_cv)  # At least 2-fold CV
            
            lasso_cv = LassoCV(
                cv=inner_cv,
                max_iter=2000,
                fit_intercept=True,
                n_alphas=50,  # Fewer alphas for speed
                random_state=42
            )
            lasso_cv.fit(X_train, y_train)
        
        # Features with non-zero coefficients
        selected_mask = np.abs(lasso_cv.coef_) > 1e-10
        n_selected = selected_mask.sum()
        
        return {
            'fold': fold_idx,
            'n_train': len(train_idx),
            'n_block': len(block_idx),
            'n_selected': n_selected,
            'alpha': lasso_cv.alpha_,
            'selected_mask': selected_mask,
            'skipped': False
        }
    
    # Fit folds - use sequential execution for small blocks to avoid parallel overhead
    # Parallel overhead exceeds benefit when blocks are small (<500 samples each)
    min_samples_for_parallel = 500
    use_parallel = (n_samples // n_splits >= min_samples_for_parallel) and (n_jobs > 1)
    
    if use_parallel:
        fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(fit_fold)(k, train_idx, block_idx) 
            for k, (train_idx, block_idx) in enumerate(folds)
        )
    else:
        # Sequential execution for small blocks (avoids serialization overhead)
        fold_results = [
            fit_fold(k, train_idx, block_idx) 
            for k, (train_idx, block_idx) in enumerate(folds)
        ]
    
    # Aggregate selection counts (only from non-skipped folds)
    n_valid_folds = 0
    for result in fold_results:
        if not result.get('skipped', False):
            selection_counts += result['selected_mask'].astype(np.int32)
            n_valid_folds += 1
            print(f"[K-Fold LassoCV] Block {result['fold']+1}: {result['n_selected']} features selected "
                  f"(α={result['alpha']:.6f}, n_train={result['n_train']})")
    
    if n_valid_folds == 0:
        print(f"[K-Fold LassoCV] ERROR: All folds skipped!")
        return [], np.array([]), {'error': 'all_folds_skipped'}
    
    # Compute selection frequency (based on valid folds only)
    selection_freq = selection_counts / n_valid_folds
    
    # Filter by selection threshold
    stable_mask = selection_freq >= selection_threshold
    n_stable = stable_mask.sum()
    
    print(f"[K-Fold LassoCV] Features with π ≥ {selection_threshold:.0%}: {n_stable} (from {n_valid_folds} valid blocks)")
    
    # Apply feature bounds
    if n_stable > max_features:
        # Cap at max_features - take top by selection frequency, then by alphabetical order
        print(f"[K-Fold LassoCV] Capping: {n_stable} > {max_features}")
        stable_indices = np.where(stable_mask)[0]
        stable_freqs = selection_freq[stable_indices]
        top_k_local = np.argsort(stable_freqs)[::-1][:max_features]
        top_indices = stable_indices[top_k_local]
        
        final_mask = np.zeros(n_features, dtype=bool)
        final_mask[top_indices] = True
        selection_method = 'kfold_block_capped' if use_block_only else 'kfold_cv_capped'
        
    elif n_stable >= min_features:
        # Within bounds - use stable features
        final_mask = stable_mask
        selection_method = 'kfold_block' if use_block_only else 'kfold_cv'
        
    else:
        # Below minimum - take top by selection frequency
        print(f"[K-Fold LassoCV] Below minimum: {n_stable} < {min_features}, using top {min_features}")
        top_indices = np.argsort(selection_freq)[::-1][:min_features]
        final_mask = np.zeros(n_features, dtype=bool)
        final_mask[top_indices] = True
        selection_method = 'kfold_block_floor' if use_block_only else 'kfold_cv_floor'
    
    selected_features = [feat for i, feat in enumerate(feature_names) if final_mask[i]]
    n_selected = len(selected_features)
    
    # Refit Ridge on selected features using all data
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
    
    print(f"[K-Fold LassoCV] Final: {n_selected} features (bounds: [{min_features}, {max_features}])")
    print(f"[K-Fold LassoCV] Time: {elapsed:.1f}s")
    
    # Build diagnostics
    diagnostics = {
        'n_splits': n_splits,
        'n_valid_folds': n_valid_folds,
        'use_block_only': use_block_only,
        'selection_threshold': selection_threshold,
        'n_features_in': n_features,
        'n_stable': n_stable,
        'n_selected': n_selected,
        'min_features': min_features,
        'max_features': max_features,
        'selection_method': selection_method,
        'ridge_alpha': ridge_alpha,
        'time_kfold_cv': elapsed,
        'selection_freq': dict(zip(feature_names, selection_freq)),
        'fold_results': fold_results
    }
    
    return selected_features, ridge_coef, diagnostics


def training_lasso_lars_ic(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str = 'bic',
    min_features: int = 12,
    max_features: int = 25,
    ridge_alpha: float = 0.01,
    use_cv: bool = True,
    cv_folds: int = 5
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Select features using LassoCV (cross-validation) or LassoLarsIC (BIC/AIC) and refit with Ridge.
    
    With use_cv=True (default):
    1. Run LassoCV to find optimal alpha via cross-validation
    2. Cap at max_features by coefficient magnitude if needed
    3. Enforce min_features floor
    4. Refit with Ridge on the selected support
    
    With use_cv=False:
    1. Run LassoLarsIC with BIC/AIC criterion
    2. Apply same min/max feature bounds
    3. Refit with Ridge
    
    Args:
        X: Standardized feature matrix (samples × features)
        y: Target series
        criterion: 'bic' or 'aic' (only used if use_cv=False)
        min_features: Minimum features to select (default 12)
        max_features: Maximum features to select (default 25) - cap for sparsity
        ridge_alpha: Ridge regularization for final refit (default 0.01)
        use_cv: Use LassoCV instead of LassoLarsIC (default True)
        cv_folds: Number of CV folds (default 5)
        
    Returns:
        Tuple of (selected_features_list, ridge_coefficients, diagnostics_dict)
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    
    start_time = time.time()
    n_features = X.shape[1]
    feature_names = list(X.columns)
    
    if use_cv:
        # Cross-validation approach - actually measures out-of-sample error
        print(f"[LassoCV] Running {cv_folds}-fold CV to find optimal alpha...", flush=True)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            lasso_cv = LassoCV(
                cv=cv_folds,
                max_iter=2000,
                fit_intercept=True,
                n_alphas=100,
                random_state=42
            )
            lasso_cv.fit(X.values, y.values)
        
        best_alpha = lasso_cv.alpha_
        lasso_coef = lasso_cv.coef_
        nonzero_mask = np.abs(lasso_coef) > 1e-10
        n_nonzero = nonzero_mask.sum()
        
        print(f"[LassoCV] Optimal alpha={best_alpha:.6f}, selected {n_nonzero}/{n_features} features", flush=True)
        selection_method = 'lasso_cv'
        
    else:
        # Information criterion approach (BIC/AIC)
        print(f"[LassoLarsIC] Running with criterion='{criterion}'", flush=True)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            lars = LassoLarsIC(
                criterion=criterion,
                max_iter=1000,
                fit_intercept=True
            )
            lars.fit(X.values, y.values)
        
        alphas = lars.alphas_
        criterion_values = lars.criterion_
        best_alpha_idx = np.argmin(criterion_values)
        best_alpha = alphas[best_alpha_idx]
        
        print(f"[LassoLarsIC] Alpha path: max={alphas[0]:.4f}, min={alphas[-1]:.6f}, best={best_alpha:.6f}", flush=True)
        print(f"[LassoLarsIC] {criterion.upper()} at best alpha: {criterion_values[best_alpha_idx]:.2f}", flush=True)
        
        lasso_coef = lars.coef_
        nonzero_mask = np.abs(lasso_coef) > 1e-10
        n_nonzero = nonzero_mask.sum()
        
        print(f"[LassoLarsIC] {criterion.upper()} selected {n_nonzero}/{n_features} features", flush=True)
        selection_method = f'lars_{criterion}'
    
    # Step 2: Apply feature bounds (min_features <= n <= max_features)
    coef_abs = np.abs(lasso_coef)
    
    if n_nonzero > max_features:
        # Cap at max_features - take top by coefficient magnitude
        print(f"[LARS] Capping features: {n_nonzero} > {max_features}, keeping top {max_features} by |coef|", flush=True)
        
        # Get indices of top features among nonzero
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_coefs = coef_abs[nonzero_indices]
        top_k_local = np.argsort(nonzero_coefs)[::-1][:max_features]
        top_indices = nonzero_indices[top_k_local]
        
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[top_indices] = True
        selection_method += '_capped'
        
    elif n_nonzero >= min_features:
        # Within bounds - use as-is
        selected_mask = nonzero_mask
        
    else:
        # Below minimum - select top by |coef| even if some are tiny
        print(f"[LARS] Below minimum: {n_nonzero} < {min_features}, using top {min_features} by |coef|", flush=True)
        
        top_indices = np.argsort(coef_abs)[::-1][:min_features]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[top_indices] = True
        selection_method += '_floor'
    
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
    
    print(f"[LARS] Final: {n_selected} features (bounds: [{min_features}, {max_features}]), Ridge refit (0.01s)", flush=True)
    
    diagnostics = {
        'criterion': criterion if not use_cv else 'cv',
        'use_cv': use_cv,
        'n_features_in': n_features,
        'n_lasso_nonzero': n_nonzero,
        'n_selected': n_selected,
        'min_features': min_features,
        'max_features': max_features,
        'selection_method': selection_method,
        'ridge_alpha': ridge_alpha,
        'best_alpha': best_alpha,
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
        print(f"[standardization] {n_near_zero} features with MAD~0 (replaced MAD with 1.0)")
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
    
    return X_standardized, parameters


# ============================================================================
# Scoring at t0
# ============================================================================

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
# Main Pipeline Orchestrator (V3)
# ============================================================================

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
    
    print(f"[v3] NaN gate passed - input data is clean", flush=True)
    
    all_diagnostics = {}
    
    # Extract Formation artifacts
    approved_features = formation_artifacts['approved_features']
    # Note: V4 no longer uses best_alpha/best_l1_ratio from Formation
    # LassoCV or LassoLarsIC selects lambda automatically
    
    method_name = "LassoCV" if config.features.lars_use_cv else "LassoLarsIC"
    print(f"[v3] Formation-approved features (S_F): {len(approved_features)}", flush=True)
    print(f"[v3] Using {method_name} (min={config.features.lars_min_features}, max={config.features.lars_max_features})", flush=True)
    
    # Initialize stage timing
    stage_times = {}
    
    # Extract approved features
    X = X_train[approved_features]
    
    # Compute weights for soft ranking (time-decay or uniform based on config)
    train_end = dates_train[-1]
    use_decay = getattr(config.features, 'use_time_decay_weights', False)
    weights = compute_time_decay_weights(
        dates_train, 
        train_end, 
        half_life=config.features.training_halflife_days,
        use_decay=use_decay
    )
    weight_mode = "time-decay" if use_decay else "uniform"
    
    # Check if we should skip per-window IC ranking
    # Default=False: re-rank by recent IC to adapt to current market conditions
    skip_ic_ranking = getattr(config.features, 'skip_per_window_ic_ranking', False)
    
    if skip_ic_ranking:
        # =====================================================================
        # NEW BEHAVIOR (per instructions.txt): Skip IC re-ranking, use S_final directly
        # Once Stage A + redundancy + Stage B are done for a formation, treat
        # S_final = S_stable as the fixed feature set. Do not re-rank or filter
        # features by per-window IC or sign consistency.
        # =====================================================================
        print(f"[v3] Skipping per-window IC ranking (skip_per_window_ic_ranking=True)", flush=True)
        stage_times['soft_ranking'] = 0.0
        selected_ranked = approved_features  # Use all approved features as-is
        all_diagnostics['soft_ranking'] = {
            'skipped': True,
            'reason': 'skip_per_window_ic_ranking=True',
            'n_start': len(approved_features),
            'n_after_ranking': len(approved_features)
        }
        print(f"[v3 Stage 1] Soft ranking: SKIPPED (using all {len(approved_features)} approved features)", flush=True)
    else:
        # Original behavior: Soft ranking by IC + sign consistency
        print(f"[v3] Using {weight_mode} weights for soft ranking", flush=True)
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
    
    # Stage 4: Feature selection - K-fold LassoCV (preferred) or single LassoCV/LassoLarsIC
    stage_start = time.time()
    
    use_kfold = getattr(config.features, 'use_kfold_lasso_cv', True)
    
    if use_kfold:
        # K-fold LassoCV with selection frequency (from fix_selection_v2.md)
        # use_block_only=True (default) means non-overlapping block-wise stability selection
        selected_final, ridge_coef, lars_diag = kfold_lasso_cv_selection(
            X_std, 
            y_train,
            dates_train,
            n_splits=getattr(config.features, 'kfold_n_splits', 5),
            selection_threshold=getattr(config.features, 'kfold_selection_threshold', 0.60),
            min_features=config.features.lars_min_features,
            max_features=getattr(config.features, 'kfold_max_features', 50),
            ridge_alpha=config.features.ridge_refit_alpha,
            n_jobs=n_jobs,
            use_block_only=getattr(config.features, 'kfold_use_block_only', False)  # Default: leave-one-out CV
        )
        mode_str = "non-overlapping" if getattr(config.features, 'kfold_use_block_only', False) else "leave-one-out"
        method_name = f"K-Fold LassoCV ({config.features.kfold_n_splits} blocks, {mode_str})"
    else:
        # Single LassoCV or LassoLarsIC (legacy)
        selected_final, ridge_coef, lars_diag = training_lasso_lars_ic(
            X_std, 
            y_train,
            criterion=config.features.lars_criterion,
            min_features=config.features.lars_min_features,
            max_features=config.features.lars_max_features,
            ridge_alpha=config.features.ridge_refit_alpha,
            use_cv=config.features.lars_use_cv,
            cv_folds=config.features.lars_cv_folds
        )
        method_name = "LassoCV" if config.features.lars_use_cv else "LassoLarsIC"
    
    n_selected = len(selected_final)
    stage_times['lars_ridge'] = time.time() - stage_start
    
    print(f"[v3 Stage 4] {method_name} + Ridge: {len(selected_nonredundant)} -> {n_selected} features ({stage_times['lars_ridge']:.2f}s)", flush=True)
    
    all_diagnostics['lars'] = lars_diag
    
    # =========================================================================
    # Stage 4b: Short-Lag Feature Inclusion (diversify horizon coverage)
    # =========================================================================
    # Add top short-lag features that may have been filtered out by LassoCV
    stage_start = time.time()
    short_lag_features_added = []
    short_lag_diag = {'enabled': False}
    
    if config.features.enable_short_lag_inclusion:
        # Compute IC for all non-redundant features to rank by
        # Use training data IC as the criterion
        ic_values = {}
        for feat in selected_nonredundant:
            ic = compute_spearman_ic(X_std[feat].values, y_train.values)
            ic_values[feat] = ic
        
        # Create pseudo-diagnostics DataFrame for select_short_lag_features
        pseudo_diag = pd.DataFrame([
            {'feature': feat, 'ic_weighted': ic, 'p_value': 0.10}  # Use generous p-value
            for feat, ic in ic_values.items()
        ])
        
        short_lag_features_added, short_lag_diag = select_short_lag_features(
            approved_features=selected_nonredundant,  # Use post-redundancy features
            formation_diagnostics=pseudo_diag,
            config=config,
            selected_so_far=selected_final,
        )
        
        # Add short-lag features to final selection
        if short_lag_features_added:
            selected_final = list(selected_final) + short_lag_features_added
            n_after_short_lag = len(selected_final)
            print(f"[v3 Stage 4b] Short-lag inclusion: {n_selected} -> {n_after_short_lag} features", flush=True)
    
    stage_times['short_lag'] = time.time() - stage_start
    all_diagnostics['short_lag'] = short_lag_diag
    
    # Create a simple model wrapper that holds the Ridge coefficients
    # We'll use sklearn Ridge for consistency
    from sklearn.linear_model import Ridge
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    # Refit Ridge on ALL selected features (including short-lag additions)
    # We need to standardize the short-lag features first
    if short_lag_features_added:
        # Standardize the short-lag features using same approach
        X_shortlag = X[short_lag_features_added]
        X_shortlag_std, shortlag_params = robust_standardization(X_shortlag)
        
        # Merge into X_std
        for feat in short_lag_features_added:
            X_std[feat] = X_shortlag_std[feat]
            std_params['median'][feat] = shortlag_params['median'][feat]
            std_params['mad'][feat] = shortlag_params['mad'][feat]
    
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
    all_diagnostics['final_n_features'] = len(selected_final)
    
    # Record timing to cache for performance summary
    cache = get_formation_cache()
    cache.record_timing('per_window_soft_ranking', stage_times['soft_ranking'])
    cache.record_timing('per_window_redundancy', stage_times['redundancy_filter'])
    cache.record_timing('per_window_standardization', stage_times['standardization'])
    cache.record_timing('per_window_lars_ridge', stage_times['lars_ridge'])
    cache.record_timing('per_window_scoring', stage_times['scoring'])
    cache.record_timing('per_window_total', all_diagnostics['total_time'])
    
    # =========================================================================
    # Feature-level diagnostics for OOS stability analysis
    # =========================================================================
    # Compute IC for each final feature and get model goodness-of-fit
    feature_details = []
    for i, feat in enumerate(selected_final):
        # Compute IC (Spearman correlation with target)
        ic = compute_spearman_ic(X_train_selected[feat].values, y_train.values)
        
        # Get coefficient from Ridge model
        coef = final_model.coef_[i]
        
        feature_details.append({
            'feature': feat,
            'coefficient': float(coef),
            'ic': float(ic),
            'abs_coef': float(abs(coef)),
            'coef_sign': 'positive' if coef > 0 else 'negative',
        })
    
    # Sort by absolute coefficient
    feature_details = sorted(feature_details, key=lambda x: x['abs_coef'], reverse=True)
    all_diagnostics['feature_details'] = feature_details
    
    # Model goodness-of-fit metrics
    y_pred_train = final_model.predict(X_train_selected.values)
    
    # R² on training data
    ss_res = np.sum((y_train.values - y_pred_train) ** 2)
    ss_tot = np.sum((y_train.values - y_train.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Overall IC (correlation of predictions with target)
    model_ic = compute_spearman_ic(y_pred_train, y_train.values)
    
    # Residual standard deviation
    residual_std = np.std(y_train.values - y_pred_train)
    
    all_diagnostics['model_fit'] = {
        'r_squared': float(r_squared),
        'model_ic': float(model_ic),
        'residual_std': float(residual_std),
        'n_samples': len(y_train),
        'n_features': len(selected_final),
    }
    
    # Feature flow summary with feature lists for bucket analysis
    all_diagnostics['feature_flow'] = {
        'input_approved': len(approved_features),
        'after_soft_ranking': len(selected_ranked),
        'after_redundancy': len(selected_nonredundant),
        'after_lars': n_selected,
        'after_short_lag': len(selected_final),  # Final count including short-lag
        'short_lag_added': len(short_lag_features_added),
    }
    
    # Store actual feature lists for per-bucket breakdown
    all_diagnostics['feature_lists'] = {
        'input_approved': approved_features,
        'after_soft_ranking': selected_ranked,
        'after_redundancy': selected_nonredundant,
        'after_lars': list(set(selected_final) - set(short_lag_features_added)),  # LassoCV only
        'short_lag_added': short_lag_features_added,
        'final': selected_final,  # All features
    }
    
    print("-" * 60, flush=True)
    print(f"[v3] PIPELINE SUMMARY:", flush=True)
    short_lag_str = f" + {len(short_lag_features_added)} short-lag" if short_lag_features_added else ""
    print(f"[v3]   Feature flow: {len(approved_features)} -> {len(selected_ranked)} -> {len(selected_nonredundant)} -> {n_selected}{short_lag_str} -> {len(selected_final)} final", flush=True)
    print(f"[v3]   Model R2={r_squared:.4f}, IC={model_ic:.4f}", flush=True)
    print(f"[v3]   Stage times: ranking={stage_times['soft_ranking']:.2f}s, redundancy={stage_times['redundancy_filter']:.2f}s, "
          f"lars={stage_times['lars_ridge']:.2f}s, score={stage_times['scoring']:.2f}s", flush=True)
    print(f"[v3]   Total time: {all_diagnostics['total_time']:.2f}s", flush=True)
    print("=" * 80, flush=True)
    
    return scores_t0, all_diagnostics, window_model


# ============================================================================
# Walk-Forward Interface (High-Level Wrapper)
# ============================================================================

def train_window_model(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    t0: pd.Timestamp,
    config: 'ResearchConfig',
    formation_artifacts: Dict
) -> Tuple[Optional['ElasticNetWindowModel'], Dict]:
    """
    High-level interface for walk-forward engine.
    
    Takes a training panel (with Date/Ticker multi-index) and returns a trained
    model that can be used to score at t0.
    
    Pipeline (V3):
    - Uses approved features from Formation FDR
    - Uses LassoCV/LassoLarsIC for automatic regularization selection
    - Applies soft IC ranking (no hard stability drops)
    - No binning
    
    Args:
        panel: Training data panel (Date/Ticker multi-index, feature columns)
        metadata: Universe metadata (ticker, sector, region, etc.)
        t0: Scoring date (not used in training, only for scoring)
        config: Research configuration
        formation_artifacts: Dict from Formation phase containing:
            - 'approved_features': List[str] (S_F from formation_fdr)
        
    Returns:
        (model, diagnostics) tuple where:
        - model: ElasticNetWindowModel (or None if training failed)
        - diagnostics: Dict with training statistics
    """
    from config import ResearchConfig
    
    logger.info(f"train_window_model called for t0={t0}")
    
    # Validate formation_artifacts is provided
    if formation_artifacts is None:
        raise ValueError(
            "formation_artifacts is required. "
            "V2 backward compatibility mode has been removed. "
            "Ensure formation_years > 0 in config."
        )
    
    # Extract dates and prepare training data
    dates_train = panel.index.get_level_values('Date').unique().sort_values()
    
    # Identify feature columns (exclude target and metadata columns)
    # Use configured target column from config (SINGLE SOURCE OF TRUTH)
    target_col = config.target.target_column
    raw_target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    
    # Validate target column exists - DO NOT silently fall back
    if target_col not in panel.columns:
        available_targets = [c for c in panel.columns if c.startswith('y_') or c.startswith('FwdRet')]
        raise ValueError(
            f"[CRITICAL] Configured target column '{target_col}' not found in panel!\n"
            f"Available target-like columns: {available_targets}\n"
            f"This suggests a mismatch between config and data."
        )
    
    logger.info(f"[train_window_model] *** USING TARGET: {target_col} ***")
    print(f"[train_window_model] Model training against target: {target_col}")
    
    # Exclude all target columns (raw FwdRet and computed y_* targets)
    
    # Exclude all target columns (raw FwdRet and computed y_* targets)
    target_columns = [
        raw_target_col, 
        f'ret_fwd_{config.time.HOLDING_PERIOD_DAYS}d',
        'y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d'
    ]
    exclude_cols = {'market_cap', 'volume', 'dollar_volume', 'Close', 'ADV_63', 'ADV_63_Rank'}
    exclude_cols.update(target_columns)
    
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
    # Just create dummy X_t0 for per_window_pipeline_v3 (which expects it)
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
    
    # Run V3 Pipeline
    try:
        logger.info("Running V3 pipeline with Formation artifacts")
        scores_t0, diagnostics, model = per_window_pipeline_v3(
            X_train=X_train,
            y_train=y_train,
            X_t0=X_t0,
            dates_train=dates_train_samples,
            formation_artifacts=formation_artifacts,
            config=config,
            n_jobs=effective_n_jobs
        )
        
        return model, diagnostics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e), 'n_start': len(feature_cols)}

