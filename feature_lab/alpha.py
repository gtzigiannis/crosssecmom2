"""
Alpha Feature Selection Pipeline.

Implements the feature_alpha.md specification:
1. Hard Gates (6 binary pass/fail)
2. Composite Scoring (5 weighted metrics)
3. Redundancy Filtering (within-family + cross-family)
4. Final Pool Checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .config import AlphaConfig, AlphaThresholds
from .base import (
    DataQualityGate, 
    RedundancyFilter, 
    get_alpha_features,
    HAS_NUMBA,
    _pearson_numba,
    _rank_array_numba,
)

# Try numba
try:
    from numba import jit, prange
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Import family classification from feature_engineering
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_engineering import classify_feature_family, get_family_features


# =============================================================================
# Alpha Feature Result
# =============================================================================

@dataclass
class AlphaFeatureResult:
    """Complete analysis result for a single alpha feature."""
    feature: str
    
    # Gate results (all must pass)
    passed_data_quality: bool = False
    passed_ic_gate: bool = False
    passed_tstat_gate: bool = False
    passed_stability_gate: bool = False
    passed_residual_ic_gate: bool = False
    passed_long_tail_gate: bool = False
    passed_all_gates: bool = False
    
    # Metrics
    coverage: float = np.nan
    outlier_frac: float = np.nan
    ic_mean: float = np.nan
    ic_abs: float = np.nan
    ic_t_stat: float = np.nan
    sign_consistency: float = np.nan
    residual_ic: float = np.nan
    long_tail_excess: float = np.nan
    
    # Scoring metrics
    spread_sharpe: float = np.nan
    turnover: float = np.nan
    stress_ic: float = np.nan
    
    # Final score
    composite_score: float = np.nan
    percentile_rank: float = np.nan
    
    # Family info
    family: str = 'other'
    
    # Failure reason (if any)
    failure_reason: Optional[str] = None


# =============================================================================
# Newey-West HAC Standard Error
# =============================================================================

def compute_newey_west_tstat(ic_series: np.ndarray, nlags: Optional[int] = None) -> float:
    """
    Compute t-statistic using Newey-West HAC standard errors.
    
    HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
    account for serial correlation in the IC time series, which is important
    because daily ICs are typically autocorrelated.
    
    Args:
        ic_series: Time series of daily ICs
        nlags: Number of lags for HAC. If None, uses floor(4*(T/100)^(2/9)) rule.
    
    Returns:
        Newey-West adjusted t-statistic
    """
    # Remove NaN values
    x = ic_series[~np.isnan(ic_series)]
    n = len(x)
    
    if n < 10:
        return 0.0
    
    # Default lag selection: Newey-West optimal lag formula
    if nlags is None:
        nlags = int(np.floor(4 * (n / 100) ** (2/9)))
        nlags = max(1, min(nlags, n // 4))  # Bound between 1 and n/4
    
    mean_x = np.mean(x)
    demean = x - mean_x
    
    # Variance (gamma_0)
    gamma_0 = np.sum(demean ** 2) / n
    
    # Autocovariances with Bartlett kernel weights
    sum_weighted_gamma = gamma_0
    for lag in range(1, nlags + 1):
        # Autocovariance at lag k
        gamma_k = np.sum(demean[lag:] * demean[:-lag]) / n
        # Bartlett kernel weight
        weight = 1 - lag / (nlags + 1)
        # HAC adds 2 * weighted autocovariance (symmetric)
        sum_weighted_gamma += 2 * weight * gamma_k
    
    # HAC variance
    hac_var = sum_weighted_gamma / n
    
    if hac_var <= 1e-10:
        return 0.0
    
    # HAC standard error
    hac_se = np.sqrt(hac_var)
    
    # t-statistic
    t_stat = mean_x / hac_se
    
    return t_stat


# =============================================================================
# Numba-optimized computations
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_quintile_spread_numba(
    X: np.ndarray,
    y: np.ndarray,
    date_starts: np.ndarray,
    date_ends: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute daily Q5 - Q1 spread (quintile spread) and top quintile excess.
    
    Per spec: Q5 = top 20%, Q1 = bottom 20%
    Spread = mean_return(Q5) - mean_return(Q1)
    
    Returns:
        spreads: Array of daily (Q5 - Q1) spreads
        top_excess: Array of daily (Q5 - universe_mean) excess returns
    """
    n_dates = len(date_starts)
    spreads = np.full(n_dates, np.nan)
    top_excess = np.full(n_dates, np.nan)
    
    for d in range(n_dates):
        start = date_starts[d]
        end = date_ends[d]
        
        X_date = X[start:end]
        y_date = y[start:end]
        
        # Filter valid
        valid_count = 0
        for i in range(len(X_date)):
            if not np.isnan(X_date[i]) and not np.isnan(y_date[i]):
                valid_count += 1
        
        if valid_count < 20:
            continue
        
        X_valid = np.empty(valid_count)
        y_valid = np.empty(valid_count)
        idx = 0
        for i in range(len(X_date)):
            if not np.isnan(X_date[i]) and not np.isnan(y_date[i]):
                X_valid[idx] = X_date[i]
                y_valid[idx] = y_date[i]
                idx += 1
        
        # Compute quintile boundaries (20% / 80%)
        X_sorted_idx = np.argsort(X_valid)
        n = len(X_valid)
        
        q1_end = int(n * 0.2)      # Bottom 20% = Q1
        q5_start = int(n * 0.8)    # Top 20% = Q5
        
        if q1_end < 1 or q5_start >= n:
            continue
        
        # Q1 (bottom quintile) mean return
        q1_sum = 0.0
        for i in range(q1_end):
            q1_sum += y_valid[X_sorted_idx[i]]
        q1_mean = q1_sum / q1_end
        
        # Q10 (top decile) mean return
        # Q5 (top quintile) mean return
        q5_count = n - q5_start
        q5_sum = 0.0
        for i in range(q5_start, n):
            q5_sum += y_valid[X_sorted_idx[i]]
        q5_mean = q5_sum / q5_count
        
        # Universe mean
        universe_mean = np.mean(y_valid)
        
        spreads[d] = q5_mean - q1_mean
        top_excess[d] = q5_mean - universe_mean
    
    return spreads, top_excess


@jit(nopython=True, cache=True)
def _compute_turnover_numba_v2(
    X: np.ndarray,
    ticker_ids: np.ndarray,
    date_starts: np.ndarray,
    date_ends: np.ndarray,
    top_pct: float = 0.1
) -> float:
    """
    Compute average turnover in top decile holdings using integer ticker IDs.
    
    Turnover = fraction of top decile that changes from day to day.
    Uses ticker_ids (integers) instead of string tickers for numba compatibility.
    Lower turnover = better (more stable signal).
    
    Args:
        X: Feature values (flattened)
        ticker_ids: Integer mapping of ticker strings (same shape as X)
        date_starts: Start indices for each date
        date_ends: End indices for each date
        top_pct: Fraction for top decile (default 0.1)
    
    Returns:
        Mean turnover across all date transitions
    """
    n_dates = len(date_starts)
    if n_dates < 2:
        return np.nan
    
    turnovers = np.empty(n_dates - 1)
    valid_count = 0
    
    # Track previous top tickers (using fixed size array with count)
    prev_top_tickers = np.empty(0, dtype=np.int64)
    prev_size = 0
    
    for d in range(n_dates):
        start = date_starts[d]
        end = date_ends[d]
        X_date = X[start:end]
        tickers_date = ticker_ids[start:end]
        
        # Get valid indices
        valid_count_date = 0
        for i in range(len(X_date)):
            if not np.isnan(X_date[i]):
                valid_count_date += 1
        
        if valid_count_date < 20:
            continue
        
        # Collect valid values and tickers
        X_valid = np.empty(valid_count_date)
        tickers_valid = np.empty(valid_count_date, dtype=np.int64)
        j = 0
        for i in range(len(X_date)):
            if not np.isnan(X_date[i]):
                X_valid[j] = X_date[i]
                tickers_valid[j] = tickers_date[i]
                j += 1
        
        # Sort to get top decile
        sorted_idx = np.argsort(X_valid)
        top_n = max(1, int(valid_count_date * top_pct))
        
        # Get top decile ticker IDs
        curr_top_tickers = np.empty(top_n, dtype=np.int64)
        for i in range(top_n):
            curr_top_tickers[i] = tickers_valid[sorted_idx[valid_count_date - 1 - i]]
        
        if prev_size > 0:
            # Compute overlap using O(n*m) comparison (faster for small n, m ~50-200)
            overlap = 0
            for i in range(top_n):
                for j in range(prev_size):
                    if curr_top_tickers[i] == prev_top_tickers[j]:
                        overlap += 1
                        break
            
            max_size = max(top_n, prev_size)
            turnover = 1.0 - (overlap / max_size)
            turnovers[valid_count] = turnover
            valid_count += 1
        
        prev_top_tickers = curr_top_tickers
        prev_size = top_n
    
    if valid_count == 0:
        return np.nan
    
    return np.mean(turnovers[:valid_count])


# =============================================================================
# AlphaSelector
# =============================================================================

class AlphaSelector:
    """
    Alpha feature selection pipeline.
    
    Implements the full feature_alpha.md specification:
    1. Hard Gates: Data quality, IC, t-stat, stability, residual IC, long-tail
    2. Scoring: IC, long-tail, spread Sharpe, turnover, stress IC
    3. Redundancy: Within-family clustering, cross-family elimination
    4. Final pool health checks
    """
    
    def __init__(
        self,
        panel_df: pd.DataFrame,
        config: Optional[AlphaConfig] = None
    ):
        self.config = config or AlphaConfig()
        self.panel_df = panel_df
        
        # Ensure MultiIndex
        if isinstance(panel_df.index, pd.MultiIndex):
            self.dates = panel_df.index.get_level_values('Date')
            self.unique_dates = pd.DatetimeIndex(self.dates.unique()).sort_values()
        else:
            raise ValueError("Panel must have (Date, Ticker) MultiIndex")
        
        # Precompute date indices for numba
        self._precompute_date_indices()
        
        # Initialize gates
        self.data_quality_gate = DataQualityGate(
            min_coverage=self.config.thresholds.min_coverage,
            max_outlier_frac=self.config.thresholds.max_outlier_frac,
            outlier_sigma=self.config.thresholds.outlier_sigma,
            n_jobs=self.config.n_jobs
        )
        
        self.redundancy_filter = RedundancyFilter(
            within_family_max_corr=self.config.thresholds.within_family_max_corr,
            cross_family_max_corr=self.config.thresholds.cross_family_max_corr,
            max_per_family=self.config.thresholds.max_per_family,
            n_jobs=self.config.n_jobs
        )
        
        # Caches
        self._daily_ic_cache: Dict[str, np.ndarray] = {}
        self._benchmark_residual_cache: Dict[str, np.ndarray] = {}
        
        # Precompute ticker integer mapping for numba turnover
        self._precompute_ticker_ids()
        
        # Results storage
        self.results: Dict[str, AlphaFeatureResult] = {}
        
        if self.config.verbose:
            print(f"[AlphaSelector] Initialized with {len(self.unique_dates)} dates, "
                  f"{len(self.panel_df)} rows (Numba: {HAS_NUMBA})")
    
    def _precompute_date_indices(self):
        """Precompute date start/end indices for numba functions."""
        dates_array = self.dates.values
        unique_dates_array = self.unique_dates.values
        
        self._date_starts = np.zeros(len(unique_dates_array), dtype=np.int64)
        self._date_ends = np.zeros(len(unique_dates_array), dtype=np.int64)
        
        for i, date in enumerate(unique_dates_array):
            mask = dates_array == date
            indices = np.where(mask)[0]
            if len(indices) > 0:
                self._date_starts[i] = indices[0]
                self._date_ends[i] = indices[-1] + 1
    
    def _precompute_ticker_ids(self):
        """
        Map ticker strings to integer IDs for numba-compatible turnover computation.
        
        Creates self._ticker_ids array with same shape as panel_df rows,
        where each ticker string is mapped to a unique integer.
        """
        tickers = self.panel_df.index.get_level_values('Ticker').values
        unique_tickers = np.unique(tickers)
        ticker_to_id = {t: i for i, t in enumerate(unique_tickers)}
        self._ticker_ids = np.array([ticker_to_id[t] for t in tickers], dtype=np.int64)
    
    # =========================================================================
    # Daily IC Computation (with caching)
    # =========================================================================
    
    def compute_daily_ic(self, feature_col: str) -> np.ndarray:
        """
        Compute daily IC for a feature (cached).
        
        Returns array of daily ICs aligned with unique_dates.
        """
        if feature_col in self._daily_ic_cache:
            return self._daily_ic_cache[feature_col]
        
        target_col = self.config.target_column
        X = self.panel_df[feature_col].values
        y = self.panel_df[target_col].values
        
        daily_ics = np.full(len(self.unique_dates), np.nan)
        
        for i, date in enumerate(self.unique_dates):
            start, end = self._date_starts[i], self._date_ends[i]
            X_date = X[start:end]
            y_date = y[start:end]
            
            valid = ~(np.isnan(X_date) | np.isnan(y_date))
            if valid.sum() < self.config.min_samples_per_date:
                continue
            
            X_v = X_date[valid]
            y_v = y_date[valid]
            
            if np.std(X_v) < 1e-10 or np.std(y_v) < 1e-10:
                continue
            
            try:
                ic, _ = spearmanr(X_v, y_v)
                daily_ics[i] = ic
            except:
                continue
        
        self._daily_ic_cache[feature_col] = daily_ics
        return daily_ics
    
    def precompute_daily_ics(self, feature_cols: List[str]):
        """Precompute daily ICs for all features in parallel."""
        def compute_one(col):
            return col, self.compute_daily_ic(col)
        
        if self.config.n_jobs == 1:
            results = [compute_one(col) for col in feature_cols]
        else:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(compute_one)(col) for col in feature_cols
            )
        
        for col, ic in results:
            self._daily_ic_cache[col] = ic
        
        if self.config.verbose:
            print(f"[AlphaSelector] Precomputed daily ICs for {len(feature_cols)} features")
    
    # =========================================================================
    # Hard Gates
    # =========================================================================
    
    def gate_global_ic(self, daily_ic: np.ndarray) -> Tuple[bool, float, float, float]:
        """
        Gate 2: Global IC magnitude.
        
        Returns: (passed, ic_mean, ic_abs, ic_t_stat)
        
        Note: t-stat uses Newey-West HAC standard errors to account for
        serial correlation in daily ICs, as per spec.
        """
        valid = ~np.isnan(daily_ic)
        if valid.sum() < self.config.min_dates:
            return False, np.nan, np.nan, np.nan
        
        ic_clean = daily_ic[valid]
        ic_mean = np.mean(ic_clean)
        ic_abs = abs(ic_mean)
        
        # Use Newey-West HAC t-statistic instead of simple t-stat
        ic_t_stat = compute_newey_west_tstat(daily_ic)
        
        passed = ic_abs >= self.config.thresholds.min_ic_abs
        return passed, ic_mean, ic_abs, ic_t_stat
    
    def gate_t_stat(self, ic_t_stat: float) -> bool:
        """Gate 3: Statistical significance."""
        return abs(ic_t_stat) >= self.config.thresholds.min_t_stat
    
    def gate_sign_stability(self, daily_ic: np.ndarray, ic_mean: float) -> Tuple[bool, float]:
        """
        Gate 4: Sign stability across periods.
        
        Returns: (passed, sign_consistency)
        """
        valid = ~np.isnan(daily_ic)
        ic_clean = daily_ic[valid]
        
        if len(ic_clean) < self.config.min_dates:
            return False, np.nan
        
        n_periods = self.config.thresholds.n_periods
        period_size = len(ic_clean) // n_periods
        
        global_sign = np.sign(ic_mean)
        sign_matches = 0
        
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(ic_clean)
            period_mean = np.mean(ic_clean[start:end])
            if np.sign(period_mean) == global_sign:
                sign_matches += 1
        
        sign_consistency = sign_matches / n_periods
        passed = sign_consistency >= self.config.thresholds.min_sign_consistency
        
        return passed, sign_consistency
    
    def gate_residual_ic(self, feature_col: str) -> Tuple[bool, float]:
        """
        Gate 5: Residual IC after orthogonalizing vs benchmark (Close%-63).
        
        Per-date residualization to match IC methodology:
        1. For each date, regress feature on benchmark within that cross-section
        2. Compute IC of residuals vs target
        3. Average daily residual IC
        """
        benchmark = self.config.thresholds.residual_benchmark
        
        if benchmark not in self.panel_df.columns:
            # If benchmark not available, pass the gate (can't compute)
            return True, np.nan
        
        X = self.panel_df[feature_col].values
        B = self.panel_df[benchmark].values
        y = self.panel_df[self.config.target_column].values
        
        daily_resid_ics = []
        
        for i in range(len(self.unique_dates)):
            start, end = self._date_starts[i], self._date_ends[i]
            X_date = X[start:end]
            B_date = B[start:end]
            y_date = y[start:end]
            
            valid = ~(np.isnan(X_date) | np.isnan(B_date) | np.isnan(y_date))
            if valid.sum() < 20:
                continue
            
            X_v, B_v, y_v = X_date[valid], B_date[valid], y_date[valid]
            
            # Per-date OLS residual: X_resid = X - (alpha + beta * B)
            # where alpha = X_mean - beta * B_mean (proper OLS intercept)
            B_mean = np.mean(B_v)
            X_mean = np.mean(X_v)
            
            cov_XB = np.mean((X_v - X_mean) * (B_v - B_mean))
            var_B = np.var(B_v)
            
            if var_B < 1e-10:
                continue
            
            beta = cov_XB / var_B
            alpha = X_mean - beta * B_mean
            X_resid = X_v - (alpha + beta * B_v)
            
            # IC of residual vs target for this date
            if np.std(X_resid) < 1e-10:
                continue
            
            try:
                resid_ic, _ = spearmanr(X_resid, y_v)
                if not np.isnan(resid_ic):
                    daily_resid_ics.append(resid_ic)
            except:
                continue
        
        if len(daily_resid_ics) < self.config.min_dates:
            return False, np.nan
        
        mean_resid_ic = np.mean(daily_resid_ics)
        passed = abs(mean_resid_ic) >= self.config.thresholds.min_residual_ic
        return passed, mean_resid_ic
    
    def gate_long_tail(self, feature_col: str) -> Tuple[bool, float]:
        """
        Gate 6: Long-tail excess return (top decile beats universe).
        """
        return_col = self.config.return_column
        if return_col not in self.panel_df.columns:
            return_col = 'y_raw_21d'
            if return_col not in self.panel_df.columns:
                return True, np.nan  # Can't compute, pass
        
        X = self.panel_df[feature_col].values
        y = self.panel_df[return_col].values
        
        if HAS_NUMBA:
            spreads, top_excess = _compute_quintile_spread_numba(
                X, y, self._date_starts, self._date_ends
            )
        else:
            # Fallback Python implementation
            spreads = []
            top_excess_list = []
            for i in range(len(self.unique_dates)):
                start, end = self._date_starts[i], self._date_ends[i]
                X_date = X[start:end]
                y_date = y[start:end]
                
                valid = ~(np.isnan(X_date) | np.isnan(y_date))
                if valid.sum() < 20:
                    continue
                
                X_v = X_date[valid]
                y_v = y_date[valid]
                
                # Use quintiles (20% / 80%) per spec
                q5_thresh = np.percentile(X_v, 80)  # Top 20% = Q5
                q1_thresh = np.percentile(X_v, 20)  # Bottom 20% = Q1
                
                q5_ret = np.mean(y_v[X_v >= q5_thresh])
                q1_ret = np.mean(y_v[X_v <= q1_thresh])
                universe_mean = np.mean(y_v)
                
                spreads.append(q5_ret - q1_ret)
                top_excess_list.append(q5_ret - universe_mean)
            
            top_excess = np.array(top_excess_list)
        
        valid_excess = top_excess[~np.isnan(top_excess)]
        if len(valid_excess) < 50:
            return False, np.nan
        
        mean_excess = np.mean(valid_excess)
        passed = mean_excess > self.config.thresholds.min_long_tail_excess
        
        return passed, mean_excess
    
    # =========================================================================
    # Scoring Metrics
    # =========================================================================
    
    def compute_spread_sharpe(self, feature_col: str) -> float:
        """
        Compute Spread Sharpe = Mean(Q5 - Q1) / Std(Q5 - Q1).
        
        Per spec: Uses quintiles (top 20% vs bottom 20%).
        """
        return_col = self.config.return_column
        if return_col not in self.panel_df.columns:
            return_col = 'y_raw_21d'
            if return_col not in self.panel_df.columns:
                return np.nan
        
        X = self.panel_df[feature_col].values
        y = self.panel_df[return_col].values
        
        if HAS_NUMBA:
            spreads, _ = _compute_quintile_spread_numba(
                X, y, self._date_starts, self._date_ends
            )
        else:
            spreads = []
            for i in range(len(self.unique_dates)):
                start, end = self._date_starts[i], self._date_ends[i]
                X_date = X[start:end]
                y_date = y[start:end]
                
                valid = ~(np.isnan(X_date) | np.isnan(y_date))
                if valid.sum() < 20:
                    continue
                
                X_v = X_date[valid]
                y_v = y_date[valid]
                
                # Use quintiles (20% / 80%) per spec
                q5_thresh = np.percentile(X_v, 80)
                q1_thresh = np.percentile(X_v, 20)
                
                q5_ret = np.mean(y_v[X_v >= q5_thresh])
                q1_ret = np.mean(y_v[X_v <= q1_thresh])
                spreads.append(q5_ret - q1_ret)
            
            spreads = np.array(spreads)
        
        valid_spreads = spreads[~np.isnan(spreads)]
        if len(valid_spreads) < 50:
            return np.nan
        
        spread_mean = np.mean(valid_spreads)
        spread_std = np.std(valid_spreads)
        
        if spread_std < 1e-10:
            return np.nan
        
        return spread_mean / spread_std
    
    def compute_turnover(self, feature_col: str) -> float:
        """
        Compute average daily turnover in top decile.
        
        Turnover = fraction of top decile holdings that change day-to-day.
        Uses integer-mapped ticker IDs for numba-accelerated computation.
        """
        X = self.panel_df[feature_col].values
        
        if HAS_NUMBA:
            # Use numba-accelerated version with precomputed ticker IDs
            return _compute_turnover_numba_v2(
                X,
                self._ticker_ids,
                self._date_starts,
                self._date_ends,
                top_pct=0.1
            )
        else:
            # Fallback: Pure Python with set-based overlap
            tickers = self.panel_df.index.get_level_values('Ticker').values
            prev_top_tickers = set()
            turnovers = []
            
            for i in range(len(self.unique_dates)):
                start, end = self._date_starts[i], self._date_ends[i]
                X_date = X[start:end]
                tickers_date = tickers[start:end]
                
                valid_idx = np.where(~np.isnan(X_date))[0]
                if len(valid_idx) < 20:
                    continue
                
                X_valid = X_date[valid_idx]
                tickers_valid = tickers_date[valid_idx]
                
                sorted_idx = np.argsort(X_valid)
                top_n = max(1, int(len(valid_idx) * 0.1))  # Top 10%
                
                # Get top decile TICKERS (not indices)
                curr_top_tickers = set(tickers_valid[sorted_idx[-top_n:]])
                
                if prev_top_tickers:
                    overlap = len(curr_top_tickers & prev_top_tickers)
                    max_size = max(len(curr_top_tickers), len(prev_top_tickers))
                    turnover = 1.0 - (overlap / max_size)
                    turnovers.append(turnover)
                
                prev_top_tickers = curr_top_tickers
            
            return np.mean(turnovers) if turnovers else np.nan
    
    def compute_stress_ic(self, feature_col: str) -> float:
        """
        Compute IC during stress periods (VT return < 0).
        """
        # Try to get benchmark return column
        bench_ret_col = self.config.benchmark_return_column
        if bench_ret_col not in self.panel_df.columns:
            # Try to find VT return
            for col in ['FwdRet_21_VT', 'FwdRet_VT', 'ret_VT']:
                if col in self.panel_df.columns:
                    bench_ret_col = col
                    break
            else:
                # Can't compute stress IC without benchmark
                return np.nan
        
        daily_ic = self.compute_daily_ic(feature_col)
        
        # Compute daily benchmark return (aggregate)
        stress_ics = []
        
        for i, date in enumerate(self.unique_dates):
            start, end = self._date_starts[i], self._date_ends[i]
            
            if np.isnan(daily_ic[i]):
                continue
            
            bench_vals = self.panel_df[bench_ret_col].values[start:end]
            bench_mean = np.nanmean(bench_vals)
            
            # Stress period = benchmark return < 0
            if bench_mean < 0:
                stress_ics.append(daily_ic[i])
        
        if len(stress_ics) < 20:
            return np.nan
        
        return np.mean(stress_ics)
    
    # =========================================================================
    # Full Feature Analysis
    # =========================================================================
    
    def analyze_feature(self, feature_col: str) -> AlphaFeatureResult:
        """
        Run complete analysis for a single feature.
        
        Applies all gates in order, computing scores only for passing features.
        """
        result = AlphaFeatureResult(feature=feature_col)
        result.family = classify_feature_family(feature_col)
        
        # Gate 1: Data Quality
        dq_result = self.data_quality_gate.check_feature(
            self.panel_df[feature_col].values, feature_col
        )
        result.passed_data_quality = dq_result.passed
        result.coverage = dq_result.coverage
        result.outlier_frac = dq_result.outlier_frac
        
        if not dq_result.passed:
            result.failure_reason = f"data_quality: {dq_result.reason}"
            return result
        
        # Gate 2 & 3: IC and t-stat
        daily_ic = self.compute_daily_ic(feature_col)
        passed_ic, ic_mean, ic_abs, ic_t_stat = self.gate_global_ic(daily_ic)
        
        result.ic_mean = ic_mean
        result.ic_abs = ic_abs
        result.ic_t_stat = ic_t_stat
        result.passed_ic_gate = passed_ic
        result.passed_tstat_gate = self.gate_t_stat(ic_t_stat)
        
        if not result.passed_ic_gate:
            result.failure_reason = f"ic_gate: |IC|={ic_abs:.4f} < {self.config.thresholds.min_ic_abs}"
            return result
        
        if not result.passed_tstat_gate:
            result.failure_reason = f"tstat_gate: t={ic_t_stat:.2f} < {self.config.thresholds.min_t_stat}"
            return result
        
        # Gate 4: Sign stability
        passed_stability, sign_consistency = self.gate_sign_stability(daily_ic, ic_mean)
        result.sign_consistency = sign_consistency
        result.passed_stability_gate = passed_stability
        
        if not passed_stability:
            result.failure_reason = f"stability_gate: sign_consistency={sign_consistency:.2f}"
            return result
        
        # Gate 5: Residual IC
        passed_residual, residual_ic = self.gate_residual_ic(feature_col)
        result.residual_ic = residual_ic
        result.passed_residual_ic_gate = passed_residual
        
        if not passed_residual:
            result.failure_reason = f"residual_ic_gate: residual_ic={residual_ic:.4f}"
            return result
        
        # Gate 6: Long-tail excess
        passed_long_tail, long_tail_excess = self.gate_long_tail(feature_col)
        result.long_tail_excess = long_tail_excess
        result.passed_long_tail_gate = passed_long_tail
        
        if not passed_long_tail:
            result.failure_reason = f"long_tail_gate: excess={long_tail_excess:.4f}"
            return result
        
        # All gates passed!
        result.passed_all_gates = True
        
        # Compute scoring metrics
        result.spread_sharpe = self.compute_spread_sharpe(feature_col)
        result.turnover = self.compute_turnover(feature_col)
        result.stress_ic = self.compute_stress_ic(feature_col)
        
        return result
    
    def analyze_features(
        self,
        feature_cols: List[str],
        parallel: bool = True
    ) -> Dict[str, AlphaFeatureResult]:
        """
        Analyze multiple features.
        
        Args:
            feature_cols: List of feature column names
            parallel: Use parallel processing
            
        Returns:
            Dict mapping feature name to result
        """
        if self.config.verbose:
            print(f"[AlphaSelector] Analyzing {len(feature_cols)} features...")
        
        if parallel and self.config.n_jobs != 1:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self.analyze_feature)(col) for col in feature_cols
            )
        else:
            results = [self.analyze_feature(col) for col in feature_cols]
        
        self.results = {r.feature: r for r in results}
        
        passed = [r for r in results if r.passed_all_gates]
        if self.config.verbose:
            print(f"[AlphaSelector] {len(passed)}/{len(feature_cols)} passed all gates")
        
        return self.results
    
    # =========================================================================
    # Composite Scoring
    # =========================================================================
    
    def compute_composite_scores(self) -> Dict[str, float]:
        """
        Compute composite scores for features that passed all gates.
        
        Returns dict mapping feature -> composite score.
        """
        passed_features = [f for f, r in self.results.items() if r.passed_all_gates]
        
        if len(passed_features) == 0:
            return {}
        
        # Collect metrics
        metrics = {
            'ic_abs': {},
            'long_tail_excess': {},
            'spread_sharpe': {},
            'turnover': {},
            'stress_ic': {},
        }
        
        for f in passed_features:
            r = self.results[f]
            metrics['ic_abs'][f] = r.ic_abs
            metrics['long_tail_excess'][f] = r.long_tail_excess
            metrics['spread_sharpe'][f] = r.spread_sharpe
            metrics['turnover'][f] = r.turnover
            metrics['stress_ic'][f] = r.stress_ic
        
        # Convert to percentiles (0-100)
        def to_percentile(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
            items = [(f, v) for f, v in values.items() if not np.isnan(v)]
            if len(items) == 0:
                return {f: 50.0 for f in values}
            
            sorted_items = sorted(items, key=lambda x: x[1], reverse=higher_is_better)
            n = len(sorted_items)
            percentiles = {}
            for rank, (f, v) in enumerate(sorted_items):
                percentiles[f] = 100 * (1 - rank / n)  # Rank 0 = 100th percentile
            
            # Fill NaN with median
            for f, v in values.items():
                if np.isnan(v):
                    percentiles[f] = 50.0
            
            return percentiles
        
        pct_ic = to_percentile(metrics['ic_abs'], higher_is_better=True)
        pct_long_tail = to_percentile(metrics['long_tail_excess'], higher_is_better=True)
        pct_spread = to_percentile(metrics['spread_sharpe'], higher_is_better=True)
        pct_turnover = to_percentile(metrics['turnover'], higher_is_better=False)  # Lower is better
        pct_stress = to_percentile(metrics['stress_ic'], higher_is_better=True)
        
        # Compute composite score
        weights = self.config.thresholds
        scores = {}
        
        for f in passed_features:
            score = (
                weights.score_weight_ic * pct_ic[f] +
                weights.score_weight_long_tail * pct_long_tail[f] +
                weights.score_weight_spread_sharpe * pct_spread[f] +
                weights.score_weight_turnover * pct_turnover[f] +
                weights.score_weight_stress_ic * pct_stress[f]
            )
            scores[f] = score
            self.results[f].composite_score = score
        
        # Compute percentile rank of composite score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_scores)
        for rank, (f, _) in enumerate(sorted_scores):
            self.results[f].percentile_rank = 100 * (1 - rank / n)
        
        return scores
    
    # =========================================================================
    # Redundancy Filtering
    # =========================================================================
    
    def apply_redundancy_filters(
        self,
        scores: Dict[str, float]
    ) -> List[str]:
        """
        Apply within-family and cross-family redundancy filters.
        
        Returns list of selected features.
        """
        passed_features = list(scores.keys())
        
        if len(passed_features) == 0:
            return []
        
        # Select top N before redundancy
        top_n = self.config.thresholds.top_n_before_redundancy
        sorted_features = sorted(passed_features, key=lambda f: scores[f], reverse=True)
        top_features = sorted_features[:top_n]
        
        if self.config.verbose:
            print(f"[AlphaSelector] Top {len(top_features)} features before redundancy")
        
        # Group by family
        families = get_family_features(top_features)
        
        # Within-family filtering
        selected_by_family = []
        for family, family_features in families.items():
            if not family_features:
                continue
            
            selected = self.redundancy_filter.filter_within_family(
                self.panel_df, family_features, scores, 
                verbose=self.config.verbose
            )
            selected_by_family.extend(selected)
            
            if self.config.verbose and len(selected) < len(family_features):
                print(f"  {family}: {len(family_features)} -> {len(selected)}")
        
        # Cross-family filtering
        protected = set(self.config.thresholds.forced_features)
        selected = self.redundancy_filter.filter_cross_family(
            self.panel_df, selected_by_family, scores, protected,
            verbose=self.config.verbose
        )
        
        # Ensure forced features are included
        for forced in self.config.thresholds.forced_features:
            if forced in self.panel_df.columns and forced not in selected:
                selected.append(forced)
        
        return selected
    
    # =========================================================================
    # Run Full Pipeline
    # =========================================================================
    
    def run_pipeline(
        self,
        feature_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Run the complete alpha feature selection pipeline.
        
        Steps:
        1. Filter to alpha features only
        2. Data quality gate
        3. Hard gates (IC, t-stat, stability, residual IC, long-tail)
        4. Composite scoring
        5. Redundancy filtering
        6. Final pool checks
        
        Args:
            feature_cols: Feature columns to analyze (default: auto-detect)
            
        Returns:
            List of selected alpha features
        """
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("ALPHA FEATURE SELECTION PIPELINE")
            print("=" * 70)
        
        # Step 0: Get feature columns
        if feature_cols is None:
            all_cols = self.panel_df.columns.tolist()
            # Exclude known non-features
            exclude = {'Date', 'Ticker', 'Close', 'Open', 'High', 'Low', 'Volume',
                       'FwdRet_1', 'FwdRet_5', 'FwdRet_21', 'y_raw_21d', 'y_resid_z_21d',
                       'ADV_63', 'ADV_63_Rank'}
            feature_cols = [c for c in all_cols if c not in exclude]
        
        # Step 1: Filter to alpha features only
        alpha_features = get_alpha_features(feature_cols)
        if self.config.verbose:
            print(f"\nStep 1: {len(alpha_features)}/{len(feature_cols)} are alpha features")
        
        # Step 2: Precompute daily ICs
        if self.config.verbose:
            print(f"\nStep 2: Precomputing daily ICs...")
        self.precompute_daily_ics(alpha_features)
        
        # Step 3: Analyze features (gates)
        if self.config.verbose:
            print(f"\nStep 3: Running hard gates...")
        self.analyze_features(alpha_features)
        
        passed = [f for f, r in self.results.items() if r.passed_all_gates]
        if self.config.verbose:
            print(f"  -> {len(passed)} features passed all gates")
        
        if len(passed) == 0:
            print("[AlphaSelector] WARNING: No features passed all gates!")
            return []
        
        # Step 4: Composite scoring
        if self.config.verbose:
            print(f"\nStep 4: Computing composite scores...")
        scores = self.compute_composite_scores()
        
        # Step 5: Redundancy filtering
        if self.config.verbose:
            print(f"\nStep 5: Applying redundancy filters...")
        selected = self.apply_redundancy_filters(scores)
        
        # Step 6: Final pool checks
        if self.config.verbose:
            print(f"\nStep 6: Final pool checks...")
        pool_stats = self.redundancy_filter.compute_final_pool_stats(
            self.panel_df, selected
        )
        
        if self.config.verbose:
            print(f"  Max pairwise corr: {pool_stats['max_pairwise_corr']:.3f} "
                  f"(target < {self.config.thresholds.final_max_pairwise_corr})")
            print(f"  Mean pairwise corr: {pool_stats['mean_pairwise_corr']:.3f} "
                  f"(target < {self.config.thresholds.final_mean_pairwise_corr})")
            print(f"  Condition number: {pool_stats['condition_number']:.1f} "
                  f"(target < {self.config.thresholds.final_max_condition_number})")
        
        # Final summary
        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print(f"SELECTED: {len(selected)} alpha features")
            print(f"{'=' * 70}")
            
            # Show by family
            families = get_family_features(selected)
            for family, feats in sorted(families.items(), key=lambda x: -len(x[1])):
                if feats:
                    print(f"  {family}: {len(feats)}")
        
        return selected
    
    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save analysis results to parquet."""
        output_path = output_path or self.config.output_dir / "alpha_selection_results.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        rows = []
        for f, r in self.results.items():
            rows.append({
                'feature': r.feature,
                'family': r.family,
                'passed_all_gates': r.passed_all_gates,
                'failure_reason': r.failure_reason,
                'coverage': r.coverage,
                'ic_mean': r.ic_mean,
                'ic_abs': r.ic_abs,
                'ic_t_stat': r.ic_t_stat,
                'sign_consistency': r.sign_consistency,
                'residual_ic': r.residual_ic,
                'long_tail_excess': r.long_tail_excess,
                'spread_sharpe': r.spread_sharpe,
                'turnover': r.turnover,
                'stress_ic': r.stress_ic,
                'composite_score': r.composite_score,
                'percentile_rank': r.percentile_rank,
            })
        
        df = pd.DataFrame(rows)
        df.to_parquet(output_path)
        
        if self.config.verbose:
            print(f"[AlphaSelector] Saved results to {output_path}")
        
        return output_path
