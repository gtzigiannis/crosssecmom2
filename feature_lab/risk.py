"""
Risk Feature Selection Pipeline.

Implements the feature_risk.md specification:
1. Risk Target Construction (RV_1d, RV_5d, TailEvent_1d, LiqTarget)
2. Feature-to-Risk Mapping
3. RiskScore Computation
4. Family-Based Selection with Redundancy Filtering
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

from .config import RiskConfig, RiskThresholds, RISK_FAMILIES, get_risk_family
from .base import (
    DataQualityGate,
    RedundancyFilter,
    get_risk_features,
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


# =============================================================================
# Risk Feature Result
# =============================================================================

@dataclass
class RiskFeatureResult:
    """Complete analysis result for a single risk feature."""
    feature: str
    
    # Gate results
    passed_data_quality: bool = False
    passed_sign_stability: bool = False   # NEW: Sign stability hard gate
    passed_all_gates: bool = False
    
    # Sign stability details
    global_corr_vs_target: float = np.nan  # Global correlation with primary risk target
    segment_corrs: Optional[List[float]] = None  # Correlation per segment
    
    # Metrics per target
    ic_vol_1d: float = np.nan
    ic_vol_5d: float = np.nan
    ic_tail: float = np.nan
    roc_auc_tail: float = np.nan   # ROC AUC for tail event prediction
    
    # Scores
    vol_score: float = np.nan      # max(|IC_RV1|, |IC_RV5|)
    tail_score: float = np.nan     # ROC AUC (rescaled: AUC - 0.5 to get 0-0.5 range)
    stab_score: float = np.nan     # Sign consistency across segments
    risk_score: float = np.nan     # 0.4*Vol + 0.4*Tail + 0.2*Stab
    
    # Meta
    coverage: float = np.nan
    family: str = 'other'
    target_type: str = 'volatility'  # volatility, tail, liquidity
    
    # Failure
    failure_reason: Optional[str] = None


# =============================================================================
# Numba-optimized computations
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_realized_vol_numba(
    returns: np.ndarray,
    window: int,
    forward: bool = True
) -> np.ndarray:
    """
    Compute FORWARD-looking realized volatility.
    
    For forward=True (default):
        RV_1d(t) = |r_{t+1}| * sqrt(252)
        RV_5d(t) = sqrt(sum(r_{t+i}^2 for i=1..5) / 5) * sqrt(252)
    
    This aligns with the spec: risk targets are forward-looking.
    """
    n = len(returns)
    rv = np.full(n, np.nan)
    
    if forward:
        # Forward-looking RV
        for i in range(n - window):
            sum_sq = 0.0
            count = 0
            for j in range(i + 1, i + 1 + window):  # FORWARD: t+1 to t+window
                if not np.isnan(returns[j]):
                    sum_sq += returns[j] ** 2
                    count += 1
            
            if count >= window // 2:
                rv[i] = np.sqrt(sum_sq / count) * np.sqrt(252)
    else:
        # Backward-looking RV (historical)
        for i in range(window - 1, n):
            sum_sq = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(returns[j]):
                    sum_sq += returns[j] ** 2
                    count += 1
            
            if count >= window // 2:
                rv[i] = np.sqrt(sum_sq / count) * np.sqrt(252)
    
    return rv


@jit(nopython=True, cache=True)
def _compute_tail_events_numba(
    returns: np.ndarray,
    lookback: int,
    k_sigma: float
) -> np.ndarray:
    """
    Compute FORWARD-looking LEFT-TAIL event indicator.
    
    TailEvent_1d(t) = 1 if r_{t+1} < -k * rolling_std(r_t) else 0
    
    Per spec: LEFT tail only (large negative returns).
    Rolling std uses lookback days BEFORE current date.
    The event is whether NEXT day return is in the left tail.
    """
    n = len(returns)
    tail = np.zeros(n)
    
    for i in range(lookback, n - 1):  # -1 because we look at t+1
        # Compute rolling std from lookback period BEFORE i
        sum_sq = 0.0
        sum_r = 0.0
        count = 0
        for j in range(i - lookback, i):
            if not np.isnan(returns[j]):
                sum_r += returns[j]
                sum_sq += returns[j] ** 2
                count += 1
        
        if count < lookback // 2:
            continue
        
        mean = sum_r / count
        var = sum_sq / count - mean ** 2
        if var < 1e-10:
            continue
        
        std = np.sqrt(var)
        
        # Check if NEXT DAY (t+1) return is in LEFT TAIL (negative extreme)
        if not np.isnan(returns[i + 1]):
            if returns[i + 1] < -k_sigma * std:  # LEFT tail only
                tail[i] = 1.0
    
    return tail


@jit(nopython=True, cache=True)
def _compute_turnover_for_ranking_numba(
    X: np.ndarray,
    date_starts: np.ndarray,
    date_ends: np.ndarray,
    top_pct: float = 0.2
) -> float:
    """
    Compute average turnover in top quintile rankings.
    
    Used for stability score: Stab = 1 - Turnover
    """
    n_dates = len(date_starts)
    if n_dates < 2:
        return np.nan
    
    turnovers = np.empty(n_dates - 1)
    valid_count = 0
    
    prev_top_set = np.empty(0, dtype=np.int64)
    
    for d in range(n_dates):
        start = date_starts[d]
        end = date_ends[d]
        X_date = X[start:end]
        
        # Get valid indices
        valid_idx_list = []
        for i in range(len(X_date)):
            if not np.isnan(X_date[i]):
                valid_idx_list.append(i)
        
        n_valid = len(valid_idx_list)
        if n_valid < 20:
            continue
        
        # Sort to get top quintile
        X_values = np.empty(n_valid)
        valid_idx = np.empty(n_valid, dtype=np.int64)
        for i, idx in enumerate(valid_idx_list):
            X_values[i] = X_date[idx]
            valid_idx[i] = idx
        
        sorted_idx = np.argsort(X_values)
        top_n = max(1, int(n_valid * top_pct))
        
        curr_top_set = np.empty(top_n, dtype=np.int64)
        for i in range(top_n):
            curr_top_set[i] = valid_idx[sorted_idx[n_valid - 1 - i]]
        
        if d > 0 and len(prev_top_set) > 0:
            overlap = 0
            for i in range(len(curr_top_set)):
                for j in range(len(prev_top_set)):
                    if curr_top_set[i] == prev_top_set[j]:
                        overlap += 1
                        break
            
            max_size = max(len(curr_top_set), len(prev_top_set))
            turnover = 1.0 - (overlap / max_size)
            turnovers[valid_count] = turnover
            valid_count += 1
        
        prev_top_set = curr_top_set
    
    if valid_count == 0:
        return np.nan
    
    return np.mean(turnovers[:valid_count])


# =============================================================================
# RiskTargetBuilder
# =============================================================================

class RiskTargetBuilder:
    """
    Constructs risk targets for the risk feature pipeline.
    
    Targets:
    - RV_1d: 1-day forward realized volatility
    - RV_5d: 5-day forward realized volatility
    - TailEvent_1d: Binary indicator of next-day tail event
    - LiqTarget: Liquidity regime (optional)
    """
    
    def __init__(
        self,
        panel_df: pd.DataFrame,
        config: Optional[RiskConfig] = None
    ):
        self.config = config or RiskConfig()
        self.panel_df = panel_df
        
        if isinstance(panel_df.index, pd.MultiIndex):
            self.dates = panel_df.index.get_level_values('Date')
            self.unique_dates = pd.DatetimeIndex(self.dates.unique()).sort_values()
        else:
            raise ValueError("Panel must have (Date, Ticker) MultiIndex")
        
        self._targets_computed = False
    
    def compute_targets(self) -> Dict[str, np.ndarray]:
        """
        Compute risk targets from returns PER TICKER.
        
        CRITICAL: Risk targets must be computed per-asset, not on flattened panel.
        For each ticker's time series:
        - RV_1d(t) = |r_{t+1}| * sqrt(252)
        - RV_5d(t) = sqrt(sum r_{t+i}^2 for i=1..5) / 5) * sqrt(252)
        - TailEvent_1d(t) = 1 if r_{t+1} < -k * rolling_σ(lookback)
        
        Uses config.return_column as the daily return source.
        Falls back to common column names if config column not found.
        
        Returns dict of target arrays aligned with panel index.
        """
        # Use config.return_column first (respect configuration)
        daily_ret_col = None
        
        if self.config.return_column in self.panel_df.columns:
            daily_ret_col = self.config.return_column
        else:
            # Fallback: search common daily return column names
            fallback_cols = ['DailyRet', 'ret_1d', 'FwdRet_1', 'Close%-1']
            for col in fallback_cols:
                if col in self.panel_df.columns:
                    daily_ret_col = col
                    import warnings
                    warnings.warn(
                        f"Config return_column '{self.config.return_column}' not found. "
                        f"Falling back to '{col}'."
                    )
                    break
        
        if daily_ret_col is None:
            # Compute from Close if available
            if 'Close' in self.panel_df.columns:
                returns_series = self.panel_df['Close'].groupby('Ticker').pct_change()
            else:
                raise ValueError("Cannot compute risk targets: no return column found")
        else:
            returns_series = self.panel_df[daily_ret_col]
        
        # Ensure returns_series is a Series with the panel's MultiIndex
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series, index=self.panel_df.index)
        
        targets = {}
        k_sigma = self.config.thresholds.tail_event_k_sigma
        lookback = self.config.thresholds.tail_lookback
        
        # =================================================================
        # Compute all risk targets PER TICKER using groupby
        # =================================================================
        
        def _compute_rv_1d_per_ticker(ret: pd.Series) -> pd.Series:
            """RV_1d(t) = |r_{t+1}| * sqrt(252)"""
            return ret.shift(-1).abs() * np.sqrt(252)
        
        def _compute_rv_5d_per_ticker(ret: pd.Series) -> pd.Series:
            """RV_5d(t) = sqrt(mean of r_{t+i}^2 for i=1..5) * sqrt(252)"""
            # Forward-looking: use shift(-1) to shift returns, then rolling
            fwd_ret = ret.shift(-1)  # r_{t+1}
            # Rolling sum of squares over 5 periods
            sq_sum = (fwd_ret ** 2).rolling(window=5, min_periods=3).mean()
            # Shift back by 4 so RV_5d(t) uses r_{t+1} to r_{t+5}
            rv_5d = np.sqrt(sq_sum.shift(-4)) * np.sqrt(252)
            return rv_5d
        
        def _compute_tail_event_per_ticker(ret: pd.Series) -> pd.Series:
            """TailEvent_1d(t) = 1 if r_{t+1} < -k * rolling_std(lookback)"""
            # Rolling std uses lookback days BEFORE current date
            rolling_std = ret.rolling(window=lookback, min_periods=lookback // 2).std()
            threshold = k_sigma * rolling_std
            next_return = ret.shift(-1)  # r_{t+1}
            # LEFT tail only: next_return < -threshold
            tail = (next_return < -threshold).astype(float)
            return tail
        
        # Group by ticker (level=1 in MultiIndex) and apply per-ticker
        # The panel is sorted by (Date, Ticker), so we need to sort by ticker first
        # for the rolling operations to work correctly along time
        
        # Get Ticker from MultiIndex level
        tickers = self.panel_df.index.get_level_values('Ticker')
        
        # Sort by (Ticker, Date) for per-ticker time series operations
        panel_sorted = returns_series.sort_index(level=['Ticker', 'Date'])
        
        # Apply per-ticker functions
        rv_1d_sorted = panel_sorted.groupby(level='Ticker', group_keys=False).apply(
            _compute_rv_1d_per_ticker
        )
        rv_5d_sorted = panel_sorted.groupby(level='Ticker', group_keys=False).apply(
            _compute_rv_5d_per_ticker
        )
        tail_sorted = panel_sorted.groupby(level='Ticker', group_keys=False).apply(
            _compute_tail_event_per_ticker
        )
        
        # Re-sort back to original (Date, Ticker) order and extract values
        targets['RV_1d'] = rv_1d_sorted.reindex(self.panel_df.index).values
        targets['RV_5d'] = rv_5d_sorted.reindex(self.panel_df.index).values
        targets['TailEvent_1d'] = tail_sorted.reindex(self.panel_df.index).values
        
        # LiqTarget: Liquidity regime (if ADV available)
        # This is cross-sectional (per date), so no per-ticker issue
        if 'ADV_63' in self.panel_df.columns:
            adv = self.panel_df['ADV_63']
            adv_rank = adv.groupby(level='Date').rank(pct=True)
            liq_target = (adv_rank < 0.2).astype(float)  # Low liquidity regime
            targets['LiqTarget'] = liq_target.values
        
        self._targets_computed = True
        self._targets = targets
        
        return targets
    
    def get_target(self, target_name: str) -> np.ndarray:
        """Get a specific risk target."""
        if not self._targets_computed:
            self.compute_targets()
        return self._targets.get(target_name, None)


# =============================================================================
# RiskSelector
# =============================================================================

class RiskSelector:
    """
    Risk feature selection pipeline.
    
    Implements the full feature_risk.md specification:
    1. Compute risk targets
    2. Compute feature-to-target IC for each risk target
    3. Compute RiskScore = 0.4*Vol + 0.4*Tail + 0.2*Stab
    4. Family-based selection with redundancy filtering
    """
    
    def __init__(
        self,
        panel_df: pd.DataFrame,
        config: Optional[RiskConfig] = None
    ):
        self.config = config or RiskConfig()
        self.panel_df = panel_df
        
        if isinstance(panel_df.index, pd.MultiIndex):
            self.dates = panel_df.index.get_level_values('Date')
            self.unique_dates = pd.DatetimeIndex(self.dates.unique()).sort_values()
        else:
            raise ValueError("Panel must have (Date, Ticker) MultiIndex")
        
        # Precompute date indices for numba
        self._precompute_date_indices()
        
        # Target builder
        self.target_builder = RiskTargetBuilder(panel_df, config)
        
        # Data quality gate
        self.data_quality_gate = DataQualityGate(
            min_coverage=self.config.thresholds.min_coverage,
            max_outlier_frac=self.config.thresholds.max_outlier_frac,
            outlier_sigma=self.config.thresholds.outlier_sigma,
            n_jobs=self.config.n_jobs
        )
        
        # Redundancy filter
        self.redundancy_filter = RedundancyFilter(
            within_family_max_corr=self.config.thresholds.within_family_max_corr,
            cross_family_max_corr=self.config.thresholds.cross_family_max_corr,
            max_per_family=self.config.thresholds.max_per_family,
            n_jobs=self.config.n_jobs
        )
        
        # Caches
        self._ic_cache: Dict[Tuple[str, str], float] = {}
        
        # Results storage
        self.results: Dict[str, RiskFeatureResult] = {}
        
        if self.config.verbose:
            print(f"[RiskSelector] Initialized with {len(self.unique_dates)} dates, "
                  f"{len(self.panel_df)} rows (Numba: {HAS_NUMBA})")
    
    def _precompute_date_indices(self):
        """Precompute date start/end indices."""
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
    
    # =========================================================================
    # IC Computation
    # =========================================================================
    
    def compute_ic_vs_target(
        self,
        feature_col: str,
        target: np.ndarray,
        target_name: str
    ) -> float:
        """
        Compute average daily IC of feature vs risk target.
        """
        cache_key = (feature_col, target_name)
        if cache_key in self._ic_cache:
            return self._ic_cache[cache_key]
        
        X = self.panel_df[feature_col].values
        y = target
        
        daily_ics = []
        
        for i in range(len(self.unique_dates)):
            start, end = self._date_starts[i], self._date_ends[i]
            X_date = X[start:end]
            y_date = y[start:end]
            
            valid = ~(np.isnan(X_date) | np.isnan(y_date))
            if valid.sum() < 20:
                continue
            
            X_v = X_date[valid]
            y_v = y_date[valid]
            
            if np.std(X_v) < 1e-10 or np.std(y_v) < 1e-10:
                continue
            
            try:
                ic, _ = spearmanr(X_v, y_v)
                if not np.isnan(ic):
                    daily_ics.append(ic)
            except:
                continue
        
        if len(daily_ics) < self.config.min_dates:
            ic_mean = np.nan
        else:
            ic_mean = np.mean(daily_ics)
        
        self._ic_cache[cache_key] = ic_mean
        return ic_mean
    
    def compute_roc_auc_vs_tail(self, feature_col: str, tail_target: np.ndarray) -> float:
        """
        Compute ROC AUC for predicting tail events.
        
        Per spec: For binary TailEvent targets, use ROC AUC instead of IC
        because it's a more appropriate metric for classification.
        
        Returns mean daily ROC AUC (0.5 = random, 1.0 = perfect).
        """
        from sklearn.metrics import roc_auc_score
        
        X = self.panel_df[feature_col].values
        y = tail_target
        
        daily_aucs = []
        
        for i in range(len(self.unique_dates)):
            start, end = self._date_starts[i], self._date_ends[i]
            X_date = X[start:end]
            y_date = y[start:end]
            
            valid = ~(np.isnan(X_date) | np.isnan(y_date))
            if valid.sum() < 20:
                continue
            
            X_v = X_date[valid]
            y_v = y_date[valid]
            
            # Need at least 2 classes for ROC AUC
            if len(np.unique(y_v)) < 2:
                continue
            
            try:
                # Higher X should predict higher probability of tail event
                # Try both directions and take the one that's > 0.5
                auc = roc_auc_score(y_v, X_v)
                # If AUC < 0.5, flip it (feature predicts inverse)
                if auc < 0.5:
                    auc = 1 - auc
                daily_aucs.append(auc)
            except Exception:
                continue
        
        if len(daily_aucs) < self.config.min_dates:
            return np.nan
        
        return np.mean(daily_aucs)
    
    # =========================================================================
    # Sign Stability Hard Gate
    # =========================================================================
    
    def gate_sign_stability(
        self, 
        feature_col: str, 
        target_name: str = 'RV_1d'
    ) -> Tuple[bool, float, List[float]]:
        """
        Hard gate for sign stability of risk features.
        
        Ensures the feature has a consistent directional relationship with
        its risk target across time segments. This is crucial for risk controls
        that must reliably predict risk in all market regimes.
        
        Algorithm:
        1. Compute global correlation between feature and target
        2. If |global_corr| < min_threshold, reject (too weak/noisy)
        3. Split history into n_segments equal periods
        4. For each segment, compute correlation
        5. Reject if ANY segment has opposite sign AND |corr| > noise_band
        
        Args:
            feature_col: Feature column name
            target_name: Which risk target to check against ('RV_1d', 'RV_5d', 'LiqTarget')
        
        Returns:
            Tuple of (passed, global_corr, segment_corrs)
        """
        thresholds = self.config.thresholds
        n_segments = thresholds.n_stability_segments
        min_global = thresholds.min_global_corr
        noise_band = thresholds.sign_flip_noise_band
        
        # Get target array
        targets = self.target_builder._targets
        if target_name not in targets:
            targets = self.target_builder.compute_targets()
        
        target_arr = targets.get(target_name)
        if target_arr is None:
            # Fallback for liquidity
            if target_name == 'LiqTarget' and 'ADV_63' in self.panel_df.columns:
                target_arr = self.panel_df['ADV_63'].values
            else:
                return True, np.nan, []  # Can't compute, pass by default
        
        X = self.panel_df[feature_col].values
        y = target_arr
        
        # Step 1: Compute GLOBAL correlation (pooled across all dates)
        valid = ~(np.isnan(X) | np.isnan(y))
        if valid.sum() < 100:
            return False, np.nan, []
        
        global_corr, _ = spearmanr(X[valid], y[valid])
        
        if np.isnan(global_corr):
            return False, np.nan, []
        
        # Step 2: Global relevance gate
        if abs(global_corr) < min_global:
            return False, global_corr, []
        
        reference_sign = 1 if global_corr > 0 else -1
        
        # Step 3: Split into segments and compute per-segment correlations
        n_dates = len(self.unique_dates)
        if n_dates < n_segments * 10:
            # Not enough data for segmentation, pass with global only
            return True, global_corr, [global_corr]
        
        segment_size = n_dates // n_segments
        segment_corrs = []
        
        for seg in range(n_segments):
            seg_start_idx = seg * segment_size
            seg_end_idx = (seg + 1) * segment_size if seg < n_segments - 1 else n_dates
            
            # Collect data for this segment
            seg_X = []
            seg_y = []
            
            for d_idx in range(seg_start_idx, seg_end_idx):
                start, end = self._date_starts[d_idx], self._date_ends[d_idx]
                seg_X.extend(X[start:end])
                seg_y.extend(y[start:end])
            
            seg_X = np.array(seg_X)
            seg_y = np.array(seg_y)
            
            valid_seg = ~(np.isnan(seg_X) | np.isnan(seg_y))
            if valid_seg.sum() < 50:
                segment_corrs.append(np.nan)
                continue
            
            seg_corr, _ = spearmanr(seg_X[valid_seg], seg_y[valid_seg])
            segment_corrs.append(seg_corr)
        
        # Step 4: Sign stability veto
        # Reject if ANY segment has opposite sign AND exceeds noise band
        for seg_corr in segment_corrs:
            if np.isnan(seg_corr):
                continue
            
            seg_sign = 1 if seg_corr > 0 else -1
            
            # Check for sign flip outside noise band
            if seg_sign != reference_sign and abs(seg_corr) > noise_band:
                # This segment has a significant correlation with OPPOSITE sign
                return False, global_corr, segment_corrs
        
        # All segments passed - feature has stable sign relationship
        return True, global_corr, segment_corrs
    
    # =========================================================================
    # Scoring
    # =========================================================================
    
    def compute_stability_score(self, feature_col: str) -> float:
        """
        Compute stability score based on sign consistency across segments.
        
        Per spec: Split data into n_periods equal segments, compute IC in each.
        Stability = fraction of segments where sign matches global sign.
        
        E.g., if global IC is positive and 3/4 segments have positive IC,
        stability = 0.75.
        """
        # First compute global IC sign
        cache_key = (feature_col, 'RV_1d')
        if cache_key in self._ic_cache:
            global_ic = self._ic_cache[cache_key]
        else:
            targets = self.target_builder._targets
            if 'RV_1d' in targets:
                global_ic = self.compute_ic_vs_target(feature_col, targets['RV_1d'], 'RV_1d')
            else:
                # Fallback to computing from volatility-related target
                global_ic = self.compute_ic_vs_target(feature_col, None, None)
        
        if np.isnan(global_ic) or abs(global_ic) < 1e-10:
            return np.nan
        
        global_sign = 1 if global_ic > 0 else -1
        
        # Split dates into n_periods segments
        n_periods = self.config.thresholds.n_periods
        n_dates = len(self.unique_dates)
        
        if n_dates < n_periods * 10:  # Need at least 10 dates per segment
            return np.nan
        
        segment_size = n_dates // n_periods
        X = self.panel_df[feature_col].values
        
        # Try to get RV_1d target for IC computation
        targets = self.target_builder._targets
        if 'RV_1d' not in targets:
            # Compute if needed
            targets = self.target_builder.compute_targets()
        
        target_arr = targets.get('RV_1d', None)
        if target_arr is None:
            return np.nan
        
        matching_signs = 0
        
        for seg in range(n_periods):
            # Define segment date range
            seg_start_date_idx = seg * segment_size
            seg_end_date_idx = (seg + 1) * segment_size if seg < n_periods - 1 else n_dates
            
            # Collect ICs for dates in this segment
            seg_ics = []
            for d_idx in range(seg_start_date_idx, seg_end_date_idx):
                start, end = self._date_starts[d_idx], self._date_ends[d_idx]
                
                X_date = X[start:end]
                y_date = target_arr[start:end]
                
                valid = ~np.isnan(X_date) & ~np.isnan(y_date)
                if valid.sum() < 20:
                    continue
                
                # Spearman IC for this date
                ic, _ = spearmanr(X_date[valid], y_date[valid])
                if not np.isnan(ic):
                    seg_ics.append(ic)
            
            if seg_ics:
                seg_mean_ic = np.mean(seg_ics)
                seg_sign = 1 if seg_mean_ic > 0 else -1
                if seg_sign == global_sign:
                    matching_signs += 1
        
        stability = matching_signs / n_periods
        return stability
    
    def compute_risk_score(
        self,
        vol_score: float,
        tail_score: float,
        stab_score: float
    ) -> float:
        """
        Compute composite risk score.
        
        RiskScore = w_vol * Vol + w_tail * Tail + w_stab * Stab
        """
        w = self.config.thresholds
        
        # Handle NaN
        vol = vol_score if not np.isnan(vol_score) else 0
        tail = tail_score if not np.isnan(tail_score) else 0
        stab = stab_score if not np.isnan(stab_score) else 0.5
        
        return w.weight_vol * vol + w.weight_tail * tail + w.weight_stab * stab
    
    def _apply_percentile_normalization(self, results: Dict[str, 'RiskFeatureResult']) -> None:
        """
        Apply percentile normalization to vol_score, tail_score, stab_score.
        
        Per spec: Scores are converted to percentiles across all risk features
        before computing RiskScore. This ensures equal weighting regardless
        of scale differences in the underlying ICs.
        
        Only features that passed both data_quality and sign_stability gates
        are included in the percentile computation.
        
        Modifies results in-place.
        """
        # Extract scores from features that passed both hard gates
        passed_features = [
            f for f, r in results.items() 
            if r.passed_data_quality and r.passed_sign_stability
        ]
        
        if len(passed_features) < 2:
            return  # Can't compute percentiles with <2 features
        
        # Collect raw scores
        vol_scores = np.array([results[f].vol_score for f in passed_features])
        tail_scores = np.array([results[f].tail_score for f in passed_features])
        stab_scores = np.array([results[f].stab_score for f in passed_features])
        
        # Compute percentile ranks (higher is better for all)
        # Use scipy.stats.rankdata for proper handling of ties
        from scipy.stats import rankdata
        
        def to_percentile(scores: np.ndarray) -> np.ndarray:
            """Convert raw scores to percentile rank (0-1)."""
            valid = ~np.isnan(scores)
            pct = np.full(len(scores), np.nan)
            if np.sum(valid) > 1:
                ranks = rankdata(scores[valid], method='average')
                pct[valid] = (ranks - 1) / (len(ranks) - 1)  # Scale to 0-1
            return pct
        
        vol_pct = to_percentile(vol_scores)
        tail_pct = to_percentile(tail_scores)
        stab_pct = to_percentile(stab_scores)
        
        # Apply percentile scores and recompute RiskScore
        w = self.config.thresholds
        for i, f in enumerate(passed_features):
            r = results[f]
            r.vol_score = vol_pct[i] if not np.isnan(vol_pct[i]) else 0
            r.tail_score = tail_pct[i] if not np.isnan(tail_pct[i]) else 0
            r.stab_score = stab_pct[i] if not np.isnan(stab_pct[i]) else 0.5
            
            # Recompute RiskScore with percentile-normalized scores
            r.risk_score = (
                w.weight_vol * r.vol_score +
                w.weight_tail * r.tail_score +
                w.weight_stab * r.stab_score
            )
    
    # =========================================================================
    # Feature Analysis
    # =========================================================================
    
    def analyze_feature(self, feature_col: str) -> RiskFeatureResult:
        """
        Analyze a single risk feature.
        
        Computes IC vs all risk targets and overall RiskScore.
        """
        result = RiskFeatureResult(feature=feature_col)
        result.family = get_risk_family(feature_col)
        
        # Data quality gate
        dq_result = self.data_quality_gate.check_feature(
            self.panel_df[feature_col].values, feature_col
        )
        result.passed_data_quality = dq_result.passed
        result.coverage = dq_result.coverage
        
        if not dq_result.passed:
            result.failure_reason = f"data_quality: {dq_result.reason}"
            return result
        
        # Get targets
        targets = self.target_builder.compute_targets()
        
        # Sign Stability Hard Gate
        # Determine which target to use based on family
        family = result.family
        if family == 'liquidity':
            sign_target = 'LiqTarget'
        else:
            sign_target = 'RV_1d'  # Default for vol/beta/regime/tail families
        
        passed_sign, global_corr, seg_corrs = self.gate_sign_stability(
            feature_col, target_name=sign_target
        )
        result.passed_sign_stability = passed_sign
        result.global_corr_vs_target = global_corr
        result.segment_corrs = seg_corrs
        
        if not passed_sign:
            if abs(global_corr) < self.config.thresholds.min_global_corr if not np.isnan(global_corr) else True:
                result.failure_reason = f"sign_stability: |global_corr|={abs(global_corr):.4f} < {self.config.thresholds.min_global_corr} (too weak)"
            else:
                result.failure_reason = f"sign_stability: sign flip detected in segments (global={global_corr:.4f}, segments={seg_corrs})"
            return result
        
        # Compute IC vs each target
        if 'RV_1d' in targets:
            result.ic_vol_1d = self.compute_ic_vs_target(
                feature_col, targets['RV_1d'], 'RV_1d'
            )
        
        if 'RV_5d' in targets:
            result.ic_vol_5d = self.compute_ic_vs_target(
                feature_col, targets['RV_5d'], 'RV_5d'
            )
        
        if 'TailEvent_1d' in targets:
            # Keep IC for reference
            result.ic_tail = self.compute_ic_vs_target(
                feature_col, targets['TailEvent_1d'], 'TailEvent_1d'
            )
            # Compute ROC AUC for tail discrimination (primary metric per spec)
            result.roc_auc_tail = self.compute_roc_auc_vs_tail(
                feature_col, targets['TailEvent_1d']
            )
        
        # Compute component scores (raw values - will be percentile-normalized later)
        # VolScore: average of |IC| vs RV_1d and RV_5d (per spec)
        ic_1d = abs(result.ic_vol_1d) if result.ic_vol_1d is not None and not np.isnan(result.ic_vol_1d) else 0
        ic_5d = abs(result.ic_vol_5d) if result.ic_vol_5d is not None and not np.isnan(result.ic_vol_5d) else 0
        result.vol_score = (ic_1d + ic_5d) / 2  # Average per spec
        # Use ROC AUC directly as raw tail score (will be percentile-normalized)
        result.tail_score = result.roc_auc_tail if not np.isnan(result.roc_auc_tail) else 0
        result.stab_score = self.compute_stability_score(feature_col)
        
        # Compute final risk score
        result.risk_score = self.compute_risk_score(
            result.vol_score, result.tail_score, result.stab_score
        )
        
        # Check minimum threshold
        if result.risk_score >= self.config.thresholds.min_risk_score:
            result.passed_all_gates = True  # data_quality and sign_stability already passed
        else:
            result.failure_reason = f"risk_score={result.risk_score:.4f} < {self.config.thresholds.min_risk_score}"
        
        return result
    
    def analyze_features(
        self,
        feature_cols: List[str],
        parallel: bool = True
    ) -> Dict[str, RiskFeatureResult]:
        """
        Analyze multiple risk features.
        
        Steps:
        1. Analyze each feature to compute raw ICs
        2. Apply percentile normalization to scores
        3. Recompute RiskScore with normalized scores
        4. Apply min_risk_score threshold
        """
        if self.config.verbose:
            print(f"[RiskSelector] Analyzing {len(feature_cols)} features...")
        
        # Compute targets once
        self.target_builder.compute_targets()
        
        if parallel and self.config.n_jobs != 1:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self.analyze_feature)(col) for col in feature_cols
            )
        else:
            results = [self.analyze_feature(col) for col in feature_cols]
        
        self.results = {r.feature: r for r in results}
        
        # Apply percentile normalization and recompute RiskScore
        if self.config.verbose:
            print(f"[RiskSelector] Applying percentile normalization to scores...")
        self._apply_percentile_normalization(self.results)
        
        # Recheck min_risk_score threshold after normalization
        for f, r in self.results.items():
            if r.passed_data_quality and r.passed_sign_stability:
                if r.risk_score >= self.config.thresholds.min_risk_score:
                    r.passed_all_gates = True
                    r.failure_reason = None
                else:
                    r.passed_all_gates = False
                    r.failure_reason = f"risk_score={r.risk_score:.4f} < {self.config.thresholds.min_risk_score}"
        
        passed = [r for r in self.results.values() if r.passed_all_gates]
        if self.config.verbose:
            print(f"[RiskSelector] {len(passed)}/{len(feature_cols)} passed gates")
        
        return self.results
    
    # =========================================================================
    # Family-Based Selection
    # =========================================================================
    
    def select_by_family(self) -> Dict[str, List[str]]:
        """
        Select top features per family based on RiskScore.
        
        Returns dict: family -> list of selected features
        """
        passed = {f: r for f, r in self.results.items() if r.passed_all_gates}
        
        # Group by family
        families: Dict[str, List[Tuple[str, float]]] = {}
        for f, r in passed.items():
            family = r.family
            if family not in families:
                families[family] = []
            families[family].append((f, r.risk_score))
        
        # Select top N per family
        selected_by_family: Dict[str, List[str]] = {}
        max_per_family = self.config.thresholds.max_per_family
        
        for family, features in families.items():
            # Sort by risk score descending
            sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
            
            # Apply within-family redundancy filter
            family_cols = [f for f, _ in sorted_features]
            scores = {f: s for f, s in sorted_features}
            
            filtered = self.redundancy_filter.filter_within_family(
                self.panel_df, family_cols, scores,
                verbose=self.config.verbose
            )
            
            selected_by_family[family] = filtered[:max_per_family]
        
        return selected_by_family
    
    # =========================================================================
    # Run Pipeline
    # =========================================================================
    
    def run_pipeline(
        self,
        feature_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Run the complete risk feature selection pipeline.
        
        Steps:
        1. Filter to risk features only
        2. Data quality gate
        3. Compute RiskScore for each feature
        4. Family-based selection with redundancy
        5. Final pool checks
        
        Returns:
            List of selected risk features
        """
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("RISK FEATURE SELECTION PIPELINE")
            print("=" * 70)
        
        # Step 0: Get feature columns
        if feature_cols is None:
            all_cols = self.panel_df.columns.tolist()
            exclude = {'Date', 'Ticker', 'Close', 'Open', 'High', 'Low', 'Volume',
                       'FwdRet_1', 'FwdRet_5', 'FwdRet_21', 'y_raw_21d', 'y_resid_z_21d',
                       'ADV_63', 'ADV_63_Rank'}
            feature_cols = [c for c in all_cols if c not in exclude]
        
        # Step 1: Filter to risk features only
        risk_features = get_risk_features(feature_cols)
        if self.config.verbose:
            print(f"\nStep 1: {len(risk_features)}/{len(feature_cols)} are risk features")
        
        if len(risk_features) == 0:
            print("[RiskSelector] WARNING: No risk features found!")
            return []
        
        # Step 2: Analyze features
        if self.config.verbose:
            print(f"\nStep 2: Analyzing risk features...")
        self.analyze_features(risk_features)
        
        passed = [f for f, r in self.results.items() if r.passed_all_gates]
        if self.config.verbose:
            print(f"  -> {len(passed)} features passed gates")
        
        if len(passed) == 0:
            print("[RiskSelector] WARNING: No features passed gates!")
            return []
        
        # Step 3: Family-based selection
        if self.config.verbose:
            print(f"\nStep 3: Family-based selection...")
        selected_by_family = self.select_by_family()
        
        # Flatten selected features
        selected = []
        for family, feats in selected_by_family.items():
            selected.extend(feats)
            if self.config.verbose and feats:
                print(f"  {family}: {len(feats)} features")
        
        # Step 4: Cross-family redundancy (optional)
        if len(selected) > self.config.thresholds.max_total:
            scores = {f: self.results[f].risk_score for f in selected}
            selected = self.redundancy_filter.filter_cross_family(
                self.panel_df, selected, scores, protected=set(),
                verbose=self.config.verbose
            )
            selected = sorted(selected, key=lambda f: scores[f], reverse=True)
            selected = selected[:self.config.thresholds.max_total]
        
        # Step 5: Final pool stats
        if self.config.verbose:
            print(f"\nStep 5: Final pool checks...")
        pool_stats = self.redundancy_filter.compute_final_pool_stats(
            self.panel_df, selected
        )
        
        if self.config.verbose:
            print(f"  Max pairwise corr: {pool_stats['max_pairwise_corr']:.3f}")
            print(f"  Mean pairwise corr: {pool_stats['mean_pairwise_corr']:.3f}")
            print(f"  Condition number: {pool_stats['condition_number']:.1f}")
        
        # Final summary
        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print(f"SELECTED: {len(selected)} risk features")
            print(f"{'=' * 70}")
            
            # Show top features by score
            top_10 = sorted(selected, key=lambda f: self.results[f].risk_score, reverse=True)[:10]
            print("\nTop 10 by RiskScore:")
            for f in top_10:
                r = self.results[f]
                print(f"  {f}: RiskScore={r.risk_score:.4f} "
                      f"(Vol={r.vol_score:.4f}, Tail={r.tail_score:.4f}, Stab={r.stab_score:.4f})")
        
        return selected
    
    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save analysis results to parquet."""
        output_path = output_path or self.config.output_dir / "risk_selection_results.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for f, r in self.results.items():
            rows.append({
                'feature': r.feature,
                'family': r.family,
                'passed_all_gates': r.passed_all_gates,
                'failure_reason': r.failure_reason,
                'coverage': r.coverage,
                'ic_vol_1d': r.ic_vol_1d,
                'ic_vol_5d': r.ic_vol_5d,
                'ic_tail': r.ic_tail,
                'vol_score': r.vol_score,
                'tail_score': r.tail_score,
                'stab_score': r.stab_score,
                'risk_score': r.risk_score,
            })
        
        df = pd.DataFrame(rows)
        df.to_parquet(output_path)
        
        if self.config.verbose:
            print(f"[RiskSelector] Saved results to {output_path}")
        
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def create_risk_pipeline(
    panel_df: pd.DataFrame,
    verbose: bool = True,
    n_jobs: int = -1
) -> RiskSelector:
    """
    Create a RiskSelector with sensible defaults.
    """
    config = RiskConfig(verbose=verbose, n_jobs=n_jobs)
    return RiskSelector(panel_df, config)


def select_risk_features(
    panel_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> List[str]:
    """
    One-liner to select risk features from a panel.
    """
    selector = create_risk_pipeline(panel_df, verbose=verbose)
    return selector.run_pipeline(feature_cols)
