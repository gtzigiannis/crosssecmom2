"""
Cross-Sectional Momentum Feature Engineering
==========================================================

Key principles:
1. Takes config object as input (no hard-coded paths/dates)
2. Computes ADV_63 for liquidity filtering
3. NO global supervised binning (done per training window)
4. NO cross-sectional transforms here (done in walk-forward)
5. Forward returns at config.time.HOLDING_PERIOD_DAYS only
6. NO FFT/wavelet features

Output: Panel with (Date, Ticker) MultiIndex containing:
- Close (raw price)
- Raw features: returns, momentum, volatility, trend, oscillators
- ADV_63, ADV_63_Rank (liquidity)
- FwdRet_H where H = HOLDING_PERIOD_DAYS
"""

import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from config import ResearchConfig
from data_manager import download_etf_data, CrossSecMomDataManager, filter_tickers_for_research_window, clean_raw_ohlcv_data  # Import from data_manager instead

# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================
# These functions validate data availability EARLY to catch problems before
# they propagate through the pipeline.


def validate_data_availability(
    data_dict: Dict[str, pd.DataFrame],
    macro_data: Dict[str, pd.Series],
    formation_start: str,
    warmup_days: int,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate that we have sufficient data for the formation window + warmup.
    
    This check runs EARLY in the pipeline to catch data availability issues
    before they propagate and cause mysterious NaN problems downstream.
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of ticker -> OHLCV DataFrame
    macro_data : Dict[str, pd.Series]
        Dictionary of macro series (vix, yields, etc.)
    formation_start : str
        Formation window start date (YYYY-MM-DD)
    warmup_days : int
        Number of warmup days needed for rolling windows
    verbose : bool
        Print detailed diagnostics
        
    Returns
    -------
    Dict with validation results:
        - 'valid': bool - True if all checks pass
        - 'ohlcv_coverage': Dict of ticker -> earliest date
        - 'macro_coverage': Dict of macro -> earliest date
        - 'issues': List of issue descriptions
    """
    import pandas as pd
    
    formation_start_dt = pd.to_datetime(formation_start)
    required_start_dt = formation_start_dt - pd.Timedelta(days=warmup_days)
    
    results = {
        'valid': True,
        'ohlcv_coverage': {},
        'macro_coverage': {},
        'issues': [],
        'formation_start': formation_start,
        'required_start': required_start_dt.strftime('%Y-%m-%d'),
        'warmup_days': warmup_days
    }
    
    if verbose:
        print(f"\n[validate] Checking data availability...")
        print(f"[validate]   Formation start: {formation_start}")
        print(f"[validate]   Warmup required: {warmup_days} days")
        print(f"[validate]   Data must start by: {required_start_dt.strftime('%Y-%m-%d')}")
    
    # Check OHLCV data coverage
    if verbose:
        print(f"\n[validate] OHLCV data coverage:")
    
    ohlcv_issues = []
    for ticker, df in data_dict.items():
        if df.empty:
            results['ohlcv_coverage'][ticker] = None
            ohlcv_issues.append(f"{ticker}: No data")
            continue
        
        earliest_date = df.index.min()
        results['ohlcv_coverage'][ticker] = earliest_date
        
        if earliest_date > required_start_dt:
            days_short = (earliest_date - required_start_dt).days
            ohlcv_issues.append(f"{ticker}: starts {earliest_date.strftime('%Y-%m-%d')} ({days_short} days late)")
    
    if ohlcv_issues:
        if verbose:
            print(f"[validate]   ⚠ {len(ohlcv_issues)} tickers have insufficient history:")
            for issue in ohlcv_issues[:5]:
                print(f"[validate]     - {issue}")
            if len(ohlcv_issues) > 5:
                print(f"[validate]     ... and {len(ohlcv_issues) - 5} more")
        results['issues'].extend(ohlcv_issues)
    else:
        if verbose:
            print(f"[validate]   ✓ All {len(data_dict)} tickers have sufficient OHLCV history")
    
    # Check macro data coverage - THIS IS CRITICAL
    if verbose:
        print(f"\n[validate] Macro data coverage:")
    
    macro_issues = []
    for name, series in macro_data.items():
        if series is None or (hasattr(series, 'empty') and series.empty):
            results['macro_coverage'][name] = None
            macro_issues.append(f"{name}: No data")
            results['valid'] = False
            continue
        
        earliest_date = series.index.min()
        results['macro_coverage'][name] = earliest_date
        
        if earliest_date > required_start_dt:
            days_short = (earliest_date - required_start_dt).days
            macro_issues.append(f"{name}: starts {earliest_date.strftime('%Y-%m-%d')} ({days_short} days late)")
            results['valid'] = False  # Macro coverage is CRITICAL
    
    if macro_issues:
        if verbose:
            print(f"[validate]   ✗ CRITICAL: Macro data has insufficient history:")
            for issue in macro_issues:
                print(f"[validate]     - {issue}")
        results['issues'].extend([f"MACRO: {i}" for i in macro_issues])
    else:
        if verbose:
            print(f"[validate]   ✓ All macro data has sufficient history")
    
    # Summary
    if verbose:
        if results['valid']:
            print(f"\n[validate] ✓ Data validation PASSED - all sources have sufficient coverage")
        else:
            print(f"\n[validate] ✗ Data validation FAILED - see issues above")
            print(f"[validate]   This will cause NaN in features and dropped features!")
    
    return results


def trim_panel_to_formation_window(
    panel_df: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Trim panel to formation window dates only.
    
    After feature engineering, the panel contains warmup data that was needed
    for rolling window calculations. This function removes those rows, keeping
    only the formation window dates that will be used for training/testing.
    
    This is the final step that eliminates any remaining warmup NaN.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) MultiIndex or Date column
    formation_start : str
        Formation window start date (YYYY-MM-DD)
    formation_end : str
        Formation window end date (YYYY-MM-DD)
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    pd.DataFrame
        Panel trimmed to formation window dates
    """
    import pandas as pd
    
    formation_start_dt = pd.to_datetime(formation_start)
    formation_end_dt = pd.to_datetime(formation_end)
    
    # Handle MultiIndex
    has_multiindex = isinstance(panel_df.index, pd.MultiIndex)
    if has_multiindex:
        dates = panel_df.index.get_level_values('Date')
    else:
        dates = panel_df['Date']
    
    n_before = len(panel_df)
    date_range_before = f"{dates.min()} to {dates.max()}"
    
    # Filter to formation window
    mask = (dates >= formation_start_dt) & (dates <= formation_end_dt)
    panel_trimmed = panel_df.loc[mask]
    
    n_after = len(panel_trimmed)
    if has_multiindex:
        dates_after = panel_trimmed.index.get_level_values('Date')
    else:
        dates_after = panel_trimmed['Date']
    date_range_after = f"{dates_after.min()} to {dates_after.max()}"
    
    if verbose:
        n_removed = n_before - n_after
        pct_removed = n_removed / n_before * 100 if n_before > 0 else 0
        print(f"[trim] Trimming panel to formation window...")
        print(f"[trim]   Before: {n_before:,} rows ({date_range_before})")
        print(f"[trim]   After:  {n_after:,} rows ({date_range_after})")
        print(f"[trim]   Removed: {n_removed:,} warmup rows ({pct_removed:.1f}%)")
    
    return panel_trimmed


# ============================================================================
# NaN HANDLING UTILITIES
# ============================================================================
# These functions handle NaN at the SOURCE to ensure clean data downstream.
# NaN handling is done WITHOUT look-ahead bias:
# - Forward-fill with limit (no future data used)
# - Cross-sectional median imputation (per-date, no temporal leakage)
# - Drop features with excessive NaN (structural data issues)


def validate_features_nan(
    df: pd.DataFrame,
    stage: str,
    nan_threshold: float = 0.10,
    drop_high_nan: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Validate and clean NaN in feature DataFrame AFTER a processing stage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (may have NaN)
    stage : str
        Name of the processing stage (for logging)
    nan_threshold : float
        Maximum allowed NaN fraction per column (default 10%)
    drop_high_nan : bool
        If True, drop columns exceeding nan_threshold
    verbose : bool
        Print detailed diagnostics
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with high-NaN columns dropped
    """
    if df.empty:
        return df
    
    n_rows = len(df)
    n_cols_start = len(df.columns)
    
    # Calculate NaN percentage per column (excluding non-feature columns)
    exclude_cols = {'Ticker', 'Date'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    nan_pct = df[feature_cols].isna().sum() / n_rows
    high_nan_cols = nan_pct[nan_pct > nan_threshold].index.tolist()
    
    if verbose and high_nan_cols:
        print(f"  [{stage}] Found {len(high_nan_cols)} features with >{nan_threshold:.0%} NaN:", flush=True)
        # Show top 5 worst offenders
        worst = nan_pct[high_nan_cols].sort_values(ascending=False).head(5)
        for col, pct in worst.items():
            print(f"    - {col}: {pct:.1%} NaN", flush=True)
        if len(high_nan_cols) > 5:
            print(f"    ... and {len(high_nan_cols) - 5} more", flush=True)
    
    if drop_high_nan and high_nan_cols:
        df = df.drop(columns=high_nan_cols)
        if verbose:
            print(f"  [{stage}] Dropped {len(high_nan_cols)} high-NaN features, {len(df.columns)} remain", flush=True)
    
    return df


def impute_nan_cross_sectional(
    panel_df: pd.DataFrame,
    feature_cols: list,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Impute remaining NaN using cross-sectional median (per date).
    
    This method is SAFE from look-ahead bias because:
    - For each date, we only use data from that same date
    - No future information is used
    - No zero imputation (which would mess up returns)
    
    Two-pass approach:
    1. First pass: Impute with same-date cross-sectional median
    2. Second pass: For dates where ALL tickers had NaN (warmup period),
       forward-fill from the first available cross-sectional median
       (still safe - uses only past data)
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with 'Date' column or MultiIndex
    feature_cols : list
        List of feature columns to impute
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    pd.DataFrame
        Panel with NaN imputed
    """
    if panel_df.empty:
        return panel_df
    
    # Work with reset index for easier groupby
    has_multiindex = isinstance(panel_df.index, pd.MultiIndex)
    if has_multiindex:
        df = panel_df.reset_index()
    else:
        df = panel_df.copy()
    
    # Count NaN before imputation
    nan_before = df[feature_cols].isna().sum().sum()
    
    if nan_before == 0:
        if verbose:
            print(f"  [impute] No NaN to impute", flush=True)
        return panel_df
    
    # PASS 1: Impute with same-date cross-sectional median
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Check if column has NaN
        if df[col].isna().any():
            # Group by date and fill NaN with median of that date
            df[col] = df.groupby('Date')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Count NaN after pass 1
    nan_after_pass1 = df[feature_cols].isna().sum().sum()
    
    if verbose:
        n_imputed_pass1 = nan_before - nan_after_pass1
        print(f"  [impute] Pass 1: Imputed {n_imputed_pass1:,} NaN with same-date cross-sectional median", flush=True)
    
    # PASS 2: Handle dates where ALL tickers had NaN (warmup period)
    # Forward-fill ONLY (safe - uses past data only)
    # NO BFILL - that would cause look-ahead bias!
    if nan_after_pass1 > 0:
        # For each feature with remaining NaN, try forward-fill per ticker
        # This handles warmup period NaN where all tickers had NaN on early dates
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            if df[col].isna().any():
                # Forward-fill within each ticker (safe - uses past data only)
                df[col] = df.groupby('Ticker')[col].ffill()
                
                # NOTE: We do NOT use bfill() - that would cause look-ahead bias!
                # Any remaining NaN after ffill will be at the START of the time series
                # These will be trimmed when we filter to formation window dates
    
    # Count NaN after pass 2
    nan_after = df[feature_cols].isna().sum().sum()
    
    if verbose:
        if nan_after_pass1 > nan_after:
            n_imputed_pass2 = nan_after_pass1 - nan_after
            print(f"  [impute] Pass 2: Imputed {n_imputed_pass2:,} NaN with forward-fill only (no bfill - safe)", flush=True)
        if nan_after > 0:
            # Show which columns still have NaN
            nan_cols = df[feature_cols].isna().sum()
            nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
            print(f"  [impute] INFO: {nan_after:,} NaN remain in {len(nan_cols)} columns (warmup period)", flush=True)
            print(f"  [impute] These will be removed when trimming to formation window", flush=True)
            for col, cnt in nan_cols.head(5).items():
                print(f"           - {col}: {cnt:,} NaN", flush=True)
            if len(nan_cols) > 5:
                print(f"           ... and {len(nan_cols) - 5} more", flush=True)
    
    # Restore index
    if has_multiindex:
        df = df.set_index(['Date', 'Ticker'])
    
    return df


def final_nan_check(panel_df: pd.DataFrame, stage: str = "final") -> bool:
    """
    Final NaN check - returns True if data is clean, False if NaN exist.
    
    This is called at the END of feature engineering to ensure clean output.
    If NaN exist, feature selection will FAIL with a clear error message.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Final panel data
    stage : str
        Stage name for logging
        
    Returns
    -------
    bool
        True if no NaN exist, False otherwise
    """
    if panel_df.empty:
        return True
    
    # Exclude non-feature columns from check
    exclude_cols = {'Ticker', 'Date', 'Close', 'ADV_63', 'ADV_63_Rank'}
    exclude_cols.update([c for c in panel_df.columns if c.startswith('FwdRet')])
    
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols]
    
    nan_count = panel_df[feature_cols].isna().sum().sum()
    total_cells = len(panel_df) * len(feature_cols)
    
    if nan_count > 0:
        nan_pct = nan_count / total_cells * 100
        print(f"\n[{stage}] ERROR: {nan_count:,} NaN values remain ({nan_pct:.2f}% of feature data)", flush=True)
        
        # Show which columns have NaN
        nan_by_col = panel_df[feature_cols].isna().sum()
        cols_with_nan = nan_by_col[nan_by_col > 0].sort_values(ascending=False)
        print(f"[{stage}] Columns with NaN:", flush=True)
        for col, count in cols_with_nan.head(10).items():
            print(f"    - {col}: {count:,} NaN", flush=True)
        
        return False
    
    print(f"[{stage}] ✓ Data is clean - no NaN in {len(feature_cols)} feature columns", flush=True)
    return True

# ============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ============================================================================

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _hurst_window(ts):
        """Compute Hurst exponent for a single window."""
        n = ts.size
        mean = 0.0
        count = 0
        for i in range(n):
            v = ts[i]
            if not np.isnan(v):
                count += 1
                mean += v
        if count == 0:
            return np.nan
        mean /= count
        
        Ymin = 1e18
        Ymax = -1e18
        accum = 0.0
        S2 = 0.0
        
        for i in range(n):
            v = ts[i]
            if np.isnan(v):
                return np.nan
            dv = v - mean
            S2 += dv * dv
            accum += dv
            if accum < Ymin:
                Ymin = accum
            if accum > Ymax:
                Ymax = accum
        
        S = np.sqrt(S2 / n)
        if S == 0.0:
            return np.nan
        R = Ymax - Ymin
        return np.log(R / S) / np.log(n)
    
    @njit(cache=True, fastmath=True)
    def _hurst_series(x, window):
        """Rolling Hurst exponent calculation."""
        n = x.size
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = np.nan
        if n < window:
            return out
        for t in range(window - 1, n):
            out[t] = _hurst_window(x[t - window + 1:t + 1])
        return out
else:
    def _hurst_series(x, window):
        """NumPy fallback for Hurst calculation."""
        n = x.size
        out = np.full(n, np.nan, dtype=np.float32)
        if n < window:
            return out
        for t in range(window - 1, n):
            seg = x[t - window + 1:t + 1]
            if np.any(np.isnan(seg)):
                continue
            mu = np.mean(seg)
            Y = np.cumsum(seg - mu)
            R = np.max(Y) - np.min(Y)
            S = np.std(seg)
            if S == 0 or R <= 0:
                continue
            out[t] = np.log(R / S) / np.log(window)
        return out

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS (VECTORIZED)
# ============================================================================

def pct_change_k(col, lags):
    """Multi-lag percentage changes."""
    return {f'{col.name}%-{k}': (col / col.shift(k) - 1.0) * 100.0 for k in lags}

def lagged_returns(col_pct, lags):
    """Lagged return features."""
    return {f'{col_pct.name}_lag{k}': col_pct.shift(k) for k in lags}

def ma_dict(col, windows):
    """Simple moving averages."""
    return {f'{col.name}_MA{w}': col.rolling(w).mean() for w in windows}

def ema_dict(col, spans):
    """Exponential moving averages."""
    return {f'{col.name}_EMA{s}': col.ewm(span=s, adjust=False).mean() for s in spans}

def std_dict(col_pct, windows):
    """Rolling standard deviations."""
    return {f'{col_pct.name}_std{w}': col_pct.rolling(w).std() for w in windows}

def skew_dict(col_pct, windows):
    """Rolling skewness."""
    return {f'{col_pct.name}_skew{w}': col_pct.rolling(w).skew() for w in windows}

def kurt_dict(col_pct, windows):
    """Rolling kurtosis."""
    return {f'{col_pct.name}_kurt{w}': col_pct.rolling(w).kurt() for w in windows}

def bollinger_dict(col_pct, windows):
    """Bollinger Bands."""
    out = {}
    for w in windows:
        r = col_pct.rolling(w)
        mu = r.mean()
        sd = r.std()
        out[f'{col_pct.name}_BollUp{w}'] = mu + 2 * sd
        out[f'{col_pct.name}_BollLo{w}'] = mu - 2 * sd
    return out

def momentum_dict(col_pct, windows):
    """Momentum features (change from t-w to t)."""
    return {f'{col_pct.name}_Mom{w}': (col_pct - col_pct.shift(w)) for w in windows}

def rsi_multi(close, windows):
    """Multi-window RSI."""
    out = {}
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    
    for w in windows:
        avg_gain = gains.rolling(w).mean()
        avg_loss = losses.rolling(w).mean()
        rs = avg_gain / avg_loss
        out[f'{close.name}_RSI{w}'] = np.float32(100.0 - (100.0 / (1.0 + rs)))
    return out

def macd_features(close):
    """MACD family of features."""
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    return {
        f'{close.name}_MACD': macd,
        f'{close.name}_MACD_Signl': macd_signal,
        f'{close.name}_MACD_Histo': macd_hist,
        f'{close.name}_MACD_HistSl': macd_hist.diff(),
        f'{close.name}_MACD_Xover': (macd > macd_signal).astype('float32'),
        f'{close.name}_MACD_SignDir': macd_signal.diff(),
        f'{close.name}_MACD_Mom': macd.diff(),
    }

def atr(high, low, close, window=14):
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return {f'{close.name}_ATR{window}': tr.rolling(window).mean()}

def williams_r_multi(high, low, close, windows):
    """Multi-window Williams %R."""
    out = {}
    for w in windows:
        high_n = high.rolling(w).max()
        low_n = low.rolling(w).min()
        wr = ((high_n - close) / (high_n - low_n)) * -100.0
        out[f'{close.name}_WilliamsR{w}'] = wr
    return out

def hurst_multi(series, windows):
    """Multi-window Hurst exponent (long-memory indicator)."""
    out = {}
    x = series.to_numpy(dtype=np.float64)
    for w in windows:
        out[f'{series.name}_Hurst{w}'] = pd.Series(
            _hurst_series(x, w),
            index=series.index
        )
    return out

def adv_features(close, volume, window=63):
    """
    Average Daily Dollar Volume (ADV) - replaces VPT for liquidity.
    
    ADV is a proper liquidity measure: higher values = more liquid.
    """
    if volume is None:
        return {}
    if bool(volume.isna().all()):
        return {}
    
    # Dollar volume = Close * Volume
    dollar_volume = close * volume
    
    # ADV: rolling mean of dollar volume
    adv = dollar_volume.rolling(window).mean()
    
    return {
        f'ADV_{window}': adv,
    }

def drawdown_features(close, windows):
    """
    Max drawdown features from crosssecmom.
    
    Computes maximum drawdown over rolling windows.
    dd = (price - rolling_max) / rolling_max
    """
    out = {}
    for w in windows:
        rolling_max = close.rolling(window=w, min_periods=max(1, w//2)).max()
        drawdown = (close - rolling_max) / rolling_max
        max_dd = drawdown.rolling(window=w, min_periods=max(1, w//2)).min()
        out[f'{close.name}_DD{w}'] = max_dd
    return out

def shock_features(returns, vol_60d):
    """
    Shock features from crosssecmom: standardized daily returns.
    
    ret_1d_z = ret_1d / vol_60d (return normalized by 60-day volatility)
    """
    ret_1d_z = returns / (vol_60d + 1e-8)  # Avoid division by zero
    return {
        f'{returns.name}_Ret1dZ': ret_1d_z
    }

# ============================================================================
# PER-TICKER FEATURE ENGINEERING
# ============================================================================

def process_ticker(
    ticker: str, 
    data: pd.DataFrame, 
    adv_window: int = 63,
    config: 'ResearchConfig' = None
) -> pd.DataFrame:
    """
    Engineer features for a single ticker.
    
    NOTE: Raw OHLCV data should already be cleaned by clean_raw_ohlcv_data()
    in data_manager.py BEFORE calling this function. This function only:
    - Computes features from the cleaned data
    - Replaces Inf with NaN (will be handled in panel-level cleaning)
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    data : pd.DataFrame
        Cleaned OHLCV data with DatetimeIndex (already forward-filled)
    adv_window : int
        Window for ADV calculation (default: 63)
    config : ResearchConfig, optional
        Configuration with NaN handling parameters (not used here anymore)
        
    Returns
    -------
    pd.DataFrame
        Features with DatetimeIndex and 'Ticker' column
    """
    try:
        # Data should already be cleaned - just check if empty
        if data.empty:
            return pd.DataFrame()
        
        close = data['Close'].astype('float32')
        close.name = 'Close'
        
        # CRITICAL: Include Close in output
        feats = {
            'Ticker': ticker,
            'Close': close,
        }
        
        # Returns-based series
        close_pct = (close.pct_change() * 100.0)
        close_pct.name = 'Close'
        
        # NEW from crosssecmom: Clip returns to ±5σ for feature calculation
        # This prevents extreme outliers from distorting features
        std_global = close_pct.std()
        threshold = 5.0 * std_global
        close_pct_clipped = close_pct.clip(lower=-threshold, upper=threshold)
        close_pct_clipped.name = 'Close'
        
        # Core return features (using clipped returns)
        feats.update(pct_change_k(close, [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]))
        feats.update(lagged_returns(close_pct_clipped, [1, 2, 3, 5, 10]))
        feats.update(std_dict(close_pct_clipped, [5, 10, 21, 42, 63, 126]))
        feats.update(skew_dict(close_pct_clipped, [21, 42, 63, 126]))
        feats.update(kurt_dict(close_pct_clipped, [21, 42, 63, 126]))
        feats.update(bollinger_dict(close_pct_clipped, [21, 50]))
        feats.update(momentum_dict(close_pct_clipped, [5, 10, 21, 42, 63]))
        
        # NEW from crosssecmom: Max drawdown features
        feats.update(drawdown_features(close, [20, 60]))
        
        # Level-based features
        feats.update(ma_dict(close, [5, 10, 21, 42, 63, 126, 200]))
        feats.update(ema_dict(close, [5, 10, 21, 42, 63, 126]))
        feats.update(rsi_multi(close, [14, 21, 42]))
        feats.update(macd_features(close))
        
        # OHLCV-dependent features
        if 'High' in data.columns and 'Low' in data.columns:
            high = data['High'].astype('float32')
            low = data['Low'].astype('float32')
            high.name = 'Close'
            low.name = 'Close'
            feats.update(atr(high, low, close, window=14))
            feats.update(williams_r_multi(high, low, close, [14, 21, 63]))
        
        # Liquidity: ADV (replaces VPT)
        if 'Volume' in data.columns:
            vol = data['Volume'].astype('float32')
            feats.update(adv_features(close, vol, window=adv_window))
        
        # Hurst exponent on returns (using clipped returns)
        feats.update(hurst_multi(close_pct_clipped, [21, 63, 126]))
        
        # NEW from crosssecmom: Shock features (standardized returns, using clipped)
        # Need vol_60d for ret_1d_z calculation
        vol_60d = close_pct_clipped.rolling(window=60, min_periods=30).std()
        feats.update(shock_features(close_pct_clipped, vol_60d))
        
        # Convert to DataFrame
        df = pd.DataFrame(feats, index=data.index)
        
        # Convert numeric columns to float32, keep Ticker as string
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype('float32')
        
        # =====================================================================
        # STEP 2: Handle Inf values from calculations (e.g., division by zero)
        # =====================================================================
        # Replace Inf/-Inf with NaN (will be handled in later cleaning steps)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
        
    except Exception as e:
        import traceback
        print(f"[error] {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame(index=data.index)

# ============================================================================
# FORWARD RETURNS (TARGET VARIABLES)
# ============================================================================

def add_forward_returns(panel_df: pd.DataFrame, horizon: int, config: 'ResearchConfig') -> pd.DataFrame:
    """
    Add forward returns at specified horizon.
    
    Uses closed-left windows: forward return from t to t+h uses Close[t] and Close[t+h].
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) index or columns
    horizon : int
        Forward horizon in days
    config : ResearchConfig
        Configuration object with feature settings
        
    Returns
    -------
    pd.DataFrame
        Panel with added FwdRet_{h} column
    """
    print(f"[fwd] Computing forward returns for horizon: {horizon}")
    
    # Compute forward returns per ticker (as decimals, not percent)
    # CRITICAL: +5% return is stored as 0.05, not 5.0
    fwd_ret = (
        panel_df.groupby('Ticker')['Close']
        .pct_change(horizon)
        .shift(-horizon)
    )
    
    # PHASE 0 CONTROL: Optional winsorization of outliers at ±n_sigma
    # Controlled by config.features.enable_winsorization
    if config.features.enable_winsorization:
        print(f"[fwd] Winsorizing forward returns at ±{config.features.winsorization_n_sigma}σ (per period)")
        
        # Group by date to winsorize within each cross-section
        def winsorize_cross_section(group, n_sigma=config.features.winsorization_n_sigma):
            """Winsorize returns within a single time period."""
            valid = group.dropna()
            if len(valid) == 0:
                return group
            
            mean = valid.mean()
            std = valid.std()
            
            if std > 0:
                lower_bound = mean - n_sigma * std
                upper_bound = mean + n_sigma * std
                return group.clip(lower=lower_bound, upper=upper_bound)
            else:
                return group
        
        # Group by Date column and winsorize, preserving original index
        fwd_ret_winsorized = fwd_ret.groupby(panel_df['Date'], group_keys=False).apply(winsorize_cross_section)
        
        # Log winsorization impact
        n_clipped = ((fwd_ret != fwd_ret_winsorized) & fwd_ret.notna()).sum()
        pct_clipped = 100 * n_clipped / fwd_ret.notna().sum()
        print(f"[fwd] Winsorized {n_clipped:,} observations ({pct_clipped:.2f}% of valid data)")
        
        panel_df[f'FwdRet_{horizon}'] = fwd_ret_winsorized.astype('float32')
    else:
        print(f"[fwd] Winsorization disabled - using raw forward returns")
        panel_df[f'FwdRet_{horizon}'] = fwd_ret.astype('float32')
    
    return panel_df

def add_macro_features(
    panel_df: pd.DataFrame,
    macro_data: Dict[str, pd.Series],
    config: ResearchConfig
) -> pd.DataFrame:
    """
    Add macro and regime features from crosssecmom.
    
    Computes 9 macro features:
    - vix_level: VIX level
    - vix_z_1y: VIX z-score over 1 year
    - yc_slope: Yield curve slope (10Y - 2Y)
    - short_rate: Short-term rate (3M T-bill)
    - credit_proxy_20: Credit spread proxy (HYG - LQD 20-day returns)
    - crash_flag: Market crash indicator (VT return < -2.5σ)
    - meltup_flag: Market melt-up indicator (VT return > +2.5σ)
    - high_vol: High volatility regime (VIX z-score > 1)
    - low_vol: Low volatility regime (VIX z-score < -1)
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with Date column
    macro_data : Dict[str, pd.Series]
        Dictionary of macro series from data_manager
    config : ResearchConfig
        Config object
        
    Returns
    -------
    pd.DataFrame
        Panel with added macro features
    """
    print(f"[macro] Adding macro/regime features...")
    
    # Get unique dates from panel
    dates = panel_df['Date'].unique()
    date_index = pd.DatetimeIndex(dates).sort_values()
    
    # Initialize macro features DataFrame
    macro_features = pd.DataFrame(index=date_index)
    
    # 1. VIX features
    if 'vix' in macro_data:
        vix = macro_data['vix'].reindex(date_index, method='ffill')
        macro_features['vix_level'] = vix
        
        # VIX z-score (1-year rolling)
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        macro_features['vix_z_1y'] = (vix - vix_mean) / (vix_std + 1e-8)
    else:
        print("  [warning] VIX data not available, setting to 0")
        macro_features['vix_level'] = 0.0
        macro_features['vix_z_1y'] = 0.0
    
    # 2. Yield curve slope (10Y - 2Y)
    if 'yield_10y' in macro_data and 'yield_2y' in macro_data:
        yield_10y = macro_data['yield_10y'].reindex(date_index, method='ffill')
        yield_2y = macro_data['yield_2y'].reindex(date_index, method='ffill')
        macro_features['yc_slope'] = yield_10y - yield_2y
    else:
        print("  [warning] Yield data not available, setting yc_slope to 0")
        macro_features['yc_slope'] = 0.0
    
    # 3. Short rate (3M T-bill)
    if 'tbill_3m' in macro_data:
        short_rate = macro_data['tbill_3m'].reindex(date_index, method='ffill')
        macro_features['short_rate'] = short_rate
    else:
        print("  [warning] T-bill data not available, setting short_rate to 0")
        macro_features['short_rate'] = 0.0
    
    # 4. Credit spread proxy (HYG - LQD 20-day returns)
    # Need to get HYG and LQD returns from panel if available
    if 'HYG' in panel_df['Ticker'].values and 'LQD' in panel_df['Ticker'].values:
        # Get 20-day returns for HYG and LQD
        hyg_data = panel_df[panel_df['Ticker'] == 'HYG'].set_index('Date')
        lqd_data = panel_df[panel_df['Ticker'] == 'LQD'].set_index('Date')
        
        if 'Close%-21' in hyg_data.columns and 'Close%-21' in lqd_data.columns:
            hyg_ret = hyg_data['Close%-21'].reindex(date_index, method='ffill')
            lqd_ret = lqd_data['Close%-21'].reindex(date_index, method='ffill')
            macro_features['credit_proxy_20'] = hyg_ret - lqd_ret
        else:
            macro_features['credit_proxy_20'] = 0.0
    else:
        print("  [warning] HYG/LQD not in universe, setting credit_proxy_20 to 0")
        macro_features['credit_proxy_20'] = 0.0
    
    # 5. Regime flags based on VT if available
    if 'VT' in panel_df['Ticker'].values:
        vt_data = panel_df[panel_df['Ticker'] == 'VT'].set_index('Date')
        
        # Calculate VT 1-day returns if not already present
        if 'Close' in vt_data.columns:
            vt_close = vt_data['Close'].reindex(date_index, method='ffill')
            vt_ret = vt_close.pct_change() * 100.0
            
            # 60-day rolling volatility
            vt_vol_60 = vt_ret.rolling(window=60, min_periods=30).std()
            
            # Crash flag: VT 1-day return < -2.5 * 60d vol
            crash_threshold = -2.5 * vt_vol_60
            macro_features['crash_flag'] = (vt_ret < crash_threshold).astype(float)
            
            # Melt-up flag: VT 1-day return > +2.5 * 60d vol
            meltup_threshold = 2.5 * vt_vol_60
            macro_features['meltup_flag'] = (vt_ret > meltup_threshold).astype(float)
        else:
            macro_features['crash_flag'] = 0.0
            macro_features['meltup_flag'] = 0.0
    else:
        print("  [warning] VT not in universe, setting regime flags to 0")
        macro_features['crash_flag'] = 0.0
        macro_features['meltup_flag'] = 0.0
    
    # 6. VIX regime flags
    if 'vix_z_1y' in macro_features.columns:
        vix_z = macro_features['vix_z_1y']
        macro_features['high_vol'] = (vix_z > 1.0).astype(float)
        macro_features['low_vol'] = (vix_z < -1.0).astype(float)
    else:
        macro_features['high_vol'] = 0.0
        macro_features['low_vol'] = 0.0
    
    # Convert to float32 for efficiency
    for col in macro_features.columns:
        macro_features[col] = macro_features[col].astype('float32')
    
    # Merge with panel (broadcast macro features to all tickers per date)
    macro_features_reset = macro_features.reset_index()
    macro_features_reset.columns = ['Date'] + list(macro_features.columns)
    
    panel_df = panel_df.merge(macro_features_reset, on='Date', how='left')
    
    print(f"  [macro] Added {len(macro_features.columns)} macro features")
    
    return panel_df

# ============================================================================
# CROSS-SECTIONAL ADV RANK (for filtering)
# ============================================================================

def add_adv_rank(panel_df: pd.DataFrame, adv_col: str = 'ADV_63') -> pd.DataFrame:
    """
    Add cross-sectional rank of ADV for liquidity filtering.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    adv_col : str
        ADV column name
        
    Returns
    -------
    pd.DataFrame
        Panel with added {adv_col}_Rank column
    """
    if adv_col not in panel_df.columns:
        return panel_df
    
    print(f"[cs] Adding cross-sectional rank for {adv_col}...")
    
    panel_df[f'{adv_col}_Rank'] = (
        panel_df.groupby('Date')[adv_col]
        .rank(pct=True, method='average')
        .astype('float32')
    )
    
    return panel_df

def add_relative_return_features(panel_df: pd.DataFrame, lookbacks=[5, 20, 60]) -> pd.DataFrame:
    """
    Add relative return features from crosssecmom:
    - Relative to VT (global market benchmark)
    - Relative to equal-weight basket
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with Date, Ticker, Close columns
    lookbacks : list
        Lookback windows for relative returns
        
    Returns
    -------
    pd.DataFrame
        Panel with added relative return features
    """
    print(f"[cs] Adding relative return features...")
    
    # Compute L-day returns per ticker
    for L in lookbacks:
        panel_df[f'ret_{L}d'] = (
            panel_df.groupby('Ticker')['Close']
            .pct_change(L)
            .shift(0) * 100.0
        ).astype('float32')
    
    # Equal-weight basket returns per date
    for L in lookbacks:
        basket_ret = panel_df.groupby('Date')[f'ret_{L}d'].mean()
        panel_df[f'Rel{L}_vs_Basket'] = (
            panel_df[f'ret_{L}d'] - panel_df['Date'].map(basket_ret)
        ).astype('float32')
    
    # VT returns (if VT exists in universe)
    if 'VT' in panel_df['Ticker'].unique():
        vt_df = panel_df[panel_df['Ticker'] == 'VT'].set_index('Date')
        for L in lookbacks:
            vt_ret = vt_df[f'ret_{L}d']
            panel_df[f'Rel{L}_vs_VT'] = (
                panel_df[f'ret_{L}d'] - panel_df['Date'].map(vt_ret)
            ).astype('float32')
    else:
        print("  [warning] VT not in universe, skipping VT relative returns")
        for L in lookbacks:
            panel_df[f'Rel{L}_vs_VT'] = 0.0
    
    # Clean up temporary columns
    for L in lookbacks:
        panel_df.drop(columns=[f'ret_{L}d'], inplace=True)
    
    return panel_df

def add_correlation_features(panel_df: pd.DataFrame, window=20) -> pd.DataFrame:
    """
    Add rolling correlation features from crosssecmom:
    - Correlation with VT (global market)
    - Correlation with BNDW (bonds)
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with Date, Ticker columns
    window : int
        Rolling window for correlation
        
    Returns
    -------
    pd.DataFrame
        Panel with added correlation features
    """
    print(f"[cs] Adding correlation features (window={window})...")
    
    # Need returns for correlation calculation
    panel_df['ret_1d'] = (
        panel_df.groupby('Ticker')['Close']
        .pct_change(1) * 100.0
    ).astype('float32')
    
    # Pivot to wide format for correlation calculation
    returns_wide = panel_df.pivot(index='Date', columns='Ticker', values='ret_1d')
    
    # VT correlations
    if 'VT' in returns_wide.columns:
        vt_ret = returns_wide['VT']
        corr_vt = returns_wide.rolling(window=window, min_periods=max(1, window//2)).corr(vt_ret)
        
        # Melt back to long format
        corr_vt_long = corr_vt.stack().reset_index()
        corr_vt_long.columns = ['Date', 'Ticker', f'Corr{window}_VT']
        panel_df = panel_df.merge(corr_vt_long, on=['Date', 'Ticker'], how='left')
    else:
        print("  [warning] VT not in universe, skipping VT correlations")
        panel_df[f'Corr{window}_VT'] = 0.0
    
    # BNDW correlations
    if 'BNDW' in returns_wide.columns:
        bndw_ret = returns_wide['BNDW']
        corr_bndw = returns_wide.rolling(window=window, min_periods=max(1, window//2)).corr(bndw_ret)
        
        # Melt back to long format
        corr_bndw_long = corr_bndw.stack().reset_index()
        corr_bndw_long.columns = ['Date', 'Ticker', f'Corr{window}_BNDW']
        panel_df = panel_df.merge(corr_bndw_long, on=['Date', 'Ticker'], how='left')
    else:
        print("  [warning] BNDW not in universe, skipping BNDW correlations")
        panel_df[f'Corr{window}_BNDW'] = 0.0
    
    # Clean up temporary column
    panel_df.drop(columns=['ret_1d'], inplace=True)
    
    return panel_df

def add_asset_type_flags(panel_df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """
    Add asset type binary flags from crosssecmom.
    
    Uses universe metadata family classification to determine:
    - is_equity: 1 if EQ_* family
    - is_bond: 1 if BOND_* family
    - is_real_asset: 1 if REAL_* family (commodities, REITs, gold, etc.)
    - is_sector: 1 if EQ_SECTOR_* family
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with Ticker column
    config : ResearchConfig
        Config object
        
    Returns
    -------
    pd.DataFrame
        Panel with added asset type flags
    """
    print(f"[cs] Adding asset type flags...")
    
    # Load universe metadata if available
    metadata_path = config.paths.universe_metadata_output
    if not Path(metadata_path).exists():
        print(f"  [warning] Universe metadata not found at {metadata_path}, creating basic flags")
        # Simple heuristics as fallback
        panel_df['is_equity'] = panel_df['Ticker'].apply(
            lambda t: 1 if t in ['VT', 'VTI', 'SPY', 'QQQ'] else 0
        ).astype('float32')
        panel_df['is_bond'] = panel_df['Ticker'].apply(
            lambda t: 1 if t in ['BND', 'BNDW', 'AGG', 'LQD', 'HYG'] else 0
        ).astype('float32')
        panel_df['is_real_asset'] = panel_df['Ticker'].apply(
            lambda t: 1 if t in ['VNQ', 'GLD', 'DBC', 'GSG'] else 0
        ).astype('float32')
        panel_df['is_sector'] = 0.0
        return panel_df
    
    # Load metadata with family classification
    metadata = pd.read_csv(metadata_path)
    
    # Create flags based on family prefix
    def get_asset_flags(row):
        family = row.get('family', 'UNKNOWN')
        return pd.Series({
            'is_equity': 1.0 if family.startswith('EQ_') else 0.0,
            'is_bond': 1.0 if family.startswith('BOND_') else 0.0,
            'is_real_asset': 1.0 if family.startswith('REAL_') or family in ['ALT_GOLD', 'ALT_COMMODITY'] else 0.0,
            'is_sector': 1.0 if family.startswith('EQ_SECTOR_') else 0.0,
        })
    
    flags = metadata.apply(get_asset_flags, axis=1)
    metadata = pd.concat([metadata[['ticker']], flags], axis=1)
    metadata.columns = ['Ticker', 'is_equity', 'is_bond', 'is_real_asset', 'is_sector']
    
    # Merge with panel
    panel_df = panel_df.merge(metadata, on='Ticker', how='left')
    
    # Fill NaN with 0
    for col in ['is_equity', 'is_bond', 'is_real_asset', 'is_sector']:
        panel_df[col] = panel_df[col].fillna(0.0).astype('float32')
    
    return panel_df

# ============================================================================
# PHASE 1: EXHAUSTIVE INTERACTION FEATURES
# ============================================================================

def generate_interaction_features(panel_df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """
    PHASE 1: Generate ALL mathematically valid interaction features.
    
    Philosophy:
    - EXHAUSTIVE, not heuristic (do NOT handpick based on "gut feel")
    - Generate ALL logical combinations, let feature selection decide
    - "engineer as many logical combinations as possible and let feature selection 
       decide if they fit - do not try to frontrun the statistical test"
    
    Interaction Types:
    1. Multiplicative: Mom×Mom, Mom×Vol, Mom×Macro, Mom×Oscillator, Vol×Macro
    2. Ratio: Mom/Vol, Mom/ADV, Vol/Vol (risk-adjusted returns)
    3. Difference: Mom acceleration, Vol regime shifts
    4. Polynomial: Mom^2, Mom^3, Vol^2 (non-linear effects)
    5. Regime-Conditional: feature × regime_flag
    
    Target: 300-500 total features (96 base + 200-400 interactions)
    
    Args:
        panel_df: Panel data with base features
        config: Research configuration
        
    Returns:
        Panel data with base + interaction features
    """
    print("[interaction] Starting exhaustive interaction feature generation...")
    print(f"[interaction] Base feature count: {len(panel_df.columns)}")
    
    # Identify base feature categories (exclude Close, FwdRet_21, Date, Ticker)
    exclude_cols = {'Close', 'FwdRet_21', 'Date', 'Ticker', 'ADV_63_Rank'}
    all_cols = [c for c in panel_df.columns if c not in exclude_cols]
    
    # Categorize features by mathematical properties
    momentum_features = [c for c in all_cols if any(x in c for x in ['Close%-', 'Mom', 'lag'])]
    volatility_features = [c for c in all_cols if any(x in c for x in ['std', 'ATR'])]
    oscillator_features = [c for c in all_cols if any(x in c for x in ['RSI', 'Williams'])]
    trend_features = [c for c in all_cols if any(x in c for x in ['MA', 'EMA', 'Boll'])]
    macd_features = [c for c in all_cols if 'MACD' in c]
    higher_moments = [c for c in all_cols if any(x in c for x in ['skew', 'kurt'])]
    hurst_features = [c for c in all_cols if 'Hurst' in c]
    drawdown_features = [c for c in all_cols if 'DD' in c]
    shock_features = [c for c in all_cols if 'Ret1dZ' in c]
    liquidity_features = [c for c in all_cols if c in ['ADV_63']]  # Always positive
    macro_features = [c for c in all_cols if any(x in c for x in ['vix_', 'yc_', 'short_rate', 'credit_proxy'])]
    regime_flags = [c for c in all_cols if any(x in c for x in ['crash_flag', 'meltup_flag', 'high_vol', 'low_vol'])]
    relative_features = [c for c in all_cols if 'Rel' in c and 'vs' in c]
    correlation_features = [c for c in all_cols if 'Corr' in c]
    
    print(f"[interaction] Momentum features: {len(momentum_features)}")
    print(f"[interaction] Volatility features: {len(volatility_features)}")
    print(f"[interaction] Oscillator features: {len(oscillator_features)}")
    print(f"[interaction] Macro features: {len(macro_features)}")
    print(f"[interaction] Regime flags: {len(regime_flags)}")
    
    interaction_count = 0
    
    # =========================================================================
    # Type 1: MULTIPLICATIVE INTERACTIONS (Products)
    # =========================================================================
    print("\n[interaction] Type 1: Multiplicative (products)...")
    
    # 1.1 Momentum × Momentum (all pairs)
    print("  [1.1] Mom × Mom (all pairs)...")
    for i, feat1 in enumerate(momentum_features):
        for feat2 in momentum_features[i+1:]:  # Avoid duplicates (A×B = B×A)
            new_col = f"{feat1}_x_{feat2}"
            panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count} Mom×Mom interactions")
    
    # 1.2 Momentum × Volatility (all pairs)
    print("  [1.2] Mom × Vol (all pairs)...")
    start_count = interaction_count
    for feat1 in momentum_features:
        for feat2 in volatility_features:
            new_col = f"{feat1}_x_{feat2}"
            panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Mom×Vol interactions")
    
    # 1.3 Momentum × Macro (all pairs)
    print("  [1.3] Mom × Macro (all pairs)...")
    start_count = interaction_count
    for feat1 in momentum_features:
        for feat2 in macro_features:
            new_col = f"{feat1}_x_{feat2}"
            panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Mom×Macro interactions")
    
    # 1.4 Momentum × Oscillator (all pairs)
    print("  [1.4] Mom × Oscillator (all pairs)...")
    start_count = interaction_count
    for feat1 in momentum_features:
        for feat2 in oscillator_features:
            new_col = f"{feat1}_x_{feat2}"
            panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Mom×Oscillator interactions")
    
    # 1.5 Volatility × Macro (all pairs)
    print("  [1.5] Vol × Macro (all pairs)...")
    start_count = interaction_count
    for feat1 in volatility_features:
        for feat2 in macro_features:
            new_col = f"{feat1}_x_{feat2}"
            panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Vol×Macro interactions")
    
    # 1.6 Momentum × Relative (all pairs)
    print("  [1.6] Mom × Relative (all pairs)...")
    start_count = interaction_count
    for feat1 in momentum_features:
        for feat2 in relative_features:
            new_col = f"{feat1}_x_{feat2}"
            panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Mom×Relative interactions")
    
    # =========================================================================
    # Type 2: RATIO INTERACTIONS (Division)
    # =========================================================================
    print("\n[interaction] Type 2: Ratio (division, denominator > 0)...")
    
    # 2.1 Momentum / Volatility (Sharpe-like ratios)
    print("  [2.1] Mom / Vol (risk-adjusted returns)...")
    start_count = interaction_count
    for feat1 in momentum_features:
        for feat2 in volatility_features:
            new_col = f"{feat1}_div_{feat2}"
            # Avoid division by zero: replace vol=0 with NaN
            denominator = panel_df[feat2].replace(0, np.nan)
            panel_df[new_col] = (panel_df[feat1] / denominator).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Mom/Vol ratios")
    
    # 2.2 Momentum / ADV (liquidity-adjusted returns)
    if liquidity_features:
        print("  [2.2] Mom / ADV (liquidity-adjusted returns)...")
        start_count = interaction_count
        for feat1 in momentum_features:
            for feat2 in liquidity_features:
                new_col = f"{feat1}_div_{feat2}"
                denominator = panel_df[feat2].replace(0, np.nan)
                panel_df[new_col] = (panel_df[feat1] / denominator).astype('float32')
                interaction_count += 1
        print(f"      Generated {interaction_count - start_count} Mom/ADV ratios")
    
    # 2.3 Volatility / Volatility (different horizons, vol regime shifts)
    print("  [2.3] Vol / Vol (regime shifts)...")
    start_count = interaction_count
    for i, feat1 in enumerate(volatility_features):
        for feat2 in volatility_features[i+1:]:
            new_col = f"{feat1}_div_{feat2}"
            denominator = panel_df[feat2].replace(0, np.nan)
            panel_df[new_col] = (panel_df[feat1] / denominator).astype('float32')
            interaction_count += 1
    print(f"      Generated {interaction_count - start_count} Vol/Vol ratios")
    
    # =========================================================================
    # Type 3: DIFFERENCE INTERACTIONS (Subtraction)
    # =========================================================================
    print("\n[interaction] Type 3: Difference (acceleration, spreads)...")
    
    # 3.1 Momentum Acceleration (adjacent horizons)
    print("  [3.1] Momentum acceleration (Mom_short - Mom_long)...")
    start_count = interaction_count
    # Define momentum hierarchy by lookback (short to long)
    mom_hierarchy = [
        'Close%-1', 'Close%-2', 'Close%-3', 'Close%-5', 'Close%-10',
        'Close%-21', 'Close%-42', 'Close%-63', 'Close%-126', 'Close%-252'
    ]
    mom_available = [m for m in mom_hierarchy if m in momentum_features]
    for i in range(len(mom_available) - 1):
        feat1 = mom_available[i]      # Shorter horizon
        feat2 = mom_available[i + 1]  # Longer horizon
        new_col = f"{feat1}_minus_{feat2}"
        panel_df[new_col] = (panel_df[feat1] - panel_df[feat2]).astype('float32')
        interaction_count += 1
    print(f"      Generated {interaction_count - start_count} momentum acceleration features")
    
    # 3.2 Volatility Changes (adjacent horizons)
    print("  [3.2] Volatility changes (Vol_short - Vol_long)...")
    start_count = interaction_count
    vol_hierarchy = ['Close_std5', 'Close_std10', 'Close_std21', 'Close_std42', 'Close_std63', 'Close_std126']
    vol_available = [v for v in vol_hierarchy if v in volatility_features]
    for i in range(len(vol_available) - 1):
        feat1 = vol_available[i]      # Shorter horizon
        feat2 = vol_available[i + 1]  # Longer horizon
        new_col = f"{feat1}_minus_{feat2}"
        panel_df[new_col] = (panel_df[feat1] - panel_df[feat2]).astype('float32')
        interaction_count += 1
    print(f"      Generated {interaction_count - start_count} volatility change features")
    
    # =========================================================================
    # Type 4: POLYNOMIAL TRANSFORMATIONS
    # =========================================================================
    print("\n[interaction] Type 4: Polynomial (squared, cubed)...")
    
    # 4.1 Momentum Squared (emphasize extremes, convex)
    print("  [4.1] Momentum squared (convex)...")
    start_count = interaction_count
    for feat in momentum_features:
        new_col = f"{feat}_sq"
        panel_df[new_col] = (panel_df[feat] ** 2).astype('float32')
        interaction_count += 1
    print(f"      Generated {interaction_count - start_count} momentum squared features")
    
    # 4.2 Momentum Cubed (preserve sign, emphasize tails)
    print("  [4.2] Momentum cubed (tail emphasis)...")
    start_count = interaction_count
    for feat in momentum_features:
        new_col = f"{feat}_cb"
        panel_df[new_col] = (panel_df[feat] ** 3).astype('float32')
        interaction_count += 1
    print(f"      Generated {interaction_count - start_count} momentum cubed features")
    
    # 4.3 Volatility Squared (variance, vol-of-vol proxy)
    print("  [4.3] Volatility squared (variance)...")
    start_count = interaction_count
    for feat in volatility_features:
        new_col = f"{feat}_sq"
        panel_df[new_col] = (panel_df[feat] ** 2).astype('float32')
        interaction_count += 1
    print(f"      Generated {interaction_count - start_count} volatility squared features")
    
    # =========================================================================
    # Type 5: REGIME-CONDITIONAL FEATURES
    # =========================================================================
    if regime_flags:
        print("\n[interaction] Type 5: Regime-conditional (feature × regime_flag)...")
        
        # 5.1 Momentum × Regime Flags
        print("  [5.1] Momentum × Regime Flags...")
        start_count = interaction_count
        for feat1 in momentum_features:
            for feat2 in regime_flags:
                new_col = f"{feat1}_in_{feat2}"
                panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
                interaction_count += 1
        print(f"      Generated {interaction_count - start_count} Mom×Regime interactions")
        
        # 5.2 Volatility × Regime Flags
        print("  [5.2] Volatility × Regime Flags...")
        start_count = interaction_count
        for feat1 in volatility_features:
            for feat2 in regime_flags:
                new_col = f"{feat1}_in_{feat2}"
                panel_df[new_col] = (panel_df[feat1] * panel_df[feat2]).astype('float32')
                interaction_count += 1
        print(f"      Generated {interaction_count - start_count} Vol×Regime interactions")
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    print("\n[interaction] Validation checks...")
    
    # Check for Inf values
    inf_cols = []
    for col in panel_df.columns:
        if col not in exclude_cols:
            if np.isinf(panel_df[col]).any():
                inf_cols.append(col)
    
    if inf_cols:
        print(f"  [WARNING] Found {len(inf_cols)} features with Inf values:")
        for col in inf_cols[:5]:  # Show first 5
            print(f"    - {col}")
        if len(inf_cols) > 5:
            print(f"    ... and {len(inf_cols) - 5} more")
        print("  [fix] Replacing Inf with NaN...")
        panel_df = panel_df.replace([np.inf, -np.inf], np.nan)
    
    # Check NaN percentage
    high_nan_cols = []
    for col in panel_df.columns:
        if col not in exclude_cols:
            nan_pct = panel_df[col].isna().sum() / len(panel_df) * 100
            if nan_pct > 50:
                high_nan_cols.append((col, nan_pct))
    
    if high_nan_cols:
        print(f"  [WARNING] Found {len(high_nan_cols)} features with > 50% NaN:")
        for col, pct in sorted(high_nan_cols, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {col}: {pct:.1f}% NaN")
        if len(high_nan_cols) > 5:
            print(f"    ... and {len(high_nan_cols) - 5} more")
    
    # Final summary
    final_feature_count = len(panel_df.columns)
    print(f"\n[interaction] ✓ Interaction generation complete!")
    print(f"[interaction] Total interactions generated: {interaction_count}")
    print(f"[interaction] Final feature count: {final_feature_count}")
    print(f"[interaction] Memory usage: {panel_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return panel_df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_feature_engineering(config: ResearchConfig) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    
    Parameters
    ----------
    config : ResearchConfig
        Configuration object
        
    Returns
    -------
    pd.DataFrame
        Panel with (Date, Ticker) MultiIndex
    """
    start_time = time.time()
    print("="*80)
    print("CROSS-SECTIONAL MOMENTUM FEATURE ENGINEERING (REFACTORED)")
    print("="*80)
    
    # Validate config
    config.validate()
    
    # -------------------------------------------------------------------------
    # 1. Load universe
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading ETF universe...")
    universe_df = pd.read_csv(config.paths.universe_csv)
    tickers = universe_df['ticker'].tolist()
    print(f"Universe size: {len(tickers)} ETFs")
    
    # -------------------------------------------------------------------------
    # 2. Download OHLCV data
    # -------------------------------------------------------------------------
    # IMPORTANT: All ETF data is downloaded starting from config.time.start_date
    # (recommended: 2007-11-04 for full ETF history). Filtering is applied LATER
    # in the walk-forward engine based on:
    #
    # ETF FILTERING CRITERIA (applied at each rebalance date):
    # ========================================================
    # 1. Core Universe Filter (in_core_after_duplicates == True):
    #    - Removes leveraged ETFs (e.g., TQQQ, SQQQ)
    #    - Removes inverse ETFs 
    #    - Removes non-canonical duplicates (keeps one ETF per duplicate group)
    #
    # 2. Liquidity Filter (ADV_63_Rank >= min_adv_percentile):
    #    - Default: 30th percentile (top 70% by liquidity)
    #    - Uses 63-day average dollar volume
    #    - Cross-sectional rank computed at each date
    #
    # 3. Data Quality Filter (non-NaN features >= min_data_quality):
    #    - Default: 80% of features must be non-NaN
    #    - Ensures sufficient data for model scoring
    #
    # 4. History Requirement (sufficient historical data):
    #    - Each ticker must have FEATURE_MAX_LAG_DAYS + TRAINING_WINDOW_DAYS
    #      of historical data before entering the eligible universe
    #    - Default: 252 + 1260 = 1512 days (~6 years)
    #    - This ensures we can calculate features AND train models
    #
    # Note: ETFs are added to the eligible universe as soon as they meet
    # all 4 criteria, allowing new ETFs to enter over time.
    # ========================================================
    
    print("\n[2/10] Downloading OHLCV data...")
    
    # Calculate warmup days needed for ALL features to be valid at formation_start
    # Two components:
    # 1. FEATURE LOOKBACK: Longest lookback in pct_change features (e.g., Close%-252)
    # 2. ROLLING WINDOW WARMUP: Longest rolling window (e.g., RSI, Bollinger, etc.)
    # 
    # Close%-252 = (Close / Close.shift(252) - 1) * 100
    # This needs 252 PRIOR days of data to produce the FIRST valid value.
    # Then we need additional days for rolling windows on top of that.
    feature_max_lookback = 252  # Close%-252 is the longest lookback feature
    rolling_window_warmup = 63  # Longest rolling windows (std63, BollUp50, etc.)
    safety_buffer = 21  # Extra safety margin
    
    # Total warmup = feature lookback + rolling warmup + buffer
    warmup_days = feature_max_lookback + rolling_window_warmup + safety_buffer
    
    # Calculate extended download start to include full warmup period
    # Use 1.5x multiplier to convert trading days to calendar days
    download_start_dt = pd.to_datetime(config.time.start_date) - pd.Timedelta(days=int(warmup_days * 1.5))
    download_start_str = download_start_dt.strftime('%Y-%m-%d')
    
    print(f"[download] Step 1: Download ALL tickers with extended range for warmup")
    print(f"[download]   Formation start: {config.time.start_date}")
    print(f"[download]   Feature max lookback: {feature_max_lookback} trading days (Close%-252)")
    print(f"[download]   Rolling window warmup: {rolling_window_warmup} trading days")
    print(f"[download]   Total warmup needed: {warmup_days} trading days")
    print(f"[download]   Download start (extended): {download_start_str}")
    print(f"[download]   Download end: {config.time.end_date}")
    
    download_start = time.time()
    data_dict_raw = download_etf_data(
        tickers,
        download_start_str,  # Extended start to include warmup
        config.time.end_date
    )
    download_elapsed = time.time() - download_start
    
    if len(data_dict_raw) == 0:
        raise RuntimeError("No data downloaded!")
    
    n_downloaded = len(data_dict_raw)
    print(f"[download] Downloaded: {n_downloaded} tickers")
    print(f"[time] Data download completed in {download_elapsed:.2f} seconds ({download_elapsed/60:.2f} minutes)")
    
    # -------------------------------------------------------------------------
    # 2.1 STEP 2: Filter tickers based on formation window start date
    # -------------------------------------------------------------------------
    # ETFs that were not listed at the formation window start date are dropped.
    # This preserves ETFs that have trading history after our formation start,
    # even if they were listed later than the download start date.
    print("\n[2.1/10] Filtering tickers for research window...")
    
    # Use same warmup_days as calculated for download
    data_dict_filtered = filter_tickers_for_research_window(
        data_dict_raw,
        formation_start_date=config.time.start_date,
        warmup_days=warmup_days,  # Already defined above
        verbose=True
    )
    
    n_after_filter = len(data_dict_filtered)
    n_dropped = n_downloaded - n_after_filter
    print(f"[filter] Result: {n_after_filter} tickers kept, {n_dropped} dropped")
    
    # -------------------------------------------------------------------------
    # 2.2 STEP 3: Clean raw OHLCV data (forward-fill gaps, no look-ahead)
    # -------------------------------------------------------------------------
    print("\n[2.2/10] Cleaning raw OHLCV data...")
    
    data_dict = clean_raw_ohlcv_data(
        data_dict_filtered,
        ffill_limit=config.features.raw_data_ffill_limit,
        verbose=True
    )
    
    # Validate that cleaned data has no NaN in Close column
    tickers_with_nan_close = []
    for ticker, df in data_dict.items():
        if 'Close' in df.columns and df['Close'].isna().any():
            tickers_with_nan_close.append(ticker)
    
    if tickers_with_nan_close:
        print(f"[warning] {len(tickers_with_nan_close)} tickers still have NaN in Close after cleaning")
        print(f"[warning] These will be handled during feature imputation")
    else:
        print(f"[clean] ✓ All {len(data_dict)} tickers have clean Close data")
    
    print(f"\n[data] Summary after Steps 1-3:")
    print(f"       - Downloaded from yfinance: {n_downloaded} tickers")
    print(f"       - After formation window filter: {n_after_filter} tickers")
    print(f"       - Ready for feature engineering: {len(data_dict)} tickers")
    
    # -------------------------------------------------------------------------
    # 3. Download macro data (WITH EXTENDED RANGE for warmup)
    # -------------------------------------------------------------------------
    print("\n[3/10] Downloading macro data...")
    macro_start = time.time()
    data_manager = CrossSecMomDataManager(config.paths.data_dir)
    macro_tickers = {
        'vix': '^VIX',         # VIX volatility index
        'yield_10y': '^TNX',   # 10-year Treasury yield
        'yield_2y': '^IRX',    # 2-year Treasury yield (proxy, actually 13-week)
        'tbill_3m': '^IRX',    # 3-month T-bill rate (same proxy)
    }
    
    # CRITICAL: Use extended download start (same as OHLCV) to avoid NaN
    # VIX z-score needs 252 days of rolling history!
    print(f"[macro] Using extended start date for macro data: {download_start_str}")
    print(f"[macro] This ensures VIX z-score (252-day rolling) has valid data")
    
    macro_data = data_manager.load_or_download_macro_data(
        macro_tickers,
        start_date=download_start_str,  # Use extended start, NOT config.time.start_date
        end_date=config.time.end_date
    )
    macro_elapsed = time.time() - macro_start
    print(f"[time] Macro data download completed in {macro_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 3.1 VALIDATE DATA AVAILABILITY (EARLY CHECK)
    # -------------------------------------------------------------------------
    print("\n[3.1/10] Validating data availability...")
    validation_results = validate_data_availability(
        data_dict=data_dict,
        macro_data=macro_data,
        formation_start=config.time.start_date,
        warmup_days=warmup_days,
        verbose=True
    )
    
    if not validation_results['valid']:
        print("\n" + "="*80)
        print("WARNING: Data validation detected issues!")
        print("="*80)
        print("Some data sources do not have sufficient history for warmup period.")
        print("This may cause NaN in features. Review issues above.")
        print("Continuing anyway, but features may be dropped due to NaN.")
        print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # 4. Feature engineering (parallel) - STEP 4: Compute features
    # -------------------------------------------------------------------------
    print("\n[4/10] Engineering features per ticker...")
    print(f"[features] Computing technical and statistical transformations...")
    
    feature_eng_start = time.time()
    results = Parallel(n_jobs=config.compute.n_jobs, backend='threading', verbose=5)(
        delayed(process_ticker)(ticker, data_dict[ticker], config.universe.adv_window, config)
        for ticker in data_dict.keys()
    )
    feature_eng_elapsed = time.time() - feature_eng_start
    print(f"[time] Feature engineering completed in {feature_eng_elapsed:.2f} seconds ({feature_eng_elapsed/60:.2f} minutes)")
    
    # -------------------------------------------------------------------------
    # 5. Combine into panel structure
    # -------------------------------------------------------------------------
    print("\n[5/10] Building panel structure...")
    
    panel_list = []
    for ticker, feat_df in zip(data_dict.keys(), results):
        if feat_df.empty:
            continue
        panel_list.append(feat_df)
    
    # Concatenate
    panel_df = pd.concat(panel_list, ignore_index=False)
    panel_df.index.name = 'Date'
    panel_df = panel_df.reset_index()
    
    # Sort and set multi-index
    panel_df = panel_df.sort_values(['Date', 'Ticker'])
    panel_df = panel_df.set_index(['Date', 'Ticker'])
    
    print(f"Panel shape: {panel_df.shape}")
    print(f"Date range: {panel_df.index.get_level_values('Date').min()} to {panel_df.index.get_level_values('Date').max()}")
    print(f"Unique tickers: {panel_df.index.get_level_values('Ticker').nunique()}")
    
    # -------------------------------------------------------------------------
    # STEP 4 CHECK: Validate NaN after feature computation
    # -------------------------------------------------------------------------
    print("\n[check] Step 4 validation: NaN after feature computation...")
    panel_df_reset = panel_df.reset_index()
    exclude_check = {'Ticker', 'Date', 'Close'}
    exclude_check.update([c for c in panel_df.columns if c.startswith('FwdRet')])
    feature_cols_check = [c for c in panel_df.columns if c not in exclude_check]
    
    nan_after_features = panel_df[feature_cols_check].isna().sum().sum()
    total_cells = len(panel_df) * len(feature_cols_check)
    nan_pct = nan_after_features / total_cells * 100
    print(f"[check]   Features: {len(feature_cols_check)}")
    print(f"[check]   NaN cells: {nan_after_features:,} / {total_cells:,} ({nan_pct:.2f}%)")
    print(f"[check]   This NaN is expected from rolling windows (warmup period)")
    
    # -------------------------------------------------------------------------
    # 6. Add forward returns
    # -------------------------------------------------------------------------
    print("\n[6/10] Computing forward returns...")
    fwd_ret_start = time.time()
    panel_df = panel_df.reset_index()
    panel_df = add_forward_returns(panel_df, config.time.HOLDING_PERIOD_DAYS, config)
    fwd_ret_elapsed = time.time() - fwd_ret_start
    print(f"[time] Forward returns computed in {fwd_ret_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 7. Add cross-sectional features
    # -------------------------------------------------------------------------
    print("\n[7/10] Adding relative return features...")
    rel_ret_start = time.time()
    panel_df = add_relative_return_features(panel_df, lookbacks=[5, 20, 60])
    rel_ret_elapsed = time.time() - rel_ret_start
    print(f"[time] Relative return features added in {rel_ret_elapsed:.2f} seconds")
    
    print("\n[7.1/10] Adding correlation features...")
    corr_start = time.time()
    panel_df = add_correlation_features(panel_df, window=20)
    corr_elapsed = time.time() - corr_start
    print(f"[time] Correlation features added in {corr_elapsed:.2f} seconds")
    
    print("\n[7.2/10] Adding asset type flags...")
    asset_start = time.time()
    panel_df = add_asset_type_flags(panel_df, config)
    asset_elapsed = time.time() - asset_start
    print(f"[time] Asset type flags added in {asset_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 7.3. Add macro features
    # -------------------------------------------------------------------------
    print("\n[7.3/10] Adding macro/regime features...")
    macro_feat_start = time.time()
    panel_df = add_macro_features(panel_df, macro_data, config)
    macro_feat_elapsed = time.time() - macro_feat_start
    print(f"[time] Macro features added in {macro_feat_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 8. Generate interaction features
    # -------------------------------------------------------------------------
    print("\n[8/10] Generating exhaustive interaction features...")
    interaction_start = time.time()
    panel_df = generate_interaction_features(panel_df, config)
    interaction_elapsed = time.time() - interaction_start
    print(f"[time] Interaction features added in {interaction_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # STEP 4 FINAL CHECK: Validate NaN after ALL feature computations
    # -------------------------------------------------------------------------
    print("\n[check] Step 4 validation: NaN after ALL feature computations (incl. interactions)...")
    exclude_cols = {'Ticker', 'Date', 'Close', 'ADV_63', 'ADV_63_Rank'}
    exclude_cols.update([c for c in panel_df.columns if c.startswith('FwdRet')])
    feature_cols_all = [c for c in panel_df.columns if c not in exclude_cols]
    
    nan_after_all = panel_df[feature_cols_all].isna().sum().sum()
    total_cells_all = len(panel_df) * len(feature_cols_all)
    nan_pct_all = nan_after_all / total_cells_all * 100
    print(f"[check]   Total features (incl. interactions): {len(feature_cols_all)}")
    print(f"[check]   NaN cells: {nan_after_all:,} / {total_cells_all:,} ({nan_pct_all:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 9. TRIM PANEL TO FORMATION WINDOW FIRST (before NaN threshold check)
    # -------------------------------------------------------------------------
    # CRITICAL: The warmup period has EXPECTED NaN values (Close%-252 needs 252 days).
    # We must trim to formation window FIRST, then check NaN thresholds.
    # Otherwise we incorrectly drop features that have valid data in formation window.
    print("\n[9/10] Trimming panel to formation window FIRST (before NaN check)...")
    rows_before_trim = len(panel_df)
    panel_df = trim_panel_to_formation_window(
        panel_df,
        formation_start=config.time.start_date,
        formation_end=config.time.end_date,
        verbose=True
    )
    rows_after_trim = len(panel_df)
    print(f"[trim] Warmup rows removed: {rows_before_trim - rows_after_trim:,}")
    
    # -------------------------------------------------------------------------
    # 9.1 NaN CLEANING: Drop high-NaN features, impute remaining (STEP 5)
    # -------------------------------------------------------------------------
    # Now that warmup rows are removed, NaN % reflects actual formation window quality
    print("\n[9.1/10] Step 5: NaN cleaning - drop >10% NaN features, impute rest...")
    nan_threshold = config.features.feature_nan_threshold
    
    n_features_before = len(feature_cols_all)
    nan_before_total = panel_df[feature_cols_all].isna().sum().sum()
    
    # Drop high-NaN features (>10% NaN in formation window = structural data issues)
    nan_pct_by_col = panel_df[feature_cols_all].isna().sum() / len(panel_df)
    high_nan_features = nan_pct_by_col[nan_pct_by_col > nan_threshold].index.tolist()
    
    if high_nan_features:
        print(f"[nan] Dropping {len(high_nan_features)} features with >{nan_threshold:.0%} NaN in formation window:", flush=True)
        # Show top 10 worst offenders
        worst = nan_pct_by_col[high_nan_features].sort_values(ascending=False).head(10)
        for col, pct in worst.items():
            print(f"    - {col}: {pct:.1%} NaN", flush=True)
        if len(high_nan_features) > 10:
            print(f"    ... and {len(high_nan_features) - 10} more", flush=True)
        
        panel_df = panel_df.drop(columns=high_nan_features)
    else:
        print(f"[nan] ✓ No features with >{nan_threshold:.0%} NaN in formation window!")
    
    # Update feature list after dropping
    feature_cols = [c for c in feature_cols_all if c not in high_nan_features]
    n_features_after_drop = len(feature_cols)
    
    # Impute remaining NaN with cross-sectional median (innocent NaN from rolling windows)
    if config.features.enable_nan_imputation:
        print(f"[nan] Imputing remaining NaN with cross-sectional median (no look-ahead)...")
        panel_df = impute_nan_cross_sectional(panel_df, feature_cols, verbose=True)
    
    # Count NaN after cleaning
    nan_after_total = panel_df[feature_cols].isna().sum().sum()
    
    print(f"\n[nan] Feature NaN summary (Step 5):")
    print(f"    - Features before cleaning: {n_features_before}")
    print(f"    - Features dropped (>{nan_threshold:.0%} NaN in formation): {len(high_nan_features)}")
    print(f"    - Features remaining: {n_features_after_drop}")
    print(f"    - NaN before imputation: {nan_before_total:,}")
    print(f"    - NaN after imputation: {nan_after_total:,}")
    
    # -------------------------------------------------------------------------
    # 9.2 Add ADV rank for filtering
    # -------------------------------------------------------------------------
    print("\n[9.2/10] Adding ADV cross-sectional rank...")
    adv_col = f'ADV_{config.universe.adv_window}'
    panel_df = add_adv_rank(panel_df, adv_col)
    
    # Set final index
    panel_df = panel_df.set_index(['Date', 'Ticker']).sort_index()
    
    # -------------------------------------------------------------------------
    # STEP 5 CHECK: Final NaN validation before delivery to feature selection
    # -------------------------------------------------------------------------
    print("\n[check] Step 5 validation: Final NaN check before delivery to feature_selection...")
    is_clean = final_nan_check(panel_df, stage="final")
    
    if is_clean:
        print(f"[check] ✓ Panel is NaN-free and ready for feature selection!")
    else:
        # This should never happen if our NaN handling is correct
        # But if it does, we fail loudly so the user knows to fix feature_engineering
        raise ValueError(
            "\n" + "="*80 + "\n"
            "FEATURE ENGINEERING FAILED: NaN values remain in output data!\n"
            "\n"
            "This indicates a bug in the NaN handling logic. Feature selection\n"
            "CANNOT proceed with NaN data (ElasticNet will produce zero coefficients).\n"
            "\n"
            "Please review the NaN cleaning steps above and fix the source of NaN.\n"
            + "="*80
        )
    
    # -------------------------------------------------------------------------
    # 10. Save outputs (STEP 6)
    # -------------------------------------------------------------------------
    print("\n[10/10] Step 6: Saving clean panel to disk...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.paths.panel_parquet), exist_ok=True)
    
    # Save panel data
    panel_df.to_parquet(config.paths.panel_parquet, engine='pyarrow', compression='snappy')
    print(f"[save] Features saved to: {config.paths.panel_parquet}")
    print(f"[save] File size: {os.path.getsize(config.paths.panel_parquet) / 1024 / 1024:.1f} MB")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total features: {len([c for c in panel_df.columns if c not in ['Close']])}")
    print(f"Forward return horizon: {config.time.HOLDING_PERIOD_DAYS} days")
    print(f"\nPanel dimensions: {panel_df.shape}")
    print(f"Memory usage: {panel_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"\nExecution time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Sample data check
    print("\n" + "="*80)
    print("SAMPLE DATA (most recent date)")
    print("="*80)
    latest_date = panel_df.index.get_level_values('Date').max()
    sample = panel_df.loc[latest_date].head(5)
    
    # Display available columns (some may exist, some may not depending on data)
    display_cols = ['Close', 'Close%-21', 'Close%-63', 'Close_Mom21', 'Close_RSI14', 'ADV_63', f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}']
    available_cols = [col for col in display_cols if col in sample.columns]
    print(sample[available_cols].to_string())
    
    print("\n[done] Feature engineering complete!")
    return panel_df


if __name__ == "__main__":
    from config import get_default_config
    
    # Load config
    config = get_default_config()
    
    # Run pipeline
    panel_df = run_feature_engineering(config)
