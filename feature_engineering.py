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
import re
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import ResearchConfig
from data_manager import download_etf_data, CrossSecMomDataManager, filter_tickers_for_research_window, clean_raw_ohlcv_data  # Import from data_manager instead

# ============================================================================
# FACTOR FAMILY CLASSIFICATION
# ============================================================================
# Used for factor grouping in LARS and feature family reporting

FACTOR_GROUPS = {
    'momentum': {
        'patterns': ['Close%-', 'RSI', 'Williams', 'MACD', 'ROC', 'Mom', '_lag', 'Rel', '_vs_'],
        'target_representatives': 5,
    },
    'volatility': {
        'patterns': ['std', 'ATR', 'BBW', 'parkinson', 'garman_klass', 'rogers_satchell', 'skew', 'kurt', 'Ret1dZ'],
        'target_representatives': 5,
    },
    'volume': {
        'patterns': ['volume', 'rel_vol', 'obv', 'up_down_vol', 'pv_corr', 'vol_per_atr'],
        'target_representatives': 4,
    },
    'liquidity': {
        'patterns': ['amihud', 'spread', 'kyle', 'illiq', 'roll_spread', 'cs_spread', 'ADV_'],
        'target_representatives': 4,
    },
    'trend': {
        'patterns': ['MA', 'EMA', 'adx', 'trend_r2', 'slope', 'trend_strength', 'trend_regime', 'Boll'],
        'target_representatives': 4,
    },
    'risk': {
        'patterns': ['beta', 'idio', 'corr_mkt', 'semi_vol', 'max_dd', 'var_', 'cvar', 'down_corr', 'DD', 'Hurst', 'Corr',
                     'downside_beta', 'r_squared', 'drawdown_corr', 'corr_VIX', 'corr_MOVE', 'beta_VT', 'beta_BNDW'],
        'target_representatives': 5,
    },
    'macro': {
        'patterns': ['vix', 'yc_', 'hy_spread', 'fci', 'fsi', 'claims', 'real_rate', 'credit', 'short_rate',
                     'baa_aaa', 'indpro', 'retail', 'unemployment', 'cpi_yoy', 'pce_yoy', 'breakeven', 'growth_',
                     'inflation', 'quality_spread', 'tsy_', 'spread_zscore', 'spread_momentum'],
        'target_representatives': 5,
    },
    'sentiment': {
        'patterns': ['sentiment', 'epu', 'umich', 'uncertainty'],
        'target_representatives': 3,
    },
    'structure': {
        'patterns': ['regime', 'streak', 'zscore', 'days_since', 'alignment', 'pct_from', 'vol_of_vol',
                     'crash_flag', 'meltup_flag', 'high_vol', 'low_vol', 'is_bond', 'is_equity', 'is_real_asset', 'is_sector'],
        'target_representatives': 4,
    },
    'interaction': {
        'patterns': ['_x_', '_div_', '_minus_', '_sq', '_cb', '_in_'],
        'target_representatives': 10,
    },
}


def classify_feature_family(feature_name: str) -> str:
    """
    Classify a feature into its factor family based on name patterns.
    
    Returns family name or 'other' if no match.
    
    NOTE: Interaction patterns are checked FIRST to ensure features like
    'Close%-1_x_Close%-2' are classified as 'interaction', not 'momentum'.
    """
    feature_lower = feature_name.lower()
    
    # Check interaction patterns FIRST (highest priority)
    # This ensures Mom×Vol interactions aren't classified as momentum
    interaction_patterns = FACTOR_GROUPS['interaction']['patterns']
    for pattern in interaction_patterns:
        if pattern.lower() in feature_lower:
            return 'interaction'
    
    # Then check other families
    for family, config in FACTOR_GROUPS.items():
        if family == 'interaction':
            continue  # Already checked above
        for pattern in config['patterns']:
            if pattern.lower() in feature_lower:
                return family
    
    return 'other'


def get_family_features(feature_columns: list) -> Dict[str, list]:
    """
    Group feature columns by factor family.
    
    Returns dict of {family: [feature_names]}
    """
    families = {family: [] for family in FACTOR_GROUPS.keys()}
    families['other'] = []
    
    for col in feature_columns:
        family = classify_feature_family(col)
        families[family].append(col)
    
    return families


# ============================================================================
# HORIZON BUCKET CLASSIFICATION
# ============================================================================
# 4 horizon buckets: {1}, {2-5}, {6-21}, {22+}
HORIZON_BOUNDS = [(1, 1), (2, 5), (6, 21), (22, 10_000_000)]
HORIZON_BUCKET_NAMES = ["H1", "H2_5", "H6_21", "H22p"]


def _last_int_token(s: str) -> Optional[int]:
    """Extract last integer from a string."""
    m = re.findall(r"(\d+)", s)
    return int(m[-1]) if m else None


def infer_horizon_days(name: str) -> Optional[int]:
    """
    Infer the horizon/lookback days from a feature name.
    
    Parses common patterns like:
    - Close%-21 -> 21
    - _lag5 -> 5
    - _MA63 -> 63
    - _std21 -> 21
    - _RSI14 -> 14
    
    Returns None if no horizon can be inferred.
    """
    # Explicit day-bearing patterns first (returns / lags / windows)
    m = re.search(r"%-(-?\d+)", name)           # e.g. Close%-5, Close%--21
    if m: return abs(int(m.group(1)))
    m = re.search(r"_lag(\d+)", name, re.I)    # e.g. _lag3
    if m: return int(m.group(1))
    m = re.search(r"_MA(\d+)", name)            # e.g. _MA21
    if m: return int(m.group(1))
    m = re.search(r"_EMA(\d+)", name)           # e.g. _EMA63
    if m: return int(m.group(1))
    m = re.search(r"_std(\d+)", name)           # e.g. _std21
    if m: return int(m.group(1))
    m = re.search(r"_skew(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"_kurt(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"_Boll(?:Up|Lo|W)(\d+)", name)  # BBW21, BollUp21
    if m: return int(m.group(1))
    m = re.search(r"_Mom(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"_RSI(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"_ATR(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"_WilliamsR(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"_DD(\d+)", name)            # Drawdown windows
    if m: return int(m.group(1))
    m = re.search(r"_Hurst(\d+)", name)
    if m: return int(m.group(1))
    m = re.search(r"Corr(\d+)", name)           # Corr21_VT
    if m: return int(m.group(1))
    m = re.search(r"Rel(\d+)", name)            # Rel5_vs_VT
    if m: return int(m.group(1))
    m = re.search(r"beta.*_(\d+)", name, re.I)  # beta_VT_63
    if m: return int(m.group(1))
    m = re.search(r"idio.*_(\d+)", name, re.I)  # idio_vol_63
    if m: return int(m.group(1))
    m = re.search(r"corr_.*_(\d+)", name, re.I) # corr_VIX_63
    if m: return int(m.group(1))
    # Fallback: last integer token (often a window/lag)
    return _last_int_token(name)


def horizon_bucket(days: Optional[int]) -> str:
    """
    Map horizon days to a bucket label.
    
    Buckets:
    - H1: 1 day
    - H2_5: 2-5 days  
    - H6_21: 6-21 days
    - H22p: 22+ days
    
    Returns "H22p" if days is None (conservative default).
    """
    if days is None:
        return "H22p"  # conservative default to "longer-term"
    for (lo, hi), label in zip(HORIZON_BOUNDS, HORIZON_BUCKET_NAMES):
        if lo <= days <= hi:
            return label
    return "H22p"


def get_feature_bucket(feature_name: str) -> Tuple[str, str]:
    """
    Get (family, horizon_bucket) for a feature.
    
    Returns tuple of (family_name, horizon_bucket_name).
    """
    family = classify_feature_family(feature_name)
    days = infer_horizon_days(feature_name)
    hb = horizon_bucket(days)
    return (family, hb)


def get_features_by_bucket(feature_columns: List[str]) -> Dict[Tuple[str, str], List[str]]:
    """
    Group features by (family, horizon_bucket) bucket.
    
    Returns dict of {(family, horizon_bucket): [feature_names]}
    """
    buckets = {}
    for col in feature_columns:
        bucket = get_feature_bucket(col)
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(col)
    return buckets


def print_bucket_summary(feature_columns: List[str]) -> None:
    """Print a summary of features by (family, horizon) bucket."""
    buckets = get_features_by_bucket(feature_columns)
    
    print("\n" + "="*70)
    print("FEATURE BUCKET SUMMARY (family × horizon)")
    print("="*70)
    
    # Aggregate by family and horizon separately
    family_counts = {}
    horizon_counts = {h: 0 for h in HORIZON_BUCKET_NAMES}
    
    for (family, hb), features in sorted(buckets.items()):
        n = len(features)
        family_counts[family] = family_counts.get(family, 0) + n
        horizon_counts[hb] = horizon_counts.get(hb, 0) + n
    
    print("\nBy Family:")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {family:15s}: {count:4d}")
    
    print("\nBy Horizon:")
    for hb in HORIZON_BUCKET_NAMES:
        print(f"  {hb:6s}: {horizon_counts[hb]:4d}")
    
    print(f"\nTotal buckets: {len(buckets)}")
    print(f"Total features: {len(feature_columns)}")
    print("="*70)


def print_family_summary(feature_columns: list) -> None:
    """Print a summary of features by family."""
    families = get_family_features(feature_columns)
    
    print("\n" + "="*60)
    print("FEATURE FAMILY SUMMARY")
    print("="*60)
    
    total = 0
    for family, features in sorted(families.items(), key=lambda x: -len(x[1])):
        if features:
            print(f"  {family:15s}: {len(features):4d} features")
            total += len(features)
    
    print("-"*60)
    print(f"  {'TOTAL':15s}: {total:4d} features")
    print("="*60)


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
    # Target columns (y_*) will have NaN for the last HOLDING_PERIOD days
    # because forward returns cannot be computed for those dates
    exclude_cols = {'Ticker', 'Date', 'Close', 'ADV_63', 'ADV_63_Rank'}
    exclude_cols.update([c for c in panel_df.columns if c.startswith('FwdRet')])
    exclude_cols.update([c for c in panel_df.columns if c.startswith('y_')])
    
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
# V2 FEATURE FAMILIES: VOLUME, LIQUIDITY, RISK/BETA, STRUCTURE
# ============================================================================
# These families provide orthogonal signals to complement momentum/volatility

def compute_volume_features(
    close: pd.Series,
    volume: pd.Series,
    high: pd.Series = None,
    low: pd.Series = None,
) -> Dict[str, pd.Series]:
    """
    Compute volume-based features from OHLCV data.
    
    Features:
    - Relative volume (vs rolling mean)
    - Volume z-score
    - Volume trend (short vs long MA)
    - Price-volume correlation
    - OBV slope
    - Up/down volume ratio
    - Volume breakout/dryup flags
    """
    feats = {}
    
    if volume is None or volume.isna().all():
        return feats
    
    returns = close.pct_change()
    
    # ---- Relative Volume ----
    # Standardized on trading-month multiples: 10, 21, 63
    for window in [10, 21, 63]:
        vol_ma = volume.rolling(window, min_periods=window//2).mean()
        feats[f'rel_volume_{window}'] = (volume / vol_ma).astype('float32')
    
    # ---- Volume Z-Score ----
    # Standardized on trading-month multiples: 21, 63
    for window in [21, 63]:
        vol_mean = volume.rolling(window, min_periods=window//2).mean()
        vol_std = volume.rolling(window, min_periods=window//2).std()
        feats[f'volume_zscore_{window}'] = ((volume - vol_mean) / (vol_std + 1e-8)).astype('float32')
    
    # ---- Volume Trend (short/long ratio) ----
    # Standardized: 10/63 (2-week vs quarterly)
    vol_ma_10 = volume.rolling(10, min_periods=5).mean()
    vol_ma_63 = volume.rolling(63, min_periods=31).mean()
    feats['volume_trend_10_63'] = (vol_ma_10 / (vol_ma_63 + 1e-8)).astype('float32')
    
    # ---- Price-Volume Correlation ----
    for window in [21, 63]:
        pv_corr = returns.rolling(window, min_periods=window//2).corr(volume.pct_change())
        feats[f'pv_corr_{window}'] = pv_corr.astype('float32')
    
    # ---- OBV (On-Balance Volume) Slope ----
    obv = (np.sign(returns) * volume).cumsum()
    for window in [21, 63]:
        obv_slope = obv.diff(window) / window
        feats[f'obv_slope_{window}'] = obv_slope.astype('float32')
    
    # ---- Up/Down Volume Ratio ----
    for window in [21, 63]:
        up_vol = volume.where(returns > 0, 0).rolling(window, min_periods=window//2).sum()
        down_vol = volume.where(returns < 0, 0).rolling(window, min_periods=window//2).sum()
        feats[f'up_down_vol_ratio_{window}'] = (up_vol / (down_vol + 1e-8)).astype('float32')
    
    # ---- Volume Anomalies ----
    # Standardized to 63-day (quarterly) window
    vol_90pct = volume.rolling(63, min_periods=31).quantile(0.9)
    vol_10pct = volume.rolling(63, min_periods=31).quantile(0.1)
    feats['volume_breakout_63'] = (volume > vol_90pct).astype('float32')
    feats['volume_dryup_63'] = (volume < vol_10pct).astype('float32')
    
    # ---- Volume per ATR (normalized activity) ----
    if high is not None and low is not None:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=7).mean()
        feats['volume_per_atr'] = (volume / (atr_14 * close + 1e-8)).astype('float32')
    
    return feats


def compute_liquidity_features(
    close: pd.Series,
    volume: pd.Series,
    high: pd.Series = None,
    low: pd.Series = None,
    open_: pd.Series = None,
) -> Dict[str, pd.Series]:
    """
    Compute liquidity and microstructure proxy features.
    
    Features:
    - Amihud illiquidity (|return| / dollar_volume)
    - Roll implied spread
    - Corwin-Schultz high-low spread estimator
    - Kyle's lambda proxy
    - Range-based volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell)
    """
    feats = {}
    returns = close.pct_change()
    
    if volume is None or volume.isna().all():
        return feats
    
    dollar_volume = close * volume
    
    # ---- Amihud Illiquidity (2002) ----
    # Higher = less liquid
    amihud_daily = np.abs(returns) / (dollar_volume + 1e-8)
    for window in [21, 63, 126]:
        feats[f'amihud_{window}'] = amihud_daily.rolling(window, min_periods=window//2).mean().astype('float32')
    
    # Log-amihud for stability
    feats['log_amihud_63'] = np.log1p(amihud_daily.rolling(63, min_periods=30).mean() * 1e6).astype('float32')
    
    # ---- Roll (1984) Implied Spread ----
    # Spread ≈ 2 * sqrt(-cov(r_t, r_{t-1})) when cov < 0
    for window in [21, 63]:
        cov = returns.rolling(window, min_periods=window//2).cov(returns.shift(1))
        roll_spread = 2 * np.sqrt(np.maximum(-cov, 0))
        feats[f'roll_spread_{window}'] = roll_spread.astype('float32')
    
    # ---- Kyle's Lambda Proxy (price impact) ----
    # |return| / sqrt(volume)
    kyle_lambda = np.abs(returns) / (np.sqrt(volume) + 1e-8)
    for window in [21, 63]:
        feats[f'kyle_lambda_{window}'] = kyle_lambda.rolling(window, min_periods=window//2).mean().astype('float32')
    
    # ---- Range-Based Volatility Estimators ----
    if high is not None and low is not None:
        # Parkinson (1980) - High-Low only
        log_hl = np.log(high / low)
        parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2)
        for window in [21, 63]:
            parkinson_vol = np.sqrt(parkinson_var.rolling(window, min_periods=window//2).mean())
            feats[f'parkinson_vol_{window}'] = parkinson_vol.astype('float32')
        
        if open_ is not None:
            # Garman-Klass (1980) - OHLC
            log_hl_sq = (np.log(high / low)) ** 2
            log_co_sq = (np.log(close / open_)) ** 2
            gk_var = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
            for window in [21, 63]:
                gk_vol = np.sqrt(gk_var.rolling(window, min_periods=window//2).mean().clip(lower=0))
                feats[f'garman_klass_vol_{window}'] = gk_vol.astype('float32')
            
            # Rogers-Satchell (1991) - drift-independent
            rs_var = (np.log(high / close) * np.log(high / open_) + 
                      np.log(low / close) * np.log(low / open_))
            for window in [21, 63]:
                rs_vol = np.sqrt(rs_var.rolling(window, min_periods=window//2).mean().clip(lower=0))
                feats[f'rogers_satchell_vol_{window}'] = rs_vol.astype('float32')
        
        # ---- Corwin-Schultz (2012) High-Low Spread Estimator ----
        beta = (np.log(high / low)) ** 2
        high_2d = high.rolling(2).max()
        low_2d = low.rolling(2).min()
        gamma = (np.log(high_2d / low_2d)) ** 2
        
        sqrt_2 = np.sqrt(2)
        alpha_denom = 3 - 2 * sqrt_2
        alpha = ((np.sqrt(2 * beta) - np.sqrt(beta)) / alpha_denom - 
                 np.sqrt(gamma / alpha_denom))
        cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        cs_spread = cs_spread.clip(lower=0, upper=0.5)  # Cap at 50% spread
        
        for window in [21, 63]:
            feats[f'cs_spread_{window}'] = cs_spread.rolling(window, min_periods=window//2).mean().astype('float32')
    
    return feats


def compute_structure_features(
    close: pd.Series,
    returns: pd.Series,
    volatility_21: pd.Series = None,
) -> Dict[str, pd.Series]:
    """
    Compute structure, shape, and regime features.
    
    Features:
    - Volatility regime (low/med/high quantile)
    - Trend regime (strong up/chop/strong down)
    - Momentum streak length
    - Days since sign flip
    - Price z-score vs MA
    - Mean reversion indicators
    - Trend R² (strength)
    """
    feats = {}
    
    # ---- Volatility Regime ----
    realized_vol = returns.rolling(21, min_periods=10).std()
    vol_pct = realized_vol.rolling(252, min_periods=126).rank(pct=True)
    feats['vol_regime_pct'] = vol_pct.astype('float32')
    
    # Vol regime classification (0=low, 1=med, 2=high)
    feats['vol_regime_class'] = pd.cut(
        vol_pct, bins=[0, 0.33, 0.67, 1.0], labels=[0, 1, 2], include_lowest=True
    ).astype('float32')
    
    # Time in current vol regime
    vol_class = feats['vol_regime_class']
    vol_regime_change = vol_class != vol_class.shift(1)
    vol_regime_duration = vol_regime_change.astype(int).groupby(vol_regime_change.cumsum()).cumcount()
    feats['vol_regime_duration'] = vol_regime_duration.astype('float32')
    
    # Vol of vol
    feats['vol_of_vol_63'] = realized_vol.rolling(63, min_periods=30).std().astype('float32')
    
    # ---- Trend Regime ----
    mom_21 = close.pct_change(21)
    mom_63 = close.pct_change(63)
    vol_21 = volatility_21 if volatility_21 is not None else realized_vol
    
    # Trend strength = |momentum| / volatility
    trend_strength = np.abs(mom_21) / (vol_21 + 1e-8)
    feats['trend_strength_21'] = trend_strength.astype('float32')
    
    # Trend alignment (short and medium momentum same sign)
    aligned = (np.sign(mom_21) == np.sign(mom_63)).astype(float)
    feats['trend_alignment'] = aligned.astype('float32')
    
    # Trend regime score (-2 to +2)
    direction = np.sign(mom_21)
    strong = (trend_strength > 1) & (aligned == 1)
    trend_score = direction * (1 + strong.astype(float))
    trend_score = trend_score.where(trend_strength >= 0.5, 0)  # Chop if weak
    feats['trend_regime_score'] = trend_score.astype('float32')
    
    # ---- Streak / Persistence Features ----
    # Days since last sign flip of momentum
    mom_sign = np.sign(mom_21)
    sign_flip = mom_sign != mom_sign.shift(1)
    days_since_flip = sign_flip.astype(int).groupby(sign_flip.cumsum()).cumcount()
    feats['days_since_mom_flip'] = days_since_flip.astype('float32')
    
    # Current up/down streak (daily returns)
    ret_sign = np.sign(returns)
    streak_change = ret_sign != ret_sign.shift(1)
    streak_length = streak_change.astype(int).groupby(streak_change.cumsum()).cumcount() + 1
    feats['price_streak'] = (streak_length * ret_sign).astype('float32')
    
    # ---- Mean Reversion Indicators ----
    for window in [50, 200]:
        sma = close.rolling(window, min_periods=window//2).mean()
        std = close.rolling(window, min_periods=window//2).std()
        zscore = (close - sma) / (std + 1e-8)
        feats[f'price_zscore_{window}'] = zscore.astype('float32')
    
    # Distance from recent high/low
    high_52w = close.rolling(252, min_periods=126).max()
    low_52w = close.rolling(252, min_periods=126).min()
    feats['pct_from_52w_high'] = ((close - high_52w) / high_52w).astype('float32')
    feats['pct_from_52w_low'] = ((close - low_52w) / low_52w).astype('float32')
    
    # ---- Trend R² (Goodness of Fit) ----
    for window in [21, 63]:
        def trend_r2(prices):
            if len(prices) < window // 2:
                return np.nan
            y = prices.values
            x = np.arange(len(y))
            
            # Fit linear regression
            x_mean = x.mean()
            y_mean = y.mean()
            
            ss_xy = ((x - x_mean) * (y - y_mean)).sum()
            ss_xx = ((x - x_mean) ** 2).sum()
            ss_yy = ((y - y_mean) ** 2).sum()
            
            if ss_xx == 0 or ss_yy == 0:
                return np.nan
            
            r = ss_xy / np.sqrt(ss_xx * ss_yy)
            return r ** 2
        
        r2 = close.rolling(window, min_periods=window//2).apply(trend_r2, raw=False)
        feats[f'trend_r2_{window}'] = r2.astype('float32')
    
    return feats


# ============================================================================
# RISK / BETA / CORRELATION FEATURES
# ============================================================================

def compute_risk_beta_features(
    returns: pd.Series,
    market_returns: pd.Series,
    close: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Compute risk, beta, and correlation structure features.
    
    Features (25+ total):
    - Rolling beta to market (SPY) at 63, 126, 252 days
    - Beta stability
    - Idiosyncratic volatility and ratio
    - Semi-volatility (downside only)
    - Max drawdown at multiple horizons
    - Days since peak
    - VaR and CVaR (5%)
    - Return skewness and kurtosis
    - Correlation with market
    - Downside correlation and correlation asymmetry
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns for the asset
    market_returns : pd.Series
        Daily log returns for the market (SPY)
    close : pd.Series
        Close price series for drawdown calculations
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary of feature name -> Series
    """
    feats = {}
    
    if market_returns is None or market_returns.empty:
        return feats
    
    # Align returns
    aligned = pd.DataFrame({'asset': returns, 'market': market_returns}).dropna()
    if len(aligned) < 63:
        return feats
    
    asset_ret = aligned['asset']
    mkt_ret = aligned['market']
    
    # ---- Rolling Beta ----
    for window in [63, 126, 252]:
        cov = asset_ret.rolling(window, min_periods=window//2).cov(mkt_ret)
        var_mkt = mkt_ret.rolling(window, min_periods=window//2).var()
        beta = cov / (var_mkt + 1e-10)
        feats[f'beta_mkt_{window}'] = beta.reindex(returns.index).astype('float32')
    
    # ---- Beta Stability ----
    beta_63 = feats.get('beta_mkt_63', pd.Series(index=returns.index, dtype='float32'))
    feats['beta_stability_126'] = beta_63.rolling(126, min_periods=63).std().astype('float32')
    
    # ---- Idiosyncratic Volatility ----
    for window in [63, 126]:
        beta = feats.get(f'beta_mkt_{window}', pd.Series(index=returns.index, dtype='float32'))
        residuals = returns - beta * market_returns.reindex(returns.index)
        idio_vol = residuals.rolling(window, min_periods=window//2).std()
        feats[f'idio_vol_{window}'] = idio_vol.astype('float32')
    
    # ---- Idiosyncratic Ratio ----
    total_vol = returns.rolling(63, min_periods=30).std()
    idio_vol_63 = feats.get('idio_vol_63', pd.Series(index=returns.index, dtype='float32'))
    feats['idio_ratio_63'] = (idio_vol_63 / (total_vol + 1e-8)).astype('float32')
    
    # ---- Semi-Volatility (Downside Only) ----
    for window in [63, 126]:
        neg_returns = returns.where(returns < 0, 0)
        semi_vol = neg_returns.rolling(window, min_periods=window//2).std()
        feats[f'semi_vol_{window}'] = semi_vol.astype('float32')
    
    # ---- Max Drawdown ----
    for window in [63, 126, 252]:
        rolling_max = close.rolling(window, min_periods=window//2).max()
        drawdown = (close - rolling_max) / rolling_max
        max_dd = drawdown.rolling(window, min_periods=window//2).min()
        feats[f'max_dd_{window}'] = max_dd.astype('float32')
    
    # ---- Days Since Peak ----
    is_at_max = close == close.expanding().max()
    days_since_peak = (~is_at_max).astype(int).groupby(is_at_max.cumsum()).cumsum()
    feats['days_since_peak'] = days_since_peak.astype('float32')
    
    # ---- VaR and CVaR ----
    for window in [63, 126]:
        var_5pct = returns.rolling(window, min_periods=window//2).quantile(0.05)
        feats[f'var_5pct_{window}'] = var_5pct.astype('float32')
        
        # CVaR = mean of returns below VaR
        def cvar_func(x):
            var = np.percentile(x, 5)
            return x[x <= var].mean() if len(x[x <= var]) > 0 else var
        
        cvar = returns.rolling(window, min_periods=window//2).apply(cvar_func, raw=True)
        feats[f'cvar_5pct_{window}'] = cvar.astype('float32')
    
    # ---- Return Skewness and Kurtosis ----
    for window in [63, 126]:
        feats[f'return_skew_{window}'] = returns.rolling(window, min_periods=window//2).skew().astype('float32')
        feats[f'return_kurt_{window}'] = returns.rolling(window, min_periods=window//2).kurt().astype('float32')
    
    # ---- Correlation with Market ----
    for window in [63, 126, 252]:
        corr = asset_ret.rolling(window, min_periods=window//2).corr(mkt_ret)
        feats[f'corr_mkt_{window}'] = corr.reindex(returns.index).astype('float32')
    
    # ---- Downside Correlation ----
    for window in [63, 126]:
        down_days = mkt_ret < 0
        asset_down = asset_ret.where(down_days)
        mkt_down = mkt_ret.where(down_days)
        down_corr = asset_down.rolling(window, min_periods=window//4).corr(mkt_down)
        feats[f'down_corr_{window}'] = down_corr.reindex(returns.index).astype('float32')
    
    # ---- Correlation Asymmetry ----
    up_days = mkt_ret > 0
    asset_up = asset_ret.where(up_days)
    mkt_up = mkt_ret.where(up_days)
    up_corr = asset_up.rolling(63, min_periods=15).corr(mkt_up)
    down_corr_63 = feats.get('down_corr_63', pd.Series(index=returns.index, dtype='float32'))
    feats['corr_asymmetry_63'] = (down_corr_63 - up_corr.reindex(returns.index)).astype('float32')
    
    return feats


# ============================================================================
# CROSS-ASSET RELATIVE STRENGTH FEATURES
# ============================================================================

def compute_cross_asset_features(
    returns: pd.Series,
    benchmark_returns: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """
    Compute cross-asset relative strength features.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns for the asset
    benchmark_returns : Dict[str, pd.Series]
        Dictionary of benchmark returns:
        - 'spy': S&P 500 returns (equity market)
        - 'tlt': Long-term treasury returns (bonds)
        - 'gld': Gold returns (safe haven)
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary of feature name -> Series
    """
    feats = {}
    
    for bm_name, bm_returns in benchmark_returns.items():
        if bm_returns is None or bm_returns.empty:
            continue
        
        # Align
        aligned = pd.DataFrame({'asset': returns, 'benchmark': bm_returns}).dropna()
        if len(aligned) < 21:
            continue
        
        asset_ret = aligned['asset']
        bm_ret = aligned['benchmark']
        
        # ---- Relative Strength ----
        for window in [21, 63, 126]:
            asset_cum = asset_ret.rolling(window, min_periods=window//2).sum()
            bm_cum = bm_ret.rolling(window, min_periods=window//2).sum()
            rel_strength = asset_cum - bm_cum
            feats[f'rel_strength_{bm_name}_{window}'] = rel_strength.reindex(returns.index).astype('float32')
        
        # ---- Relative Strength Z-Score ----
        rel_str_63 = feats.get(f'rel_strength_{bm_name}_63')
        if rel_str_63 is not None:
            rs_zscore = (rel_str_63 - rel_str_63.rolling(252, min_periods=126).mean()) / (
                rel_str_63.rolling(252, min_periods=126).std() + 1e-8
            )
            feats[f'rel_strength_{bm_name}_zscore'] = rs_zscore.astype('float32')
        
        # ---- Rolling Correlation ----
        for window in [63, 126]:
            corr = asset_ret.rolling(window, min_periods=window//2).corr(bm_ret)
            feats[f'corr_{bm_name}_{window}'] = corr.reindex(returns.index).astype('float32')
    
    return feats


# ============================================================================
# MACRO / SENTIMENT FEATURES FROM FRED
# ============================================================================

def compute_macro_features_from_fred(
    date_index: pd.DatetimeIndex,
    fred_data: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """
    Compute macro/regime features from FRED data.
    
    Features:
    - Yield curve slope and momentum
    - Credit spreads and changes
    - Financial conditions indices
    - Sentiment indicators
    - Economic uncertainty
    
    Parameters
    ----------
    date_index : pd.DatetimeIndex
        Date index to align FRED data to (typically from ETF prices)
    fred_data : Dict[str, pd.Series]
        Dictionary of FRED series from load_or_download_fred_data()
        Keys: tsy_2y, tsy_10y, tsy_30y, yc_slope_10_2, ig_spread, hy_spread,
              breakeven_10y, chicago_fci, stl_fsi, umich_sentiment, epu_daily,
              initial_claims
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary of macro feature name -> Series
    """
    feats = {}
    
    # ---- Yield Curve Features ----
    if 'tsy_10y' in fred_data and 'tsy_2y' in fred_data:
        tsy_10y = fred_data['tsy_10y'].reindex(date_index, method='ffill')
        tsy_2y = fred_data['tsy_2y'].reindex(date_index, method='ffill')
        
        yc_slope = tsy_10y - tsy_2y
        feats['yc_slope_10_2'] = yc_slope.astype('float32')
        feats['yc_slope_momentum_21'] = yc_slope.diff(21).astype('float32')
        
        # Inversion flag
        feats['yc_inverted'] = (yc_slope < 0).astype('float32')
        
        # Yield curve z-score
        yc_mean = yc_slope.rolling(252, min_periods=126).mean()
        yc_std = yc_slope.rolling(252, min_periods=126).std()
        feats['yc_slope_zscore'] = ((yc_slope - yc_mean) / (yc_std + 1e-8)).astype('float32')
    
    if 'tsy_30y' in fred_data and 'tsy_10y' in fred_data and 'tsy_2y' in fred_data:
        tsy_30y = fred_data['tsy_30y'].reindex(date_index, method='ffill')
        tsy_10y = fred_data['tsy_10y'].reindex(date_index, method='ffill')
        tsy_2y = fred_data['tsy_2y'].reindex(date_index, method='ffill')
        
        # Curvature (butterfly)
        curvature = tsy_2y + tsy_30y - 2 * tsy_10y
        feats['yc_curvature'] = curvature.astype('float32')
    
    # Real rate
    if 'tsy_10y' in fred_data and 'breakeven_10y' in fred_data:
        tsy_10y = fred_data['tsy_10y'].reindex(date_index, method='ffill')
        breakeven = fred_data['breakeven_10y'].reindex(date_index, method='ffill')
        feats['real_rate_10y'] = (tsy_10y - breakeven).astype('float32')
    
    # ---- Credit Spread Features ----
    if 'hy_spread' in fred_data:
        hy_spread = fred_data['hy_spread'].reindex(date_index, method='ffill')
        feats['hy_spread'] = hy_spread.astype('float32')
        feats['hy_spread_momentum_21'] = hy_spread.diff(21).astype('float32')
        
        # Credit spread z-score
        cs_mean = hy_spread.rolling(252, min_periods=126).mean()
        cs_std = hy_spread.rolling(252, min_periods=126).std()
        feats['hy_spread_zscore'] = ((hy_spread - cs_mean) / (cs_std + 1e-8)).astype('float32')
        
        # Credit stress flag
        feats['credit_stress'] = (hy_spread > hy_spread.rolling(252).quantile(0.8)).astype('float32')
    
    if 'ig_spread' in fred_data and 'hy_spread' in fred_data:
        ig = fred_data['ig_spread'].reindex(date_index, method='ffill')
        hy = fred_data['hy_spread'].reindex(date_index, method='ffill')
        feats['quality_spread'] = (hy - ig).astype('float32')
    
    # ---- Financial Conditions ----
    if 'chicago_fci' in fred_data:
        fci = fred_data['chicago_fci'].reindex(date_index, method='ffill')
        feats['fci'] = fci.astype('float32')
        feats['fci_momentum_21'] = fci.diff(21).astype('float32')
        
        # Tight conditions flag (FCI > 0 means tighter than average)
        feats['fci_tight'] = (fci > 0).astype('float32')
    
    if 'stl_fsi' in fred_data:
        fsi = fred_data['stl_fsi'].reindex(date_index, method='ffill')
        feats['fsi'] = fsi.astype('float32')
        feats['fsi_stress'] = (fsi > 0).astype('float32')
    
    # ---- Sentiment / Uncertainty ----
    if 'umich_sentiment' in fred_data:
        sentiment = fred_data['umich_sentiment'].reindex(date_index, method='ffill')
        feats['consumer_sentiment'] = sentiment.astype('float32')
        
        # Sentiment momentum (3-month change)
        feats['sentiment_momentum_63'] = sentiment.diff(63).astype('float32')
        
        # Sentiment z-score
        sent_mean = sentiment.rolling(252, min_periods=126).mean()
        sent_std = sentiment.rolling(252, min_periods=126).std()
        feats['sentiment_zscore'] = ((sentiment - sent_mean) / (sent_std + 1e-8)).astype('float32')
    
    if 'epu_daily' in fred_data:
        epu = fred_data['epu_daily'].reindex(date_index, method='ffill')
        feats['epu'] = epu.astype('float32')
        
        # EPU z-score
        epu_mean = epu.rolling(252, min_periods=126).mean()
        epu_std = epu.rolling(252, min_periods=126).std()
        feats['epu_zscore'] = ((epu - epu_mean) / (epu_std + 1e-8)).astype('float32')
        
        # High uncertainty flag
        feats['epu_spike'] = (epu > epu.rolling(252).quantile(0.9)).astype('float32')
    
    # ---- Initial Claims (Labor Market) ----
    if 'initial_claims' in fred_data:
        claims = fred_data['initial_claims'].reindex(date_index, method='ffill')
        
        # 4-week moving average
        claims_4wk = claims.rolling(4, min_periods=2).mean()
        feats['claims_4wk_ma'] = claims_4wk.astype('float32')
        
        # Claims momentum (vs 4 weeks ago)
        feats['claims_momentum_4w'] = claims_4wk.pct_change(4).astype('float32')
        
        # Claims z-score
        claims_mean = claims_4wk.rolling(252, min_periods=126).mean()
        claims_std = claims_4wk.rolling(252, min_periods=126).std()
        feats['claims_zscore'] = ((claims_4wk - claims_mean) / (claims_std + 1e-8)).astype('float32')
    
    # =========================================================================
    # NEW MACRO FEATURES (V4): Yield Curve Levels, Credit, Growth, Inflation
    # =========================================================================
    
    # ---- Short Rate (3-month Treasury) ----
    if 'tsy_3mo' in fred_data:
        tsy_3mo = fred_data['tsy_3mo'].reindex(date_index, method='ffill')
        feats['short_rate_3mo'] = tsy_3mo.astype('float32')
        
        # 10Y-3M spread (additional yield curve measure)
        if 'tsy_10y' in fred_data:
            tsy_10y = fred_data['tsy_10y'].reindex(date_index, method='ffill')
            yc_10y_3mo = tsy_10y - tsy_3mo
            feats['yc_slope_10_3mo'] = yc_10y_3mo.astype('float32')
            
            # Inversion flag for 10Y-3M
            feats['yc_inverted_3mo'] = (yc_10y_3mo < 0).astype('float32')
            
            # Z-score
            yc_3mo_mean = yc_10y_3mo.rolling(252, min_periods=126).mean()
            yc_3mo_std = yc_10y_3mo.rolling(252, min_periods=126).std()
            feats['yc_slope_10_3mo_zscore'] = ((yc_10y_3mo - yc_3mo_mean) / (yc_3mo_std + 1e-8)).astype('float32')
    
    # ---- Baa-Aaa Credit Spread ----
    if 'baa_yield' in fred_data and 'aaa_yield' in fred_data:
        baa = fred_data['baa_yield'].reindex(date_index, method='ffill')
        aaa = fred_data['aaa_yield'].reindex(date_index, method='ffill')
        
        baa_aaa_spread = baa - aaa
        feats['baa_aaa_spread'] = baa_aaa_spread.astype('float32')
        feats['baa_aaa_spread_momentum_21'] = baa_aaa_spread.diff(21).astype('float32')
        
        # Z-score
        cs_mean = baa_aaa_spread.rolling(252, min_periods=126).mean()
        cs_std = baa_aaa_spread.rolling(252, min_periods=126).std()
        feats['baa_aaa_spread_zscore'] = ((baa_aaa_spread - cs_mean) / (cs_std + 1e-8)).astype('float32')
        
        # Credit stress regime
        feats['credit_stress_baa_aaa'] = (baa_aaa_spread > baa_aaa_spread.rolling(252).quantile(0.8)).astype('float32')
    
    # ---- Growth Proxies ----
    # Industrial Production (monthly -> forward-fill to daily)
    if 'indpro' in fred_data:
        indpro = fred_data['indpro'].reindex(date_index, method='ffill')
        
        # YoY change (% change vs 12 months ago, approximated by ~252 days)
        # For monthly data, we use 12-period lag after ffill
        indpro_yoy = indpro.pct_change(252) * 100
        feats['indpro_yoy'] = indpro_yoy.astype('float32')
        
        # Z-score of YoY growth
        ip_mean = indpro_yoy.rolling(252, min_periods=126).mean()
        ip_std = indpro_yoy.rolling(252, min_periods=126).std()
        feats['indpro_yoy_zscore'] = ((indpro_yoy - ip_mean) / (ip_std + 1e-8)).astype('float32')
        
        # Growth regime: positive/negative
        feats['growth_positive'] = (indpro_yoy > 0).astype('float32')
    
    # Retail Sales (monthly -> forward-fill to daily)
    if 'retail_sales' in fred_data:
        retail = fred_data['retail_sales'].reindex(date_index, method='ffill')
        
        # YoY change
        retail_yoy = retail.pct_change(252) * 100
        feats['retail_yoy'] = retail_yoy.astype('float32')
        
        # Z-score
        retail_mean = retail_yoy.rolling(252, min_periods=126).mean()
        retail_std = retail_yoy.rolling(252, min_periods=126).std()
        feats['retail_yoy_zscore'] = ((retail_yoy - retail_mean) / (retail_std + 1e-8)).astype('float32')
    
    # Unemployment Rate (monthly -> forward-fill)
    if 'unemployment_rate' in fred_data:
        unemp = fred_data['unemployment_rate'].reindex(date_index, method='ffill')
        feats['unemployment_rate'] = unemp.astype('float32')
        
        # Change vs 12 months ago (level change, not percentage)
        feats['unemployment_change_12m'] = unemp.diff(252).astype('float32')
        
        # Z-score
        unemp_mean = unemp.rolling(252, min_periods=126).mean()
        unemp_std = unemp.rolling(252, min_periods=126).std()
        feats['unemployment_zscore'] = ((unemp - unemp_mean) / (unemp_std + 1e-8)).astype('float32')
        
        # High unemployment flag
        feats['unemployment_elevated'] = (unemp > unemp.rolling(252).quantile(0.75)).astype('float32')
    
    # ---- Inflation Proxies ----
    # CPI YoY (monthly index -> compute YoY)
    if 'cpi_index' in fred_data:
        cpi = fred_data['cpi_index'].reindex(date_index, method='ffill')
        
        # YoY inflation rate
        cpi_yoy = (cpi / cpi.shift(252) - 1) * 100
        feats['cpi_yoy'] = cpi_yoy.astype('float32')
        
        # Z-score
        cpi_mean = cpi_yoy.rolling(252, min_periods=126).mean()
        cpi_std = cpi_yoy.rolling(252, min_periods=126).std()
        feats['cpi_yoy_zscore'] = ((cpi_yoy - cpi_mean) / (cpi_std + 1e-8)).astype('float32')
        
        # High inflation regime
        feats['inflation_high'] = (cpi_yoy > cpi_yoy.rolling(504).quantile(0.8)).astype('float32')
    
    # PCE YoY (monthly index -> compute YoY)
    if 'pce_index' in fred_data:
        pce = fred_data['pce_index'].reindex(date_index, method='ffill')
        
        # YoY inflation rate
        pce_yoy = (pce / pce.shift(252) - 1) * 100
        feats['pce_yoy'] = pce_yoy.astype('float32')
        
        # Z-score
        pce_mean = pce_yoy.rolling(252, min_periods=126).mean()
        pce_std = pce_yoy.rolling(252, min_periods=126).std()
        feats['pce_yoy_zscore'] = ((pce_yoy - pce_mean) / (pce_std + 1e-8)).astype('float32')
    
    # Breakeven (already have, add z-score if not present)
    if 'breakeven_10y' in fred_data and 'breakeven_10y' not in feats:
        be = fred_data['breakeven_10y'].reindex(date_index, method='ffill')
        feats['breakeven_10y'] = be.astype('float32')
        
        be_mean = be.rolling(252, min_periods=126).mean()
        be_std = be.rolling(252, min_periods=126).std()
        feats['breakeven_10y_zscore'] = ((be - be_mean) / (be_std + 1e-8)).astype('float32')
    
    return feats


# ============================================================================
# COMPOSITE REGIME INDICATOR
# ============================================================================

def compute_regime_composite(
    macro_features: Dict[str, pd.Series],
    vix: pd.Series = None,
) -> Dict[str, pd.Series]:
    """
    Compute composite regime indicators combining multiple signals.
    
    Parameters
    ----------
    macro_features : Dict[str, pd.Series]
        Macro features from compute_macro_features_from_fred()
    vix : pd.Series, optional
        VIX index values
        
    Returns
    -------
    Dict[str, pd.Series]
        - macro_regime_score: -3 to +3 (risk-off to risk-on)
        - risk_on: Flag for favorable macro conditions
        - risk_off: Flag for unfavorable macro conditions
        - vix_zscore: VIX z-score if VIX provided
    """
    feats = {}
    
    # Get features
    yc_slope = macro_features.get('yc_slope_10_2')
    hy_spread = macro_features.get('hy_spread')
    vix_z = None
    
    # Use yc_slope index as reference for alignment
    ref_index = yc_slope.index if yc_slope is not None else None
    
    if vix is not None and ref_index is not None:
        # Align VIX to macro features index
        vix_aligned = vix.reindex(ref_index, method='ffill')
        vix_mean = vix_aligned.rolling(252, min_periods=126).mean()
        vix_std = vix_aligned.rolling(252, min_periods=126).std()
        vix_z = (vix_aligned - vix_mean) / (vix_std + 1e-8)
        feats['vix_zscore'] = vix_z.astype('float32')
    
    # Build composite score
    if yc_slope is not None and hy_spread is not None and vix_z is not None:
        # Align hy_spread to same index
        hy_spread_aligned = hy_spread.reindex(ref_index, method='ffill')
        
        # Yield curve component: positive slope = risk-on
        yc_score = np.where(yc_slope > 0.5, 1, np.where(yc_slope < 0, -1, 0))
        
        # Credit spread component: low spreads = risk-on
        cs_pct = hy_spread_aligned.rolling(252, min_periods=126).rank(pct=True)
        cs_score = np.where(cs_pct < 0.3, 1, np.where(cs_pct > 0.7, -1, 0))
        
        # VIX component: low VIX = risk-on
        vix_score = np.where(vix_z < -0.5, 1, np.where(vix_z > 1, -1, 0))
        
        # Composite (-3 to +3)
        composite = yc_score + cs_score + vix_score
        feats['macro_regime_score'] = pd.Series(composite, index=ref_index).astype('float32')
        
        # Risk-on/off flags
        feats['risk_on'] = (feats['macro_regime_score'] >= 2).astype('float32')
        feats['risk_off'] = (feats['macro_regime_score'] <= -2).astype('float32')
    
    return feats


# ============================================================================
# ENHANCED RISK / FACTOR EXPOSURE FEATURES (V4)
# ============================================================================

def compute_risk_factor_features(
    etf_returns: pd.Series,
    macro_ref_returns: Dict[str, pd.Series],
    windows: List[int] = [21, 63, 126],
) -> Dict[str, pd.Series]:
    """
    Compute risk/factor exposure features for a single ETF at multiple timeframes.
    
    Uses rolling regressions of ETF returns on macro reference returns (VT, BNDW, 
    VIX, MOVE) to compute:
    - Rolling betas to each factor at 21/63/126 days
    - Idiosyncratic volatility (residual sigma) at 21/63/126 days
    - Downside beta (beta only when market is negative) at 21/63/126 days
    - Correlation in drawdowns at 21/63/126 days
    
    Parameters
    ----------
    etf_returns : pd.Series
        Daily returns for the ETF
    macro_ref_returns : Dict[str, pd.Series]
        Dictionary of macro reference returns:
        - 'vt': Vanguard Total World Stock returns
        - 'bndw': Vanguard Total World Bond returns  
        - 'vix': VIX daily changes (for correlation, not returns)
        - 'move': MOVE index daily changes
    windows : List[int]
        Rolling windows for computation (default: [21, 63, 126])
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary of risk feature name -> Series
    """
    feats = {}
    
    # Align all returns to ETF index
    idx = etf_returns.index
    
    # ---- Rolling Beta to VT (Global Equity) at multiple windows ----
    if 'vt' in macro_ref_returns:
        vt_ret = macro_ref_returns['vt'].reindex(idx, method='ffill')
        
        for window in windows:
            # Rolling covariance and variance
            cov_vt = etf_returns.rolling(window, min_periods=window//2).cov(vt_ret)
            var_vt = vt_ret.rolling(window, min_periods=window//2).var()
            beta_vt = cov_vt / (var_vt + 1e-10)
            feats[f'beta_VT_{window}'] = beta_vt.astype('float32')
            
            # Downside beta (only when VT is negative)
            vt_down = vt_ret.where(vt_ret < 0)
            etf_down = etf_returns.where(vt_ret < 0)
            cov_down = etf_down.rolling(window, min_periods=window//3).cov(vt_down)
            var_down = vt_down.rolling(window, min_periods=window//3).var()
            downside_beta = cov_down / (var_down + 1e-10)
            feats[f'downside_beta_VT_{window}'] = downside_beta.astype('float32')
            
            # Idiosyncratic volatility (residual from VT regression)
            predicted = beta_vt * vt_ret
            residual = etf_returns - predicted
            idio_vol = residual.rolling(window, min_periods=window//2).std()
            feats[f'idio_vol_{window}'] = idio_vol.astype('float32')
            
            # R-squared of the regression (systematic vs idiosyncratic)
            total_var = etf_returns.rolling(window, min_periods=window//2).var()
            r_squared = 1 - (idio_vol**2 / (total_var + 1e-10))
            feats[f'r_squared_VT_{window}'] = r_squared.clip(0, 1).astype('float32')
            
            # Drawdown Correlation
            etf_cum = (1 + etf_returns).cumprod()
            vt_cum = (1 + vt_ret).cumprod()
            etf_dd = etf_cum / etf_cum.rolling(window, min_periods=window//2).max() - 1
            vt_dd = vt_cum / vt_cum.rolling(window, min_periods=window//2).max() - 1
            dd_corr = etf_dd.rolling(window, min_periods=window//2).corr(vt_dd)
            feats[f'drawdown_corr_VT_{window}'] = dd_corr.astype('float32')
    
    # ---- Rolling Beta to BNDW (Global Bonds) ----
    if 'bndw' in macro_ref_returns:
        bndw_ret = macro_ref_returns['bndw'].reindex(idx, method='ffill')
        
        for window in windows:
            cov_bndw = etf_returns.rolling(window, min_periods=window//2).cov(bndw_ret)
            var_bndw = bndw_ret.rolling(window, min_periods=window//2).var()
            beta_bndw = cov_bndw / (var_bndw + 1e-10)
            feats[f'beta_BNDW_{window}'] = beta_bndw.astype('float32')
    
    # ---- Rolling Correlation with VIX ----
    if 'vix' in macro_ref_returns:
        vix_ret = macro_ref_returns['vix'].reindex(idx, method='ffill')
        
        for window in windows:
            corr_vix = etf_returns.rolling(window, min_periods=window//2).corr(vix_ret)
            feats[f'corr_VIX_{window}'] = corr_vix.astype('float32')
    
    # ---- Rolling Correlation with MOVE ----
    if 'move' in macro_ref_returns:
        move_ret = macro_ref_returns['move'].reindex(idx, method='ffill')
        
        for window in windows:
            corr_move = etf_returns.rolling(window, min_periods=window//2).corr(move_ret)
            feats[f'corr_MOVE_{window}'] = corr_move.astype('float32')
    
    return feats


def _process_ticker_risk_features(
    ticker: str,
    ticker_df: pd.DataFrame,
    macro_ref_returns: Dict[str, pd.Series],
    windows: List[int],
) -> Tuple[str, Dict[str, np.ndarray], pd.Index]:
    """
    Helper function to compute risk features for a single ticker (for parallel execution).
    
    Handles both:
    - DataFrame with 'Date' column (reset_index style)
    - DataFrame with Date as index (from MultiIndex .xs() extraction)
    
    Returns tuple of (ticker, {feat_name: values_array}, date_index).
    """
    # Handle both formats: Date as column or Date as index
    if 'Date' in ticker_df.columns:
        ticker_data = ticker_df.set_index('Date').sort_index()
    else:
        # Date is already the index (from MultiIndex .xs() extraction)
        ticker_data = ticker_df.sort_index()
    
    if 'Close' not in ticker_data.columns:
        return ticker, {}, ticker_data.index
        
    ticker_returns = ticker_data['Close'].pct_change()
    risk_feats = compute_risk_factor_features(ticker_returns, macro_ref_returns, windows)
    
    # Convert to numpy arrays for efficient assignment
    feat_arrays = {}
    for feat_name, feat_series in risk_feats.items():
        aligned_values = feat_series.reindex(ticker_data.index)
        feat_arrays[feat_name] = aligned_values.values
    
    return ticker, feat_arrays, ticker_data.index


def add_risk_factor_features_to_panel(
    panel_df: pd.DataFrame,
    macro_ref_returns: Dict[str, pd.Series],
    windows: List[int] = [21, 63, 126],
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Add risk/factor exposure features to the panel dataframe at multiple timeframes.
    
    Computes rolling betas, idiosyncratic vol, downside beta for each ticker
    at 21/63/126 day windows. Uses parallel processing for efficiency.
    
    NOTE: Synthetic BNDW is created from BND + BNDX data in the panel (if available).
    This gives us data from 2013 vs actual BNDW's 2018 start date.
    Synthetic BNDW = (BND returns + BNDX returns) / 2, with 98.45% correlation to actual.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with Date, Ticker, Close columns
    macro_ref_returns : Dict[str, pd.Series]
        Dictionary of macro reference returns from data_manager.load_macro_reference_returns()
        NOTE: Does NOT include 'bndw' - synthetic BNDW is created here from BND+BNDX
    windows : List[int]
        Rolling windows for computations (default: [21, 63, 126])
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
        
    Returns
    -------
    pd.DataFrame
        Panel with added risk factor features
    """
    print(f"[risk] Computing risk factor features (windows={windows}, parallel)...")
    
    # Handle both MultiIndex (Date, Ticker) and flat index with Ticker column
    is_multiindex = isinstance(panel_df.index, pd.MultiIndex) and 'Ticker' in panel_df.index.names
    
    if is_multiindex:
        tickers = panel_df.index.get_level_values('Ticker').unique()
    else:
        tickers = panel_df['Ticker'].unique()
    n_tickers = len(tickers)
    
    # === Create synthetic BNDW from BND + BNDX if both are in the panel ===
    if 'bndw' not in macro_ref_returns:
        if 'BND' in tickers and 'BNDX' in tickers:
            print("[risk] Creating synthetic BNDW from BND + BNDX in panel...")
            try:
                # Extract BND and BNDX data from panel
                if is_multiindex:
                    bnd_data = panel_df.xs('BND', level='Ticker')
                    bndx_data = panel_df.xs('BNDX', level='Ticker')
                else:
                    bnd_data = panel_df[panel_df['Ticker'] == 'BND'].set_index('Date')
                    bndx_data = panel_df[panel_df['Ticker'] == 'BNDX'].set_index('Date')
                
                # Compute returns
                bnd_ret = bnd_data['Close'].pct_change()
                bndx_ret = bndx_data['Close'].pct_change()
                
                # Find common dates - use intersection for safe alignment
                common_idx = bnd_ret.index.intersection(bndx_ret.index)
                
                # Synthetic BNDW = average of BND and BNDX returns
                synthetic_bndw_ret = ((bnd_ret.loc[common_idx] + bndx_ret.loc[common_idx]) / 2).dropna()
                
                # Add to macro_ref_returns
                macro_ref_returns = macro_ref_returns.copy()  # Don't mutate the input
                macro_ref_returns['bndw'] = synthetic_bndw_ret
                
                print(f"[risk] Created synthetic BNDW: {len(synthetic_bndw_ret)} returns from {synthetic_bndw_ret.index.min().date()} to {synthetic_bndw_ret.index.max().date()}")
            except Exception as e:
                print(f"[risk] WARNING: Could not create synthetic BNDW from BND+BNDX: {e}")
                print("[risk] beta_BNDW features will not be computed")
        else:
            missing = []
            if 'BND' not in tickers:
                missing.append('BND')
            if 'BNDX' not in tickers:
                missing.append('BNDX')
            print(f"[risk] WARNING: Cannot create synthetic BNDW - missing {missing} in panel")
            print("[risk] beta_BNDW features will not be computed")
    
    # Generate all risk feature names for the specified windows
    risk_feature_names = []
    for w in windows:
        risk_feature_names.extend([
            f'beta_VT_{w}', f'downside_beta_VT_{w}', f'idio_vol_{w}', 
            f'r_squared_VT_{w}', f'drawdown_corr_VT_{w}',
            f'beta_BNDW_{w}', f'corr_VIX_{w}', f'corr_MOVE_{w}'
        ])
    
    # Initialize new columns
    for feat in risk_feature_names:
        panel_df[feat] = np.nan
    
    # Group data by ticker for efficient parallel processing
    if is_multiindex:
        ticker_groups = {ticker: panel_df.xs(ticker, level='Ticker').copy() 
                         for ticker in tickers}
    else:
        ticker_groups = {ticker: panel_df[panel_df['Ticker'] == ticker].copy() 
                         for ticker in tickers}
    
    # Parallel processing with joblib
    print(f"[risk] Processing {n_tickers} tickers in parallel...")
    results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
        delayed(_process_ticker_risk_features)(
            ticker, ticker_groups[ticker], macro_ref_returns, windows
        )
        for ticker in tickers
    )
    
    # Assign results back to panel
    for ticker, feat_arrays, date_index in results:
        if not feat_arrays:
            continue
        if is_multiindex:
            # For MultiIndex, use pd.IndexSlice
            mask = panel_df.index.get_level_values('Ticker') == ticker
        else:
            mask = panel_df['Ticker'] == ticker
        for feat_name, values in feat_arrays.items():
            panel_df.loc[mask, feat_name] = values
    
    # Convert to float32
    for feat in risk_feature_names:
        if feat in panel_df.columns:
            panel_df[feat] = panel_df[feat].astype('float32')
    
    print(f"[risk] Added {len(risk_feature_names)} risk factor features ({len(windows)} windows)")
    
    return panel_df


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
        feats.update(bollinger_dict(close_pct_clipped, [21, 63]))  # Standardized trading-month multiples
        feats.update(momentum_dict(close_pct_clipped, [5, 10, 21, 42, 63]))
        
        # NEW from crosssecmom: Max drawdown features
        feats.update(drawdown_features(close, [21, 63]))  # Standardized trading-month multiples
        
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
        
        # =====================================================================
        # V2 FEATURE FAMILIES: Volume, Liquidity, Structure
        # =====================================================================
        # These add orthogonal signals beyond momentum/volatility
        
        # Volume features (from OHLCV)
        if 'Volume' in data.columns:
            vol = data['Volume'].astype('float32')
            high = data['High'].astype('float32') if 'High' in data.columns else None
            low = data['Low'].astype('float32') if 'Low' in data.columns else None
            feats.update(compute_volume_features(close, vol, high, low))
        
        # Liquidity / microstructure features
        if 'Volume' in data.columns:
            vol = data['Volume'].astype('float32')
            high = data['High'].astype('float32') if 'High' in data.columns else None
            low = data['Low'].astype('float32') if 'Low' in data.columns else None
            open_ = data['Open'].astype('float32') if 'Open' in data.columns else None
            feats.update(compute_liquidity_features(close, vol, high, low, open_))
        
        # Structure / regime features
        returns_decimal = close.pct_change()  # decimal returns for structure
        volatility_21 = returns_decimal.rolling(21, min_periods=10).std()
        feats.update(compute_structure_features(close, returns_decimal, volatility_21))
        
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
    # 
    # BUG FIX (2025-12-01): The old formula used pct_change(horizon).shift(-horizon)
    # but the shift(-horizon) was applied to the ENTIRE MultiIndex, not per-ticker.
    # This caused forward returns to be mixed up between tickers.
    #
    # NEW FORMULA: Direct calculation of Close[t+horizon] / Close[t] - 1
    # This is clearer and avoids the shift() issue with MultiIndex DataFrames.
    #
    # Determine if panel is MultiIndex or has Ticker column
    if isinstance(panel_df.index, pd.MultiIndex) and 'Ticker' in panel_df.index.names:
        # MultiIndex case: group by level
        future_close = panel_df['Close'].groupby(level='Ticker').shift(-horizon)
        fwd_ret = (future_close / panel_df['Close']) - 1
    elif 'Ticker' in panel_df.columns:
        # Column case: group by column
        future_close = panel_df.groupby('Ticker')['Close'].shift(-horizon)
        fwd_ret = (future_close / panel_df['Close']) - 1
    else:
        # Single ticker case: no grouping needed
        future_close = panel_df['Close'].shift(-horizon)
        fwd_ret = (future_close / panel_df['Close']) - 1
    
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
    - credit_proxy_21: Credit spread proxy (HYG - LQD 21-day returns)
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
    
    # 4. Credit spread proxy (HYG - LQD 21-day returns)
    # Need to get HYG and LQD returns from panel if available
    if 'HYG' in panel_df['Ticker'].values and 'LQD' in panel_df['Ticker'].values:
        # Get 21-day returns for HYG and LQD
        hyg_data = panel_df[panel_df['Ticker'] == 'HYG'].set_index('Date')
        lqd_data = panel_df[panel_df['Ticker'] == 'LQD'].set_index('Date')
        
        if 'Close%-21' in hyg_data.columns and 'Close%-21' in lqd_data.columns:
            hyg_ret = hyg_data['Close%-21'].reindex(date_index, method='ffill')
            lqd_ret = lqd_data['Close%-21'].reindex(date_index, method='ffill')
            macro_features['credit_proxy_21'] = hyg_ret - lqd_ret
        else:
            macro_features['credit_proxy_21'] = 0.0
    else:
        print("  [warning] HYG/LQD not in universe, setting credit_proxy_21 to 0")
        macro_features['credit_proxy_21'] = 0.0
    
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

def add_relative_return_features(panel_df: pd.DataFrame, lookbacks=[5, 21, 63]) -> pd.DataFrame:
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

def add_correlation_features(panel_df: pd.DataFrame, window=21) -> pd.DataFrame:
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
    
    Uses FACTOR_GROUPS classification to ensure ALL features are properly grouped.
    FULLY VECTORIZED using numpy broadcasting for maximum performance.
    
    Interaction Types:
    1. Multiplicative (cross-family): All C(k,2) family pairs = 7914 interactions
    2. Ratio: momentum/volatility (Sharpe-like)
    3. Difference: momentum acceleration, vol changes
    4. Polynomial: squared, cubed for key features
    5. Regime-Conditional: feature × regime_flag
    
    Args:
        panel_df: Panel data with base features
        config: Research configuration
        
    Returns:
        Panel data with base + interaction features
    """
    import time
    t0 = time.time()
    
    print("[interaction] Starting EXHAUSTIVE interaction feature generation...")
    print("[interaction] Using VECTORIZED numpy operations for speed...")
    print(f"[interaction] Base feature count: {len(panel_df.columns)}")
    
    # ==========================================================================
    # USE FACTOR_GROUPS CLASSIFICATION (not hardcoded patterns)
    # ==========================================================================
    exclude_cols = {'Close', 'FwdRet_21', 'Date', 'Ticker', 'ADV_63_Rank'}
    all_cols = [c for c in panel_df.columns if c not in exclude_cols]
    
    # Get family features using the canonical FACTOR_GROUPS classification
    family_features = get_family_features(all_cols)
    
    # Remove 'interaction' and 'other' families (we're generating interactions, not using existing ones)
    family_features.pop('interaction', None)
    family_features.pop('other', None)
    
    # Report family sizes
    print("\n[interaction] Feature families (from FACTOR_GROUPS):")
    total_base = 0
    for fam in sorted(family_features.keys()):
        feats = family_features[fam]
        if feats:
            print(f"  {fam:15s}: {len(feats):3d} features")
            total_base += len(feats)
    print(f"  {'TOTAL':15s}: {total_base:3d} base features")
    
    # Calculate expected cross-family interactions
    family_sizes = [len(f) for f in family_features.values() if f]
    expected_cross = (sum(family_sizes)**2 - sum(s**2 for s in family_sizes)) // 2
    print(f"\n[interaction] Expected cross-family multiplicative interactions: {expected_cross}")
    
    interaction_count = 0
    
    # ==========================================================================
    # TYPE 1: EXHAUSTIVE CROSS-FAMILY MULTIPLICATIVE INTERACTIONS (VECTORIZED)
    # ==========================================================================
    # Generate ALL family×family combinations: C(k,2) pairs
    # Uses numpy broadcasting for massive speedup
    
    print("\n[interaction] Type 1: EXHAUSTIVE Cross-Family Multiplicative (VECTORIZED)...")
    print("[interaction] Generating ALL unique family pairs (i < j to avoid duplicates)...")
    
    families = sorted([f for f in family_features.keys() if family_features[f]])
    
    # Pre-allocate list for batch DataFrame creation
    new_columns_data = {}
    
    for i, fam1 in enumerate(families):
        for fam2 in families[i+1:]:  # Only pairs where fam1 < fam2 (avoid duplicates)
            feats1 = family_features[fam1]
            feats2 = family_features[fam2]
            
            if not feats1 or not feats2:
                continue
            
            n_combinations = len(feats1) * len(feats2)
            t1 = time.time()
            
            # VECTORIZED: Extract matrices and compute outer product
            # Shape: (n_rows, n_feats1) @ (n_rows, n_feats2).T won't work
            # Instead: use broadcasting with einsum or direct multiplication
            
            mat1 = panel_df[feats1].values.astype(np.float32)  # (n_rows, n_feats1)
            mat2 = panel_df[feats2].values.astype(np.float32)  # (n_rows, n_feats2)
            
            # Generate all pairwise products using broadcasting
            # mat1[:, :, None] * mat2[:, None, :] -> (n_rows, n_feats1, n_feats2)
            # Then reshape to (n_rows, n_feats1 * n_feats2)
            products = mat1[:, :, np.newaxis] * mat2[:, np.newaxis, :]  # (n_rows, n_feats1, n_feats2)
            products = products.reshape(len(panel_df), -1)  # (n_rows, n_feats1 * n_feats2)
            
            # Generate column names
            col_names = [f"{f1}_x_{f2}" for f1 in feats1 for f2 in feats2]
            
            # Add to batch dict
            for idx, col_name in enumerate(col_names):
                new_columns_data[col_name] = products[:, idx]
            
            interaction_count += n_combinations
            elapsed = time.time() - t1
            print(f"  [{fam1}×{fam2}] {len(feats1)}×{len(feats2)} = {n_combinations} ({elapsed:.2f}s)")
    
    # BATCH ADD all Type 1 columns at once (much faster than one-by-one)
    print(f"\n[interaction] Batch adding {len(new_columns_data)} multiplicative columns...")
    t1 = time.time()
    new_df = pd.DataFrame(new_columns_data, index=panel_df.index)
    panel_df = pd.concat([panel_df, new_df], axis=1)
    print(f"  ✓ Batch add completed in {time.time() - t1:.2f}s")
    
    print(f"[interaction] Cross-family multiplicative total: {interaction_count}")
    
    # ==========================================================================
    # TYPE 2: RATIO INTERACTIONS (Division) - VECTORIZED
    # ==========================================================================
    # Sharpe-like ratios: momentum / volatility
    print("\n[interaction] Type 2: Ratio interactions (momentum / volatility) - VECTORIZED...")
    
    momentum_feats = family_features.get('momentum', [])
    volatility_feats = family_features.get('volatility', [])
    
    if momentum_feats and volatility_feats:
        t1 = time.time()
        
        mat_mom = panel_df[momentum_feats].values.astype(np.float32)
        mat_vol = panel_df[volatility_feats].values.astype(np.float32)
        
        # Replace zeros with nan in volatility
        mat_vol = np.where(mat_vol == 0, np.nan, mat_vol)
        
        # Vectorized division: (n_rows, n_mom, 1) / (n_rows, 1, n_vol)
        ratios = mat_mom[:, :, np.newaxis] / mat_vol[:, np.newaxis, :]
        ratios = ratios.reshape(len(panel_df), -1)
        
        col_names = [f"{f1}_div_{f2}" for f1 in momentum_feats for f2 in volatility_feats]
        
        ratio_df = pd.DataFrame(ratios, index=panel_df.index, columns=col_names, dtype='float32')
        panel_df = pd.concat([panel_df, ratio_df], axis=1)
        
        n_ratios = len(col_names)
        interaction_count += n_ratios
        print(f"  ✓ Generated {n_ratios} momentum/volatility ratios in {time.time() - t1:.2f}s")
    
    # ==========================================================================
    # TYPE 3: DIFFERENCE INTERACTIONS (Subtraction) - Already fast (few features)
    # ==========================================================================
    print("\n[interaction] Type 3: Difference interactions (acceleration)...")
    
    # 3.1 Momentum Acceleration (adjacent horizons)
    mom_hierarchy = ['Close%-1', 'Close%-2', 'Close%-3', 'Close%-5', 'Close%-10',
                     'Close%-21', 'Close%-42', 'Close%-63', 'Close%-126', 'Close%-252']
    mom_available = [m for m in mom_hierarchy if m in momentum_feats]
    
    diff_data = {}
    for i in range(len(mom_available) - 1):
        feat1, feat2 = mom_available[i], mom_available[i + 1]
        diff_data[f"{feat1}_minus_{feat2}"] = (panel_df[feat1].values - panel_df[feat2].values).astype(np.float32)
        interaction_count += 1
    
    if diff_data:
        panel_df = pd.concat([panel_df, pd.DataFrame(diff_data, index=panel_df.index)], axis=1)
    print(f"  ✓ Generated {len(diff_data)} momentum acceleration features")
    
    # 3.2 Volatility Changes (adjacent horizons)
    vol_hierarchy = ['Close_std5', 'Close_std10', 'Close_std21', 'Close_std42', 'Close_std63', 'Close_std126']
    vol_available = [v for v in vol_hierarchy if v in volatility_feats]
    
    diff_data = {}
    for i in range(len(vol_available) - 1):
        feat1, feat2 = vol_available[i], vol_available[i + 1]
        diff_data[f"{feat1}_minus_{feat2}"] = (panel_df[feat1].values - panel_df[feat2].values).astype(np.float32)
        interaction_count += 1
    
    if diff_data:
        panel_df = pd.concat([panel_df, pd.DataFrame(diff_data, index=panel_df.index)], axis=1)
    print(f"  ✓ Generated {len(diff_data)} volatility change features")
    
    # ==========================================================================
    # TYPE 4: POLYNOMIAL TRANSFORMATIONS - VECTORIZED
    # ==========================================================================
    print("\n[interaction] Type 4: Polynomial (squared, cubed) - VECTORIZED...")
    
    # 4.1 Momentum Squared and Cubed
    if momentum_feats:
        t1 = time.time()
        mat_mom = panel_df[momentum_feats].values.astype(np.float32)
        
        sq_data = {f"{f}_sq": mat_mom[:, i] ** 2 for i, f in enumerate(momentum_feats)}
        cb_data = {f"{f}_cb": mat_mom[:, i] ** 3 for i, f in enumerate(momentum_feats)}
        
        poly_df = pd.DataFrame({**sq_data, **cb_data}, index=panel_df.index)
        panel_df = pd.concat([panel_df, poly_df], axis=1)
        
        n_poly = len(momentum_feats) * 2
        interaction_count += n_poly
        print(f"  ✓ Generated {n_poly} momentum polynomial features in {time.time() - t1:.2f}s")
    
    # 4.2 Volatility Squared
    if volatility_feats:
        t1 = time.time()
        mat_vol = panel_df[volatility_feats].values.astype(np.float32)
        
        sq_data = {f"{f}_sq": mat_vol[:, i] ** 2 for i, f in enumerate(volatility_feats)}
        
        panel_df = pd.concat([panel_df, pd.DataFrame(sq_data, index=panel_df.index)], axis=1)
        
        interaction_count += len(volatility_feats)
        print(f"  ✓ Generated {len(volatility_feats)} volatility squared features in {time.time() - t1:.2f}s")
    
    # ==========================================================================
    # TYPE 5: REGIME-CONDITIONAL FEATURES - VECTORIZED
    # ==========================================================================
    # structure family contains regime flags
    structure_feats = family_features.get('structure', [])
    regime_flags = [f for f in structure_feats if any(x in f for x in ['crash_flag', 'meltup_flag', 'high_vol', 'low_vol'])]
    
    if regime_flags:
        print("\n[interaction] Type 5: Regime-conditional (feature × regime_flag) - VECTORIZED...")
        
        mat_flags = panel_df[regime_flags].values.astype(np.float32)  # (n_rows, n_flags)
        
        # Momentum × Regime
        if momentum_feats:
            t1 = time.time()
            mat_mom = panel_df[momentum_feats].values.astype(np.float32)
            
            # (n_rows, n_mom, 1) * (n_rows, 1, n_flags) -> (n_rows, n_mom, n_flags)
            regime_prod = mat_mom[:, :, np.newaxis] * mat_flags[:, np.newaxis, :]
            regime_prod = regime_prod.reshape(len(panel_df), -1)
            
            col_names = [f"{f1}_in_{f2}" for f1 in momentum_feats for f2 in regime_flags]
            regime_df = pd.DataFrame(regime_prod, index=panel_df.index, columns=col_names, dtype='float32')
            panel_df = pd.concat([panel_df, regime_df], axis=1)
            
            interaction_count += len(col_names)
            print(f"  ✓ Generated {len(col_names)} momentum×regime interactions in {time.time() - t1:.2f}s")
        
        # Volatility × Regime
        if volatility_feats:
            t1 = time.time()
            mat_vol = panel_df[volatility_feats].values.astype(np.float32)
            
            regime_prod = mat_vol[:, :, np.newaxis] * mat_flags[:, np.newaxis, :]
            regime_prod = regime_prod.reshape(len(panel_df), -1)
            
            col_names = [f"{f1}_in_{f2}" for f1 in volatility_feats for f2 in regime_flags]
            regime_df = pd.DataFrame(regime_prod, index=panel_df.index, columns=col_names, dtype='float32')
            panel_df = pd.concat([panel_df, regime_df], axis=1)
            
            interaction_count += len(col_names)
            print(f"  ✓ Generated {len(col_names)} volatility×regime interactions in {time.time() - t1:.2f}s")
    
    print(f"\n[interaction] Total interaction generation time: {time.time() - t0:.2f}s")
    
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    print("\n[interaction] Validation checks...")
    
    # Check for Inf values
    inf_cols = []
    for col in panel_df.columns:
        if col not in exclude_cols:
            if np.isinf(panel_df[col]).any():
                inf_cols.append(col)
    
    if inf_cols:
        print(f"  [WARNING] Found {len(inf_cols)} features with Inf values")
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
        print(f"  [WARNING] Found {len(high_nan_cols)} features with > 50% NaN")
    
    # Final summary
    final_feature_count = len(panel_df.columns)
    print(f"\n[interaction] ✓ EXHAUSTIVE interaction generation complete!")
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
    # 3.0.1 Download FRED data for enhanced macro features
    # -------------------------------------------------------------------------
    print("\n[3.0.1/10] Downloading FRED data...")
    fred_start = time.time()
    fred_data = data_manager.load_or_download_fred_data(
        start_date=download_start_str,
        end_date=config.time.end_date
    )
    fred_elapsed = time.time() - fred_start
    print(f"[time] FRED data download completed in {fred_elapsed:.2f} seconds")
    
    # Merge FRED data into macro_data dict for feature computation
    if fred_data:
        macro_data.update(fred_data)
        print(f"[macro] Combined yfinance + FRED data: {len(macro_data)} series")
    
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
    panel_df = add_relative_return_features(panel_df, lookbacks=[5, 21, 63])
    rel_ret_elapsed = time.time() - rel_ret_start
    print(f"[time] Relative return features added in {rel_ret_elapsed:.2f} seconds")
    
    print("\n[7.1/10] Adding correlation features...")
    corr_start = time.time()
    panel_df = add_correlation_features(panel_df, window=21)
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
    # 7.4. Add risk factor features (rolling betas to VT, BNDW, VIX, MOVE)
    # -------------------------------------------------------------------------
    print("\n[7.4/10] Adding risk factor features...")
    risk_feat_start = time.time()
    
    # Load macro reference returns (VT, BNDW, VIX, MOVE)
    macro_ref_returns = data_manager.load_macro_reference_returns(
        start_date=download_start_str,
        end_date=config.time.end_date
    )
    
    if macro_ref_returns:
        panel_df = add_risk_factor_features_to_panel(
            panel_df, macro_ref_returns, windows=[21, 63, 126], n_jobs=config.compute.n_jobs
        )
    else:
        print("[risk] WARNING: No macro reference returns available, skipping risk features")
    
    risk_feat_elapsed = time.time() - risk_feat_start
    print(f"[time] Risk factor features added in {risk_feat_elapsed:.2f} seconds")
    
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
    
    # -------------------------------------------------------------------------
    # 9.3 Compute target labels (y_raw, y_cs, y_resid, y_resid_z)
    # -------------------------------------------------------------------------
    print("\n[9.3/10] Computing target labels (cross-sectional demeaning, risk adjustment)...")
    target_start = time.time()
    from label_engineering import compute_targets
    
    # Ensure Date column is available for compute_targets
    if 'Date' not in panel_df.columns:
        panel_df = panel_df.reset_index()
    
    # Rename date column for label_engineering compatibility
    if 'Date' in panel_df.columns and 'date' not in panel_df.columns:
        panel_df = panel_df.rename(columns={'Date': 'date'})
    
    # Compute all target variants
    raw_return_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    panel_df = compute_targets(panel_df, config, raw_return_col=raw_return_col)
    
    # Restore Date column name for consistency
    if 'date' in panel_df.columns:
        panel_df = panel_df.rename(columns={'date': 'Date'})
    
    target_elapsed = time.time() - target_start
    print(f"[time] Target labels computed in {target_elapsed:.2f} seconds")
    print(f"[target] Active target column: {config.target.target_column}")

    # Set final index (handle case where Date/Ticker may already be in index)
    if isinstance(panel_df.index, pd.MultiIndex) and 'Date' in panel_df.index.names and 'Ticker' in panel_df.index.names:
        # Already indexed - just sort
        panel_df = panel_df.sort_index()
    elif 'Date' in panel_df.columns and 'Ticker' in panel_df.columns:
        # Columns exist - set as index
        panel_df = panel_df.set_index(['Date', 'Ticker']).sort_index()
    else:
        # Mixed state - reset and re-index
        panel_df = panel_df.reset_index(drop=False)
        if 'Date' in panel_df.columns and 'Ticker' in panel_df.columns:
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
    
    # Save panel data with error handling and chunked writing
    try:
        import gc
        gc.collect()  # Free memory before saving
        
        # Use row_group_size to write in chunks and reduce memory peak
        panel_df.to_parquet(
            config.paths.panel_parquet, 
            engine='pyarrow', 
            compression='snappy',
            row_group_size=50000  # Write in chunks of 50k rows
        )
        print(f"[save] Features saved to: {config.paths.panel_parquet}")
        print(f"[save] File size: {os.path.getsize(config.paths.panel_parquet) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"[ERROR] Failed to save parquet: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: try saving without compression
        print("[save] Retrying without compression...")
        try:
            panel_df.to_parquet(
                config.paths.panel_parquet, 
                engine='pyarrow', 
                compression=None,
                row_group_size=25000
            )
            print(f"[save] Features saved (uncompressed) to: {config.paths.panel_parquet}")
            print(f"[save] File size: {os.path.getsize(config.paths.panel_parquet) / 1024 / 1024:.1f} MB")
        except Exception as e2:
            print(f"[CRITICAL] Save failed completely: {e2}")
            traceback.print_exc()
            raise
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total features: {len([c for c in panel_df.columns if c not in ['Close']])}")
    print(f"Forward return horizon: {config.time.HOLDING_PERIOD_DAYS} days")
    print(f"Active target column: {config.target.target_column}")
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
    display_cols = [
        'Close', 'Close%-21', 'Close%-63', 'Close_Mom21', 'Close_RSI14', 'ADV_63',
        f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}',
        config.target.target_column_raw,
        config.target.target_column,
    ]
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
