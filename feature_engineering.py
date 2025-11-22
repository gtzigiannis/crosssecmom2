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
from data_manager import download_etf_data, CrossSecMomDataManager  # Import from data_manager instead

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

def process_ticker(ticker: str, data: pd.DataFrame, adv_window: int = 63) -> pd.DataFrame:
    """
    Engineer features for a single ticker.
    
    CRITICAL FIX: Returns Close column in the features dict!
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    data : pd.DataFrame
        Raw OHLCV data with DatetimeIndex
    adv_window : int
        Window for ADV calculation (default: 63)
        
    Returns
    -------
    pd.DataFrame
        Features with DatetimeIndex and 'Ticker' column
    """
    try:
        close = data['Close'].astype('float32')
        close.name = 'Close'
        
        # CRITICAL: Include Close in output
        feats = {
            'Ticker': ticker,
            'Close': close,  # <-- FIX: Was missing!
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
        
        return df
        
    except Exception as e:
        import traceback
        print(f"[error] {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame(index=data.index)

# ============================================================================
# FORWARD RETURNS (TARGET VARIABLES)
# ============================================================================

def add_forward_returns(panel_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add forward returns at specified horizon.
    
    Uses closed-left windows: forward return from t to t+h uses Close[t] and Close[t+h].
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) index or columns
    horizon : int
        Forward horizon in days
        
    Returns
    -------
    pd.DataFrame
        Panel with added FwdRet_{h} column
    """
    print(f"[fwd] Computing forward returns for horizon: {horizon}")
    
    # Compute forward returns per ticker
    panel_df[f'FwdRet_{horizon}'] = (
        panel_df.groupby('Ticker')['Close']
        .pct_change(horizon)
        .shift(-horizon) * 100.0
    ).astype('float32')
    
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
    
    print("\n[2/8] Downloading OHLCV data...")
    print(f"[download] All ETFs will be downloaded from {config.time.start_date} to {config.time.end_date}")
    print(f"[download] Filtering will be applied during walk-forward backtest")
    
    download_start = time.time()
    data_dict = download_etf_data(
        tickers,
        config.time.start_date,
        config.time.end_date
    )
    download_elapsed = time.time() - download_start
    
    if len(data_dict) == 0:
        raise RuntimeError("No data downloaded!")
    
    print(f"[time] Data download completed in {download_elapsed:.2f} seconds ({download_elapsed/60:.2f} minutes)")
    
    # -------------------------------------------------------------------------
    # 2.5. Download macro data (NEW)
    # -------------------------------------------------------------------------
    print("\n[2.5/8] Downloading macro data...")
    macro_start = time.time()
    data_manager = CrossSecMomDataManager(config.paths.data_dir)
    macro_tickers = {
        'vix': '^VIX',         # VIX volatility index
        'yield_10y': '^TNX',   # 10-year Treasury yield
        'yield_2y': '^IRX',    # 2-year Treasury yield (proxy, actually 13-week)
        'tbill_3m': '^IRX',    # 3-month T-bill rate (same proxy)
    }
    
    macro_data = data_manager.load_or_download_macro_data(
        macro_tickers,
        start_date=config.time.start_date,
        end_date=config.time.end_date
    )
    macro_elapsed = time.time() - macro_start
    print(f"[time] Macro data download completed in {macro_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 3. Feature engineering (parallel)
    # -------------------------------------------------------------------------
    print("\n[3/8] Engineering features per ticker...")
    
    feature_eng_start = time.time()
    results = Parallel(n_jobs=config.compute.n_jobs, backend='threading', verbose=5)(
        delayed(process_ticker)(ticker, data_dict[ticker], config.universe.adv_window)
        for ticker in data_dict.keys()
    )
    feature_eng_elapsed = time.time() - feature_eng_start
    print(f"[time] Feature engineering completed in {feature_eng_elapsed:.2f} seconds ({feature_eng_elapsed/60:.2f} minutes)")
    
    # -------------------------------------------------------------------------
    # 4. Combine into panel structure
    # -------------------------------------------------------------------------
    print("\n[4/8] Building panel structure...")
    
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
    # 5. Add forward returns
    # -------------------------------------------------------------------------
    print("\n[5/8] Computing forward returns...")
    fwd_ret_start = time.time()
    panel_df = panel_df.reset_index()
    panel_df = add_forward_returns(panel_df, config.time.HOLDING_PERIOD_DAYS)
    fwd_ret_elapsed = time.time() - fwd_ret_start
    print(f"[time] Forward returns computed in {fwd_ret_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 6. Add cross-sectional features (NEW from crosssecmom)
    # -------------------------------------------------------------------------
    print("\n[6/8] Adding relative return features...")
    rel_ret_start = time.time()
    panel_df = add_relative_return_features(panel_df, lookbacks=[5, 20, 60])
    rel_ret_elapsed = time.time() - rel_ret_start
    print(f"[time] Relative return features added in {rel_ret_elapsed:.2f} seconds")
    
    print("\n[7/8] Adding correlation features...")
    corr_start = time.time()
    panel_df = add_correlation_features(panel_df, window=20)
    corr_elapsed = time.time() - corr_start
    print(f"[time] Correlation features added in {corr_elapsed:.2f} seconds")
    
    print("\n[7.5/8] Adding asset type flags...")
    asset_start = time.time()
    panel_df = add_asset_type_flags(panel_df, config)
    asset_elapsed = time.time() - asset_start
    print(f"[time] Asset type flags added in {asset_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 7.6. Add macro features (NEW)
    # -------------------------------------------------------------------------
    print("\n[7.6/8] Adding macro/regime features...")
    macro_feat_start = time.time()
    panel_df = add_macro_features(panel_df, macro_data, config)
    macro_feat_elapsed = time.time() - macro_feat_start
    print(f"[time] Macro features added in {macro_feat_elapsed:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 8. Add ADV rank for filtering
    # -------------------------------------------------------------------------
    print("\n[8/8] Adding ADV cross-sectional rank...")
    adv_col = f'ADV_{config.universe.adv_window}'
    panel_df = add_adv_rank(panel_df, adv_col)
    
    # Set final index
    panel_df = panel_df.set_index(['Date', 'Ticker']).sort_index()
    
    # -------------------------------------------------------------------------
    # 7. Save outputs
    # -------------------------------------------------------------------------
    print("\n[save] Saving outputs...")
    
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
