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
import yfinance as yf
import time
import os
from datetime import datetime
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from config import ResearchConfig

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
    if volume is None or volume.isna().all():
        return {}
    
    # Dollar volume = Close * Volume
    dollar_volume = close * volume
    
    # ADV: rolling mean of dollar volume
    adv = dollar_volume.rolling(window).mean()
    
    return {
        f'ADV_{window}': adv,
    }

# ============================================================================
# PER-TICKER FEATURE ENGINEERING
# ============================================================================

def process_ticker(ticker: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for a single ticker.
    
    CRITICAL FIX: Returns Close column in the features dict!
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    data : pd.DataFrame
        Raw OHLCV data with DatetimeIndex
        
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
        
        # Core return features
        feats.update(pct_change_k(close, [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]))
        feats.update(lagged_returns(close_pct, [1, 2, 3, 5, 10]))
        feats.update(std_dict(close_pct, [5, 10, 21, 42, 63, 126]))
        feats.update(skew_dict(close_pct, [21, 42, 63, 126]))
        feats.update(kurt_dict(close_pct, [21, 42, 63, 126]))
        feats.update(bollinger_dict(close_pct, [21, 50]))
        feats.update(momentum_dict(close_pct, [5, 10, 21, 42, 63]))
        
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
            feats.update(adv_features(close, vol, window=63))
        
        # Hurst exponent on returns
        feats.update(hurst_multi(close_pct, [21, 63, 126]))
        
        # Convert to DataFrame
        df = pd.DataFrame(feats, index=data.index)
        return df.astype('float32')
        
    except Exception as e:
        print(f"[error] {ticker}: {e}")
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

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_etf_data(
    tickers: list,
    start_date: str,
    end_date: str,
    batch_sleep: float = 1.0
) -> dict:
    """
    Download OHLCV data for all tickers.
    
    Returns
    -------
    dict
        {ticker: DataFrame} with OHLCV data
    """
    print(f"[download] Fetching data for {len(tickers)} ETFs from {start_date} to {end_date}...")
    
    data_dict = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                failed.append(ticker)
                continue
            
            # Keep only OHLCV
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.astype('float32')
            data_dict[ticker] = df
            
            if (i + 1) % 20 == 0:
                print(f"[download] Progress: {i+1}/{len(tickers)}")
                time.sleep(batch_sleep)
                
        except Exception as e:
            print(f"[error] {ticker}: {e}")
            failed.append(ticker)
    
    print(f"[download] Successfully downloaded: {len(data_dict)}/{len(tickers)}")
    if failed:
        print(f"[download] Failed tickers: {failed}")
    
    return data_dict

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
    # 2. Download data
    # -------------------------------------------------------------------------
    print("\n[2/6] Downloading OHLCV data...")
    data_dict = download_etf_data(
        tickers,
        config.time.start_date,
        config.time.end_date
    )
    
    if len(data_dict) == 0:
        raise RuntimeError("No data downloaded!")
    
    # -------------------------------------------------------------------------
    # 3. Feature engineering (parallel)
    # -------------------------------------------------------------------------
    print("\n[3/6] Engineering features per ticker...")
    
    results = Parallel(n_jobs=config.compute.n_jobs, backend='threading', verbose=5)(
        delayed(process_ticker)(ticker, data_dict[ticker])
        for ticker in data_dict.keys()
    )
    
    # -------------------------------------------------------------------------
    # 4. Combine into panel structure
    # -------------------------------------------------------------------------
    print("\n[4/6] Building panel structure...")
    
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
    print("\n[5/6] Computing forward returns...")
    panel_df = panel_df.reset_index()
    panel_df = add_forward_returns(panel_df, config.time.HOLDING_PERIOD_DAYS)
    
    # -------------------------------------------------------------------------
    # 6. Add ADV rank for filtering
    # -------------------------------------------------------------------------
    print("\n[6/6] Adding ADV cross-sectional rank...")
    panel_df = add_adv_rank(panel_df, 'ADV_63')
    
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
    print(sample[['Close', 'Close%-21', 'Close%-63', 'Mom21', 'RSI14', 'ADV_63', f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}']].to_string())
    
    print("\n[done] Feature engineering complete!")
    return panel_df


if __name__ == "__main__":
    from config import get_default_config
    
    # Load config
    config = get_default_config()
    
    # Run pipeline
    panel_df = run_feature_engineering(config)
