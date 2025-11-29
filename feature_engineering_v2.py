"""
Feature Engineering V2: Orthogonal Signal Taxonomy
==================================================

This module implements the expanded feature set with orthogonal signal families:

1. VOLUME - Activity and conviction signals from OHLCV
2. LIQUIDITY - Trading friction / microstructure proxies  
3. RISK/BETA - Systematic exposure and idiosyncratic risk
4. MACRO/REGIME - FRED expansion (sentiment, uncertainty, conditions)
5. STRUCTURE/SHAPE - Regime flags, persistence, streaks

Design Principles:
- Each family captures economically DISTINCT signals
- Features are grouped by family for factor grouping in LARS
- Horizons: short (5-21d), medium (42-126d), long (189-252d)
- All computations are backward-looking only (no lookahead bias)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. VOLUME FEATURES (from OHLCV only)
# ============================================================================

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
    for window in [10, 20, 60]:
        vol_ma = volume.rolling(window, min_periods=window//2).mean()
        feats[f'rel_volume_{window}'] = (volume / vol_ma).astype('float32')
    
    # ---- Volume Z-Score ----
    for window in [20, 60]:
        vol_mean = volume.rolling(window, min_periods=window//2).mean()
        vol_std = volume.rolling(window, min_periods=window//2).std()
        feats[f'volume_zscore_{window}'] = ((volume - vol_mean) / (vol_std + 1e-8)).astype('float32')
    
    # ---- Volume Trend (short/long ratio) ----
    vol_ma_10 = volume.rolling(10, min_periods=5).mean()
    vol_ma_50 = volume.rolling(50, min_periods=25).mean()
    feats['volume_trend_10_50'] = (vol_ma_10 / (vol_ma_50 + 1e-8)).astype('float32')
    
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
    vol_90pct = volume.rolling(60, min_periods=30).quantile(0.9)
    vol_10pct = volume.rolling(60, min_periods=30).quantile(0.1)
    feats['volume_breakout_60'] = (volume > vol_90pct).astype('float32')
    feats['volume_dryup_60'] = (volume < vol_10pct).astype('float32')
    
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


# ============================================================================
# 2. LIQUIDITY / MICROSTRUCTURE FEATURES
# ============================================================================

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


# ============================================================================
# 3. RISK / BETA / CORRELATION STRUCTURE
# ============================================================================

def compute_risk_beta_features(
    returns: pd.Series,
    market_returns: pd.Series,
    close: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Compute risk, beta, and correlation structure features.
    
    Features:
    - Rolling beta to market (SPY/VT)
    - Idiosyncratic volatility
    - Semi-volatility (downside only)
    - Max drawdown and time since max
    - VaR and CVaR
    - Downside correlation
    - Correlation asymmetry
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
# 4. STRUCTURE / SHAPE / REGIME FEATURES
# ============================================================================

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
# 5. CROSS-ASSET RELATIVE STRENGTH
# ============================================================================

def compute_cross_asset_features(
    returns: pd.Series,
    benchmark_returns: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """
    Compute cross-asset relative strength features.
    
    benchmark_returns should include:
    - 'spy': S&P 500 returns
    - 'tlt': Long-term treasury returns (optional)
    - 'gld': Gold returns (optional)
    - 'uup': US Dollar returns (optional)
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
        
        # ---- Relative Strength Rank (normalized) ----
        for window in [63]:
            rel_str = feats.get(f'rel_strength_{bm_name}_{window}')
            if rel_str is not None:
                rs_zscore = (rel_str - rel_str.rolling(252, min_periods=126).mean()) / (
                    rel_str.rolling(252, min_periods=126).std() + 1e-8
                )
                feats[f'rel_strength_{bm_name}_zscore'] = rs_zscore.astype('float32')
        
        # ---- Rolling Correlation ----
        for window in [63, 126]:
            corr = asset_ret.rolling(window, min_periods=window//2).corr(bm_ret)
            feats[f'corr_{bm_name}_{window}'] = corr.reindex(returns.index).astype('float32')
    
    return feats


# ============================================================================
# 6. MACRO / SENTIMENT FEATURES FROM FRED
# ============================================================================

# FRED series mapping for easy download
FRED_SERIES = {
    # Interest Rates & Yield Curve
    'DFF': 'fed_funds',
    'DTB3': 'tbill_3m', 
    'DGS2': 'tsy_2y',
    'DGS5': 'tsy_5y',
    'DGS10': 'tsy_10y',
    'DGS30': 'tsy_30y',
    'T10Y2Y': 'yc_slope_10_2',
    'T10Y3M': 'yc_slope_10_3m',
    
    # Credit Spreads
    'BAMLC0A0CM': 'ig_spread',
    'BAMLH0A0HYM2': 'hy_spread',
    'TEDRATE': 'ted_spread',
    'AAA10Y': 'aaa_spread',
    'BAA10Y': 'baa_spread',
    
    # Economic Activity
    'ICSA': 'initial_claims',
    'UNRATE': 'unemployment',
    'INDPRO': 'industrial_prod',
    
    # Inflation
    'T5YIE': 'breakeven_5y',
    'T10YIE': 'breakeven_10y',
    
    # Financial Conditions
    'NFCI': 'chicago_fci',
    'STLFSI4': 'stl_fsi',
    
    # Sentiment / Uncertainty
    'UMCSENT': 'umich_sentiment',
    'USEPUINDXD': 'epu_daily',
}


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
    
    return feats


# ============================================================================
# 7. COMPOSITE REGIME INDICATOR
# ============================================================================

def compute_regime_composite(
    macro_features: Dict[str, pd.Series],
    vix: pd.Series = None,
) -> Dict[str, pd.Series]:
    """
    Compute composite regime indicators combining multiple signals.
    
    Returns:
    - macro_regime_score: -3 to +3 (risk-off to risk-on)
    - regime components
    """
    feats = {}
    
    # Get features
    yc_slope = macro_features.get('yc_slope_10_2')
    hy_spread = macro_features.get('hy_spread')
    vix_z = None
    
    if vix is not None:
        vix_mean = vix.rolling(252, min_periods=126).mean()
        vix_std = vix.rolling(252, min_periods=126).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        feats['vix_zscore'] = vix_z.astype('float32')
    
    # Build composite score
    if yc_slope is not None and hy_spread is not None and vix_z is not None:
        # Yield curve component
        yc_score = np.where(yc_slope > 0.5, 1, np.where(yc_slope < 0, -1, 0))
        
        # Credit spread component
        cs_pct = hy_spread.rolling(252, min_periods=126).rank(pct=True)
        cs_score = np.where(cs_pct < 0.3, 1, np.where(cs_pct > 0.7, -1, 0))
        
        # VIX component
        vix_score = np.where(vix_z < -0.5, 1, np.where(vix_z > 1, -1, 0))
        
        # Composite (-3 to +3)
        composite = yc_score + cs_score + vix_score
        feats['macro_regime_score'] = pd.Series(composite, index=yc_slope.index).astype('float32')
        
        # Risk-on/off flags
        feats['risk_on'] = (composite >= 2).astype('float32')
        feats['risk_off'] = (composite <= -2).astype('float32')
    
    return feats


# ============================================================================
# 8. FACTOR GROUPING CONFIGURATION
# ============================================================================

FACTOR_GROUPS = {
    'momentum': {
        'patterns': ['Close%-', 'RSI', 'Williams', 'MACD', 'ROC', 'Mom', '_lag'],
        'target_representatives': 5,
    },
    'volatility': {
        'patterns': ['std', 'vol', 'ATR', 'BBW', 'parkinson', 'garman_klass', 'rogers_satchell', 'skew', 'kurt'],
        'target_representatives': 5,
    },
    'volume': {
        'patterns': ['volume', 'rel_vol', 'obv', 'vwap', 'up_down_vol'],
        'target_representatives': 4,
    },
    'liquidity': {
        'patterns': ['amihud', 'spread', 'kyle', 'illiq', 'roll_spread', 'cs_spread'],
        'target_representatives': 4,
    },
    'trend': {
        'patterns': ['MA', 'EMA', 'adx', 'trend_r2', 'slope', 'trend_strength', 'trend_regime'],
        'target_representatives': 4,
    },
    'risk_beta': {
        'patterns': ['beta', 'idio', 'corr_mkt', 'semi_vol', 'max_dd', 'var_', 'cvar', 'down_corr'],
        'target_representatives': 5,
    },
    'macro': {
        'patterns': ['vix', 'yc_', 'hy_spread', 'fci', 'fsi', 'claims', 'real_rate', 'credit'],
        'target_representatives': 5,
    },
    'sentiment': {
        'patterns': ['sentiment', 'epu', 'umich', 'uncertainty'],
        'target_representatives': 3,
    },
    'structure': {
        'patterns': ['regime', 'streak', 'zscore', 'days_since', 'alignment', 'pct_from'],
        'target_representatives': 4,
    },
    'cross_asset': {
        'patterns': ['rel_strength', 'corr_tlt', 'corr_gld', 'corr_uup'],
        'target_representatives': 3,
    },
}


def classify_feature_family(feature_name: str) -> str:
    """
    Classify a feature into its factor family based on name patterns.
    
    Returns family name or 'other' if no match.
    """
    feature_lower = feature_name.lower()
    
    for family, config in FACTOR_GROUPS.items():
        for pattern in config['patterns']:
            if pattern.lower() in feature_lower:
                return family
    
    return 'other'


def get_family_features(
    feature_columns: List[str]
) -> Dict[str, List[str]]:
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
# 9. MAIN INTERFACE: ENGINEER ALL V2 FEATURES
# ============================================================================

def engineer_v2_features_for_ticker(
    ohlcv: pd.DataFrame,
    market_returns: pd.Series = None,
    benchmark_returns: Dict[str, pd.Series] = None,
) -> pd.DataFrame:
    """
    Engineer all V2 features for a single ticker.
    
    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data with DatetimeIndex and columns [Open, High, Low, Close, Volume]
    market_returns : pd.Series
        Market benchmark returns (SPY) for beta calculation
    benchmark_returns : Dict[str, pd.Series]
        Dict of benchmark returns for cross-asset features
        
    Returns
    -------
    pd.DataFrame
        V2 features with DatetimeIndex
    """
    close = ohlcv['Close'].astype('float32')
    high = ohlcv.get('High', pd.Series(dtype='float32')).astype('float32') if 'High' in ohlcv else None
    low = ohlcv.get('Low', pd.Series(dtype='float32')).astype('float32') if 'Low' in ohlcv else None
    open_ = ohlcv.get('Open', pd.Series(dtype='float32')).astype('float32') if 'Open' in ohlcv else None
    volume = ohlcv.get('Volume', pd.Series(dtype='float32')).astype('float32') if 'Volume' in ohlcv else None
    
    returns = close.pct_change()
    volatility_21 = returns.rolling(21, min_periods=10).std()
    
    all_features = {}
    
    # 1. Volume features
    vol_feats = compute_volume_features(close, volume, high, low)
    all_features.update(vol_feats)
    
    # 2. Liquidity features
    liq_feats = compute_liquidity_features(close, volume, high, low, open_)
    all_features.update(liq_feats)
    
    # 3. Risk/Beta features (requires market returns)
    if market_returns is not None:
        risk_feats = compute_risk_beta_features(returns, market_returns, close)
        all_features.update(risk_feats)
    
    # 4. Structure/Shape features
    struct_feats = compute_structure_features(close, returns, volatility_21)
    all_features.update(struct_feats)
    
    # 5. Cross-asset features (requires benchmark returns)
    if benchmark_returns is not None:
        cross_feats = compute_cross_asset_features(returns, benchmark_returns)
        all_features.update(cross_feats)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features, index=ohlcv.index)
    
    # Replace inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def engineer_macro_features(
    date_index: pd.DatetimeIndex,
    fred_data: Dict[str, pd.Series],
    vix: pd.Series = None,
) -> pd.DataFrame:
    """
    Engineer macro/sentiment features that are common to all tickers.
    
    Returns DataFrame that can be broadcast to all tickers by date.
    """
    all_features = {}
    
    # FRED-based macro features
    macro_feats = compute_macro_features_from_fred(date_index, fred_data)
    all_features.update(macro_feats)
    
    # Composite regime indicator
    regime_feats = compute_regime_composite(macro_feats, vix)
    all_features.update(regime_feats)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features, index=date_index)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df
