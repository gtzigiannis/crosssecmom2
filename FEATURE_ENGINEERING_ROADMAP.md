# Feature Engineering Roadmap: Orthogonal Signal Taxonomy

## Executive Summary

**Problem:** Current 1324 features are 95% momentum variants → BIC correctly collapses to 1 feature.

**Solution:** Create orthogonal signal sources across 8-11 factor families.

### Implementation Status

| Component | File | Status |
|-----------|------|--------|
| V2 Feature Functions | `feature_engineering_v2.py` | ✅ Created |
| FRED Data Manager | `fred_data_manager.py` | ✅ Created |
| Factor Grouping for LARS | `feature_selection.py` | ⏳ Pending |
| Integration with Pipeline | `walk_forward_engine.py` | ⏳ Pending |

### New Features Implemented (V2)

| Family | # Features | Key Signals |
|--------|------------|-------------|
| **Volume** | ~15 | rel_volume, volume_zscore, obv_slope, up_down_vol_ratio |
| **Liquidity** | ~20 | amihud, roll_spread, cs_spread, parkinson_vol, garman_klass_vol |
| **Risk/Beta** | ~25 | beta_mkt, idio_vol, semi_vol, max_dd, var, cvar, down_corr |
| **Structure** | ~15 | vol_regime, trend_regime, days_since_flip, price_zscore |
| **Cross-Asset** | ~10 | rel_strength_spy, corr_tlt, corr_gld |
| **Macro (FRED)** | ~25 | yc_slope, hy_spread, fci, real_rate, epu, sentiment |

### Next Steps (Priority Order)

1. **Get FRED API Key** - Free from https://fred.stlouisfed.org/docs/api/api_key.html
2. **Run**: `pip install fredapi`
3. **Set env**: `set FRED_API_KEY=your_key`
4. **Integrate V2 features** into `feature_engineering.py`
5. **Add factor grouping** to LARS feature selection
6. **Test with profiling script**

---

## Overview

This document defines the complete feature taxonomy for crosssecmom2, organized by:
1. **Factor Family** (economic signal type)
2. **Horizon Bucket** (temporal scale)

The goal is to create **orthogonal signal sources** that don't all collapse to momentum.

---

## Discussion: Factor Families and Grouping Strategy

### Q: Are 8 factor families too many?

**Proposed families:** `{momentum, volatility, trend, volume, macro, liquidity, structure, interaction}`

**Answer:** No, 8 families is appropriate. Here's why:

1. **Economically distinct signals** - Each family captures a different economic phenomenon:
   - **Momentum**: Price persistence / trend following
   - **Volatility**: Risk levels / uncertainty
   - **Trend**: Direction and strength of price movement
   - **Volume**: Activity and conviction
   - **Macro**: Systematic/market-wide factors
   - **Liquidity**: Trading friction / crowding
   - **Structure/Shape**: Regime and pattern recognition
   - **Interaction**: Cross-family signals

2. **Academic support** - Factor investing research (Fama-French, AQR) typically uses 5-7 core factors. Adding liquidity, structure, and interactions is a reasonable extension.

3. **Target granularity** - With ~8 families and ~3-5 representatives per family, we get ~24-40 truly orthogonal features, which is ideal for LARS feature selection.

### Recommended Family Taxonomy (Expanded)

| # | Family | Economic Signal | Horizon Sensitivity |
|---|--------|-----------------|---------------------|
| 1 | **Momentum** | Price persistence | Short/Medium |
| 2 | **Volatility** | Risk/uncertainty | Short/Medium/Long |
| 3 | **Volume** | Activity/conviction | Short/Medium |
| 4 | **Liquidity** | Trading friction | Medium/Long |
| 5 | **Trend** | Direction strength | Medium/Long |
| 6 | **Risk/Beta** | Systematic exposure | Medium/Long |
| 7 | **Macro/Regime** | Market-wide factors | All horizons |
| 8 | **Sentiment** | Psychology/positioning | Short/Medium |
| 9 | **Structure** | Patterns/regimes | All horizons |
| 10 | **Cross-Asset** | Relative strength | Medium/Long |
| 11 | **Fundamental** | Valuation (limited) | Long |

**Note:** We can collapse some families if correlation analysis shows they measure the same thing:
- Trend + Momentum → might merge if corr > 0.85
- Volume + Liquidity → might merge
- Structure + Sentiment → might merge

### Horizon Buckets

| Bucket | Windows | Use Case |
|--------|---------|----------|
| **Short** | 5, 10, 14, 21 days | Tactical signals, mean reversion |
| **Medium** | 42, 63, 126 days | Trend following, momentum |
| **Long** | 189, 252 days | Strategic allocation, regime |

---

## Factor Family Taxonomy

### Current State (1324 features)

| Family | Count | % of Total | Notes |
|--------|-------|------------|-------|
| Momentum | 709 | 54% | Oversaturated |
| Interactions | 842 | 64% | Mostly momentum × something |
| Volatility | 131 | 10% | Good coverage |
| Macro | 93 | 7% | VIX, rates, credit - needs expansion |
| Volume | 23 | 2% | **Severely underdeveloped** |
| Trend | 6 | <1% | **Severely underdeveloped** |
| Liquidity | ~0 | 0% | **Missing entirely** |
| Risk/Beta | ~0 | 0% | **Missing entirely** |
| Sentiment | ~0 | 0% | **Missing entirely** |
| Fundamental | ~0 | 0% | **Missing entirely** |

### Target State

| Family | Target Features | Priority | Data Source |
|--------|-----------------|----------|-------------|
| **Momentum** | 50-100 | LOW | Existing (prune redundancy) |
| **Volatility** | 50-80 | MED | Existing + range estimators |
| **Volume** | 40-60 | HIGH | OHLCV-derived |
| **Liquidity** | 30-50 | HIGH | OHLCV-derived |
| **Trend** | 30-50 | MED | Existing transforms |
| **Risk/Beta** | 40-60 | HIGH | Rolling regressions |
| **Macro/Regime** | 60-100 | HIGH | FRED expansion |
| **Sentiment** | 20-40 | HIGH | FRED sentiment indices |
| **Cross-Asset** | 30-50 | MED | Relative strength |
| **Structure/Shape** | 30-50 | MED | Regime flags, persistence |
| **Fundamental** | 20-40 | LOW | FRED (limited for ETFs) |
| **Interactions** | 100-200 | LOW | Cross-family only |

---

## Horizon Buckets

All features should be computed at multiple horizons:

| Bucket | Windows | Use Case |
|--------|---------|----------|
| **Short** | 5, 10, 14, 21 days | Tactical signals, mean reversion |
| **Medium** | 42, 63, 126 days | Trend following, momentum |
| **Long** | 189, 252 days | Strategic allocation, regime |

---

## Complete Feature Specification by Family

### 1. MOMENTUM (Existing - Prune to Best Representatives)

Keep only the best variants, remove 90% of redundant transformations.

**Base signals to keep:**
- `Close%-{h}` for h ∈ {5, 10, 21, 42, 63, 126, 252}
- `Close_RSI{h}` for h ∈ {14, 21, 42}
- `Close_MACD`, `Close_MACD_Signal`
- `Close_WilliamsR{h}` for h ∈ {14, 21, 63}

**Remove:** Most interaction terms, redundant normalizations

---

### 2. VOLATILITY (Existing + Enhancements)

#### 2.1 Existing (Keep)
- `Close_std{h}` - Rolling standard deviation
- `Close_ATR{h}` - Average True Range
- `Close_BBW{h}` - Bollinger Band Width
- `Close_skew{h}`, `Close_kurt{h}` - Higher moments

#### 2.2 New: Range-Based Estimators
More efficient volatility estimates using OHLC structure:

```python
# Parkinson (1980) - uses High/Low only
parkinson_vol = np.sqrt((1/(4*np.log(2))) * (np.log(High/Low))**2)

# Garman-Klass (1980) - uses OHLC
gk_vol = np.sqrt(0.5*(np.log(High/Low))**2 - (2*np.log(2)-1)*(np.log(Close/Open))**2)

# Rogers-Satchell (1991) - drift-independent
rs_vol = np.sqrt(np.log(High/Close)*np.log(High/Open) + np.log(Low/Close)*np.log(Low/Open))

# Yang-Zhang (2000) - combines overnight and intraday
# (more complex, see implementation)
```

#### 2.3 New: Volatility Regime Features
```python
# Vol regime classification
vol_percentile = rolling_rank(realized_vol, 252)  # 0-1
vol_regime = pd.cut(vol_percentile, bins=[0, 0.33, 0.67, 1], labels=['low', 'med', 'high'])

# Vol of vol
vol_of_vol = realized_vol.rolling(21).std()

# Vol term structure
vol_ratio_short_long = std_21 / std_126
```

---

### 3. VOLUME (Major Expansion Needed)

#### 3.1 Basic Volume Features
```python
# Relative volume
rel_volume_20 = Volume / Volume.rolling(20).mean()
rel_volume_60 = Volume / Volume.rolling(60).mean()

# Volume z-score
volume_zscore_20 = (Volume - Volume.rolling(20).mean()) / Volume.rolling(20).std()
volume_zscore_60 = (Volume - Volume.rolling(60).mean()) / Volume.rolling(60).std()

# Volume trend
volume_sma_ratio = Volume.rolling(10).mean() / Volume.rolling(50).mean()
```

#### 3.2 Price-Volume Relationship
```python
# Volume-weighted price indicators
vwap_deviation = Close / VWAP - 1  # Deviation from VWAP

# OBV (On-Balance Volume)
obv = (np.sign(Close.diff()) * Volume).cumsum()
obv_slope = obv.diff(21) / 21

# Price-volume correlation
pv_corr_21 = Close.pct_change().rolling(21).corr(Volume.pct_change())

# Volume on up vs down days
up_volume_ratio = volume_on_up_days.rolling(21).sum() / total_volume.rolling(21).sum()
```

#### 3.3 Volume Anomalies
```python
# Volume breakout
volume_breakout = Volume > Volume.rolling(60).quantile(0.9)

# Dry-up indicator (unusually low volume)
volume_dryup = Volume < Volume.rolling(60).quantile(0.1)

# Volume / ATR (normalized activity)
volume_per_atr = Volume / ATR_14
```

---

### 4. LIQUIDITY / MICROSTRUCTURE (New Family)

#### 4.1 Amihud Illiquidity
```python
# Classic Amihud (2002)
amihud = np.abs(returns) / (Volume * Close)  # |r| / dollar_volume
amihud_21 = amihud.rolling(21).mean()
amihud_63 = amihud.rolling(63).mean()

# Log version for stability
log_amihud = np.log1p(amihud * 1e6)
```

#### 4.2 Bid-Ask Spread Proxies
```python
# Roll (1984) implied spread
# Spread ≈ 2 * sqrt(-cov(r_t, r_{t-1})) if cov < 0
roll_spread = 2 * np.sqrt(np.maximum(-returns.rolling(21).cov(returns.shift(1)), 0))

# Corwin-Schultz (2012) high-low spread estimator
# Uses 2-day high/low to estimate spread
beta = (np.log(High/Low))**2
gamma = (np.log(High.rolling(2).max()/Low.rolling(2).min()))**2
alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / (3 - 2*np.sqrt(2)) - np.sqrt(gamma/(3 - 2*np.sqrt(2)))
cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
```

#### 4.3 Trading Friction
```python
# Kyle's lambda proxy (price impact)
kyle_lambda = np.abs(returns) / np.sqrt(Volume)

# Turnover
turnover = Volume / shares_outstanding  # if available, else use ADV proxy

# Relative spread to volatility
spread_to_vol = cs_spread / realized_vol
```

---

### 5. TREND (Expand from 6 features)

#### 5.1 Trend Strength
```python
# ADX (Average Directional Index) - already may exist
adx_14 = ta.ADX(High, Low, Close, 14)
adx_28 = ta.ADX(High, Low, Close, 28)

# Trend strength via R² of price regression
def trend_r2(prices, window):
    """R² of linear regression = trend strength"""
    y = prices.values
    x = np.arange(len(y))
    slope, intercept, r, p, se = stats.linregress(x, y)
    return r**2

trend_r2_21 = Close.rolling(21).apply(lambda x: trend_r2(x, 21))
trend_r2_63 = Close.rolling(63).apply(lambda x: trend_r2(x, 63))
```

#### 5.2 Trend Direction
```python
# Price vs moving averages
price_vs_sma50 = Close / Close.rolling(50).mean() - 1
price_vs_sma200 = Close / Close.rolling(200).mean() - 1

# Golden/Death cross signals
sma_50_200_ratio = Close.rolling(50).mean() / Close.rolling(200).mean()

# Slope of moving average
sma_slope_50 = Close.rolling(50).mean().diff(10) / 10
```

#### 5.3 Trend Regime
```python
# Trend regime classification
def classify_trend(mom_21, mom_63, vol_21):
    """Classify trend regime"""
    if mom_21 > vol_21 and mom_63 > 0:
        return 'strong_up'
    elif mom_21 < -vol_21 and mom_63 < 0:
        return 'strong_down'
    else:
        return 'chop'

# Numeric encoding
trend_regime_score = np.where(trend == 'strong_up', 1, 
                              np.where(trend == 'strong_down', -1, 0))
```

---

### 6. RISK / BETA / CORRELATION (New Family)

#### 6.1 Market Beta
```python
# Rolling beta to SPY
def rolling_beta(asset_returns, market_returns, window):
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var

beta_spy_63 = rolling_beta(returns, spy_returns, 63)
beta_spy_252 = rolling_beta(returns, spy_returns, 252)

# Beta stability (std of rolling beta)
beta_stability = beta_spy_63.rolling(126).std()
```

#### 6.2 Idiosyncratic Risk
```python
# Residual volatility from market regression
def idio_vol(asset_returns, market_returns, window):
    beta = rolling_beta(asset_returns, market_returns, window)
    residuals = asset_returns - beta * market_returns
    return residuals.rolling(window).std()

idio_vol_63 = idio_vol(returns, spy_returns, 63)
idio_vol_252 = idio_vol(returns, spy_returns, 252)

# Idio vol ratio (idio / total)
idio_ratio = idio_vol_63 / returns.rolling(63).std()
```

#### 6.3 Downside Risk
```python
# Semi-volatility (downside only)
def semi_volatility(returns, window):
    neg_returns = returns.where(returns < 0, 0)
    return neg_returns.rolling(window).std()

semi_vol_63 = semi_volatility(returns, 63)

# Sortino-style ratio
sortino_component = returns.rolling(63).mean() / semi_vol_63

# Max drawdown
def rolling_max_drawdown(prices, window):
    rolling_max = prices.rolling(window).max()
    drawdown = prices / rolling_max - 1
    return drawdown.rolling(window).min()

max_dd_63 = rolling_max_drawdown(Close, 63)
max_dd_252 = rolling_max_drawdown(Close, 252)

# Time since max drawdown
days_since_max_dd = (Close == Close.rolling(252).max()).astype(int).groupby((Close != Close.rolling(252).max()).cumsum()).cumcount()
```

#### 6.4 Tail Risk
```python
# VaR (Value at Risk)
var_5pct_63 = returns.rolling(63).quantile(0.05)
var_1pct_63 = returns.rolling(63).quantile(0.01)

# Expected Shortfall (CVaR)
def expected_shortfall(returns, window, alpha=0.05):
    var = returns.rolling(window).quantile(alpha)
    return returns.where(returns <= var).rolling(window).mean()

cvar_5pct_63 = expected_shortfall(returns, 63, 0.05)

# Skewness of returns
return_skew_63 = returns.rolling(63).skew()
return_kurt_63 = returns.rolling(63).kurt()
```

#### 6.5 Correlation Structure
```python
# Rolling correlation with market
corr_spy_63 = returns.rolling(63).corr(spy_returns)
corr_spy_252 = returns.rolling(252).corr(spy_returns)

# Downside correlation (correlation on down days only)
def downside_correlation(asset_ret, market_ret, window):
    down_days = market_ret < 0
    return asset_ret.where(down_days).rolling(window).corr(market_ret.where(down_days))

down_corr_spy_63 = downside_correlation(returns, spy_returns, 63)

# Correlation asymmetry (down_corr - up_corr)
up_corr = returns.where(spy_returns > 0).rolling(63).corr(spy_returns.where(spy_returns > 0))
corr_asymmetry = down_corr_spy_63 - up_corr
```

#### 6.6 Cross-Sectional Positioning
```python
# Cross-sectional rank/z-score of risk metrics
beta_xsrank = beta_spy_63.groupby('Date').rank(pct=True)
vol_xsrank = realized_vol.groupby('Date').rank(pct=True)
dd_xsrank = max_dd_63.groupby('Date').rank(pct=True)
```

---

### 7. MACRO / REGIME (FRED Expansion)

#### 7.1 Existing Macro Features
- VIX level, VIX z-score
- Credit spreads
- Yield curve slope
- Short rate

#### 7.2 New FRED Series to Add

**Interest Rates & Yield Curve:**
```python
FRED_RATES = {
    'DFF': 'fed_funds_rate',           # Fed Funds Rate
    'DTB3': 'tbill_3m',                # 3-Month T-Bill
    'DGS2': 'tsy_2y',                  # 2-Year Treasury
    'DGS5': 'tsy_5y',                  # 5-Year Treasury
    'DGS10': 'tsy_10y',                # 10-Year Treasury
    'DGS30': 'tsy_30y',                # 30-Year Treasury
    'T10Y2Y': 'yc_slope_10_2',         # 10Y-2Y Spread
    'T10Y3M': 'yc_slope_10_3m',        # 10Y-3M Spread
}

# Derived features
yc_curvature = tsy_2y + tsy_30y - 2 * tsy_10y
yc_slope_momentum = yc_slope_10_2.diff(21)
real_rate = tsy_10y - breakeven_inflation
```

**Credit Spreads:**
```python
FRED_CREDIT = {
    'BAMLC0A0CM': 'ig_spread',         # IG Corporate Spread
    'BAMLH0A0HYM2': 'hy_spread',       # HY Corporate Spread
    'TEDRATE': 'ted_spread',           # TED Spread (LIBOR - T-Bill)
    'AAA10Y': 'aaa_spread',            # AAA - 10Y Spread
    'BAA10Y': 'baa_spread',            # BAA - 10Y Spread
}

# Derived features
credit_spread_momentum = hy_spread.diff(21)
credit_spread_zscore = (hy_spread - hy_spread.rolling(252).mean()) / hy_spread.rolling(252).std()
ig_hy_spread = hy_spread - ig_spread  # Quality spread
```

**Economic Activity:**
```python
FRED_ECONOMY = {
    'ICSA': 'initial_claims',          # Initial Jobless Claims
    'UNRATE': 'unemployment',          # Unemployment Rate
    'PAYEMS': 'nonfarm_payrolls',      # Nonfarm Payrolls
    'INDPRO': 'industrial_prod',       # Industrial Production
    'RSXFS': 'retail_sales',           # Retail Sales
    'HOUST': 'housing_starts',         # Housing Starts
    'PERMIT': 'building_permits',      # Building Permits
}

# Derived features
claims_4wk_ma = initial_claims.rolling(4).mean()
claims_momentum = claims_4wk_ma.pct_change(4)  # YoY change proxy
```

**Inflation:**
```python
FRED_INFLATION = {
    'CPIAUCSL': 'cpi',                 # CPI All Urban
    'CPILFESL': 'core_cpi',            # Core CPI
    'T5YIE': 'breakeven_5y',           # 5Y Breakeven Inflation
    'T10YIE': 'breakeven_10y',         # 10Y Breakeven Inflation
    'PCEPI': 'pce',                    # PCE Price Index
}

# Derived features
inflation_surprise = cpi.pct_change(12) - breakeven_10y.shift(252)
real_rate_10y = tsy_10y - breakeven_10y
```

**Financial Conditions:**
```python
FRED_FIN_CONDITIONS = {
    'NFCI': 'chicago_fci',             # Chicago Fed National Financial Conditions
    'STLFSI4': 'stl_fsi',              # St. Louis Fed Financial Stress Index
    'GSFCI': 'gs_fci',                 # Goldman Sachs FCI (if available)
}
```

#### 7.3 Regime Indicators from Macro
```python
# Macro regime classification
def classify_macro_regime(yc_slope, credit_spread, vix):
    """
    Risk-On: Steep curve, tight spreads, low VIX
    Risk-Off: Flat/inverted curve, wide spreads, high VIX
    """
    score = 0
    score += np.where(yc_slope > 0.5, 1, np.where(yc_slope < 0, -1, 0))
    score += np.where(credit_spread < credit_spread.rolling(252).quantile(0.3), 1,
                      np.where(credit_spread > credit_spread.rolling(252).quantile(0.7), -1, 0))
    score += np.where(vix < 20, 1, np.where(vix > 30, -1, 0))
    return score  # -3 to +3

macro_regime_score = classify_macro_regime(yc_slope_10_2, hy_spread, vix)
```

---

### 8. SENTIMENT (New Family)

#### 8.1 News Sentiment
```python
FRED_SENTIMENT = {
    'DNFSFFNQ': 'sf_news_sentiment',   # SF Fed Daily News Sentiment (Quarterly)
    # Note: Daily version may need direct download from SF Fed
}

# Transform to daily (forward fill quarterly)
news_sentiment_daily = sf_news_sentiment.resample('D').ffill()
news_sentiment_ma = news_sentiment_daily.rolling(21).mean()
news_sentiment_momentum = news_sentiment_daily.diff(21)
```

#### 8.2 Consumer Sentiment
```python
FRED_CONSUMER = {
    'UMCSENT': 'umich_sentiment',      # U of Michigan Consumer Sentiment
    'CSCICP03USM665S': 'oecd_cci',     # OECD Consumer Confidence
}

# Derived features
sentiment_surprise = umich_sentiment - umich_sentiment.shift(1)
sentiment_zscore = (umich_sentiment - umich_sentiment.rolling(60).mean()) / umich_sentiment.rolling(60).std()
sentiment_momentum = umich_sentiment.pct_change(3)  # 3-month momentum
```

#### 8.3 Economic Policy Uncertainty
```python
# Baker-Bloom-Davis Indices (available on policyuncertainty.com, some on FRED)
FRED_UNCERTAINTY = {
    'USEPUINDXD': 'epu_daily',         # Economic Policy Uncertainty (Daily)
    'WLEMUINDXD': 'emv_daily',         # Equity Market Volatility Tracker (Daily)
}

# Derived features
epu_zscore = (epu_daily - epu_daily.rolling(252).mean()) / epu_daily.rolling(252).std()
epu_spike = epu_daily > epu_daily.rolling(252).quantile(0.9)
emv_vix_ratio = emv_daily / vix  # News-implied vs option-implied vol
```

#### 8.4 Positioning / Flow Sentiment
```python
# VIX Term Structure (sentiment proxy)
vix_term_structure = vix_3m / vix  # >1 = contango (complacent), <1 = backwardation (fear)

# Put/Call Ratio (need separate data source - CBOE)
# pc_ratio_equity = put_volume / call_volume
# pc_ratio_zscore = (pc_ratio - pc_ratio.rolling(63).mean()) / pc_ratio.rolling(63).std()

# AAII Sentiment (weekly, needs separate source)
# aaii_bull_bear_spread = aaii_bullish - aaii_bearish
```

---

### 9. STRUCTURE / SHAPE (New Family)

#### 9.1 Volatility Regime
```python
# Vol regime classification
def vol_regime(realized_vol, window=252):
    pct = realized_vol.rolling(window).rank(pct=True)
    return pd.cut(pct, bins=[0, 0.33, 0.67, 1], labels=[0, 1, 2])  # low/med/high

vol_regime_252 = vol_regime(std_21, 252)

# Time in current vol regime
vol_regime_duration = vol_regime_252.groupby((vol_regime_252 != vol_regime_252.shift()).cumsum()).cumcount()
```

#### 9.2 Trend Regime
```python
# Trend regime based on momentum sign and magnitude
def trend_regime(mom_21, mom_63, vol_21):
    """
    +2: Strong uptrend (mom > vol, aligned)
    +1: Weak uptrend
     0: Chop / range
    -1: Weak downtrend
    -2: Strong downtrend
    """
    strength = np.abs(mom_21) / vol_21  # Momentum vs vol
    aligned = np.sign(mom_21) == np.sign(mom_63)  # Short and medium aligned
    
    score = np.sign(mom_21)  # Base direction
    score = np.where(strength > 1 & aligned, score * 2, score)  # Strong if > 1 vol and aligned
    score = np.where(strength < 0.5, 0, score)  # Chop if < 0.5 vol
    return score

trend_regime_score = trend_regime(mom_21, mom_63, vol_21)
```

#### 9.3 Persistence / Streak Features
```python
# Days since last sign flip
def days_since_sign_flip(series):
    sign = np.sign(series)
    flip = sign != sign.shift()
    return flip.groupby(flip.cumsum()).cumcount()

mom_streak_length = days_since_sign_flip(returns.rolling(21).mean())

# Current up/down streak
def current_streak(returns):
    """Count consecutive up or down days"""
    sign = np.sign(returns)
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    return streak * sign  # Positive for up streak, negative for down

price_streak = current_streak(returns)
```

#### 9.4 Mean Reversion Indicators
```python
# Distance from rolling mean (z-score)
price_zscore_50 = (Close - Close.rolling(50).mean()) / Close.rolling(50).std()
price_zscore_200 = (Close - Close.rolling(200).mean()) / Close.rolling(200).std()

# RSI extremes (overbought/oversold)
rsi_extreme = np.where(rsi_14 > 70, 1, np.where(rsi_14 < 30, -1, 0))

# Bollinger Band position
bb_position = (Close - bb_lower) / (bb_upper - bb_lower)  # 0-1, >1 = above upper, <0 = below lower
```

---

### 10. CROSS-ASSET (Expand)

#### 10.1 Relative Strength vs Benchmarks
```python
# Relative strength vs SPY
rel_strength_spy_21 = returns.rolling(21).sum() - spy_returns.rolling(21).sum()
rel_strength_spy_63 = returns.rolling(63).sum() - spy_returns.rolling(63).sum()

# Relative strength vs sector
rel_strength_sector = returns.rolling(21).sum() - sector_returns.rolling(21).sum()

# Rank within asset class
rel_strength_rank = rel_strength_spy_63.groupby('Date').rank(pct=True)
```

#### 10.2 Cross-Asset Correlations
```python
# Rolling correlation with other asset classes
corr_with_bonds = returns.rolling(63).corr(bond_returns)  # TLT or AGG
corr_with_gold = returns.rolling(63).corr(gold_returns)   # GLD
corr_with_dollar = returns.rolling(63).corr(usd_returns)  # UUP or DXY

# Correlation regime
equity_bond_corr = spy_returns.rolling(63).corr(bond_returns)
# Negative = flight to quality, Positive = risk-on/off together
```

#### 10.3 Lead-Lag Relationships
```python
# Does credit lead equity?
credit_equity_lead = hy_spread.shift(5).rolling(21).corr(spy_returns)

# Does VIX lead ETF?
vix_etf_lead = vix.shift(5).rolling(21).corr(returns)
```

---

### 11. FUNDAMENTAL (Limited for ETFs)

#### 11.1 Available from FRED
```python
# Shiller CAPE (for SPY/equity ETFs)
FRED_FUNDAMENTAL = {
    'CAPE': 'shiller_cape',            # Shiller CAPE Ratio (monthly)
}

cape_zscore = (cape - cape.rolling(120).mean()) / cape.rolling(120).std()
cape_vs_avg = cape / cape.rolling(120).mean()
```

#### 11.2 Earnings-Related (Proxy)
```python
# S&P 500 earnings (for equity ETFs)
# Can use SPY price / CAPE as proxy for earnings
implied_earnings = spy_price / cape

# Earnings momentum (proxy)
earnings_momentum = implied_earnings.pct_change(12)
```

#### 11.3 Valuation Spreads
```python
# Value vs Growth spread (using ETF prices)
value_growth_spread = value_etf_return - growth_etf_return  # VTV - VUG
small_large_spread = small_etf_return - large_etf_return    # VB - VV
```

---

### 12. INTERACTIONS (Selective Cross-Family)

Only create interactions between **different families**, not within family.

```python
# Momentum × Volatility
mom_vol_interaction = mom_21 * vol_zscore_63

# Momentum × Regime
mom_regime_interaction = mom_21 * macro_regime_score

# Volume × Momentum (confirmation)
volume_confirms_trend = (rel_volume_20 > 1.5) & (np.abs(mom_21) > vol_21)

# Sentiment × Momentum (contrarian or confirmation)
sentiment_mom_interaction = news_sentiment_daily * mom_21
```

---

## Implementation Priority

### Phase 1: Quick Wins (Use Existing Data)
1. **Liquidity features** (Amihud, spread proxies) - OHLCV only
2. **Risk/Beta features** - Rolling regressions with SPY
3. **Structure/Shape features** - Transform existing momentum
4. **Volume expansion** - OHLCV only
5. **Range-based volatility** - Parkinson, Garman-Klass

### Phase 2: FRED Expansion
1. **Macro regime** - Yield curve, credit spreads
2. **Sentiment indices** - EPU, consumer sentiment
3. **Financial conditions** - NFCI, stress indices

### Phase 3: Cross-Asset & Advanced
1. **Cross-asset correlations** - Requires bond/gold/FX data
2. **Fundamental proxies** - CAPE, earnings
3. **Selective interactions** - Cross-family only

---

## Factor Grouping Configuration

```python
FACTOR_GROUPS = {
    'momentum': {
        'patterns': ['Close%', 'RSI', 'Williams', 'MACD', 'ROC', 'MOM'],
        'target_representatives': 5,  # Keep best 5
    },
    'volatility': {
        'patterns': ['std', 'vol', 'ATR', 'BBW', 'parkinson', 'gk_vol', 'skew', 'kurt'],
        'target_representatives': 5,
    },
    'volume': {
        'patterns': ['Volume', 'rel_vol', 'obv', 'vwap', 'volume_z'],
        'target_representatives': 4,
    },
    'liquidity': {
        'patterns': ['amihud', 'spread', 'kyle', 'illiq'],
        'target_representatives': 3,
    },
    'trend': {
        'patterns': ['sma', 'ema', 'adx', 'trend_r2', 'slope'],
        'target_representatives': 3,
    },
    'risk_beta': {
        'patterns': ['beta', 'idio', 'corr_spy', 'semi_vol', 'drawdown', 'var', 'cvar'],
        'target_representatives': 5,
    },
    'macro': {
        'patterns': ['vix', 'yc_', 'spread', 'rate', 'fci', 'claims'],
        'target_representatives': 5,
    },
    'sentiment': {
        'patterns': ['sentiment', 'epu', 'emv', 'umich'],
        'target_representatives': 3,
    },
    'structure': {
        'patterns': ['regime', 'streak', 'zscore', 'extreme'],
        'target_representatives': 3,
    },
    'cross_asset': {
        'patterns': ['rel_strength', 'corr_with_', 'lead'],
        'target_representatives': 3,
    },
}

# Total target: ~40 orthogonal features from 10 families
```

---

## Summary

| Family | Current | Target | Priority | Data Needed |
|--------|---------|--------|----------|-------------|
| Momentum | 709 | 20-30 | PRUNE | Existing |
| Volatility | 131 | 40-50 | MED | Existing + formulas |
| Volume | 23 | 40-60 | HIGH | Existing OHLCV |
| Liquidity | 0 | 30-50 | HIGH | Existing OHLCV |
| Trend | 6 | 30-50 | MED | Existing |
| Risk/Beta | 0 | 40-60 | HIGH | Existing + SPY |
| Macro | 93 | 80-120 | HIGH | FRED expansion |
| Sentiment | 0 | 20-40 | HIGH | FRED |
| Structure | ~0 | 30-50 | MED | Existing transforms |
| Cross-Asset | Few | 30-50 | MED | Bond/Gold/FX |
| Fundamental | 0 | 10-20 | LOW | FRED CAPE |
| Interactions | 842 | 50-100 | LOW | Cross-family only |

**Total Target: 400-650 diverse features vs current 1324 redundant features**
