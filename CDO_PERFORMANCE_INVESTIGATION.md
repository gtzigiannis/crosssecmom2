# Chief Development Officer: Performance Investigation Directive
## Cross-Sectional Momentum Strategy - Critical Performance Analysis

**Date**: November 22, 2025  
**Status**: 🔴 URGENT - Strategy Underperformance Requiring Comprehensive Investigation  
**Context**: Following complete accounting/leverage fix verification (FIX 0-5)  
**Performance**: -69.64% cumulative return, 38.92% win rate (2011-2025)

---

## Executive Summary

**AI Assistant Directive**: This is NOT an accounting bug. All accounting and leverage mechanics have been verified mathematically correct (error < machine precision). The strategy's poor performance stems from a **FUNDAMENTAL MODEL FAILURE**: short positions consistently appreciate instead of depreciate. This document provides an exhaustive investigation framework.

### Verified Facts (Post-Accounting Fixes)
- ✅ Accounting formulas: CORRECT (verified to machine precision)
- ✅ Capital compounding: CORRECT (error < 1e-10)
- ✅ Leverage mechanics: CORRECT (matches margin regime)
- ✅ Cost calculations: CORRECT (decimals throughout)
- ✅ Long positions: PROFITABLE (+0.57% mean return) ✓
- ❌ Short positions: **LOSING MONEY** (-0.81% mean return - shorts go UP!) ✗

### The Core Problem
**The momentum model consistently picks the WRONG stocks to short.** Stocks selected for shorting (predicted to decline) are actually APPRECIATING by 0.81% per period. This is a model prediction failure, not an execution or accounting issue.

---

## I. SCIENTIFIC HYPOTHESIS TREE

### Hypothesis 1: Momentum Reversal in ETF Universe
**Severity**: 🔴 CRITICAL  
**Likelihood**: HIGH

#### Thesis
Cross-sectional momentum may exhibit reversal patterns in ETFs due to:
1. **Mean reversion in sector rotations**: ETFs track sectors/themes that cycle
2. **Momentum decay horizon mismatch**: Short-term reversals vs medium-term trends
3. **Liquidity-driven reversals**: Large ETFs mean-revert faster than stocks
4. **Crowding effects**: Popular momentum signals become contrarian indicators

#### Investigation Steps
```python
# A. Test momentum persistence by decile
import pandas as pd
import numpy as np

# Load results
results = pd.read_csv('D:/REPOSITORY/Data/crosssecmom2/cs_momentum_results.csv')
panel = pd.read_parquet('D:/REPOSITORY/Data/crosssecmom2/cs_momentum_features.parquet')

# For each rebalance period:
# 1. Compute decile performance AFTER selection
# 2. Check if bottom decile (shorts) outperforms top decile (longs)
# 3. Test at 1-day, 5-day, 21-day, 63-day horizons

# B. Compute autocorrelation of returns
for ticker in tickers:
    returns = panel.loc[pd.IndexSlice[:, ticker], 'Close'].pct_change()
    acf_21 = returns.autocorr(21)  # Should be positive for momentum
    acf_63 = returns.autocorr(63)
    # If acf < 0 at short horizons → reversal present

# C. Test momentum vs reversal signals
momentum_21d = panel.groupby('Ticker')['Close'].pct_change(21)
reversal_5d = panel.groupby('Ticker')['Close'].pct_change(5)
# Check correlation: negative correlation suggests reversal dominates
```

#### Potential Fixes
1. **Flip the signal**: If systematic reversal, invert short selection
2. **Filter by momentum persistence**: Exclude ETFs with negative autocorrelation
3. **Adjust lookback windows**: Test 3M, 6M, 12M momentum instead of mixed features
4. **Sector-neutral shorts**: Short within sectors to avoid rotation effects

---

### Hypothesis 2: Feature Engineering Flaws
**Severity**: 🔴 CRITICAL  
**Likelihood**: MEDIUM-HIGH

#### Thesis A: Forward Returns Leakage
Despite FIX 0, there may be subtle look-ahead bias:
1. **Gap between training and scoring**: 21-day gap may be insufficient
2. **Feature lag alignment**: Some features may inadvertently peek forward
3. **ETF creation/redemption artifacts**: Intraday NAV adjustments cause spurious signals

#### Investigation Steps
```python
# A. Audit feature construction dates
for feature in feature_list:
    max_lag = get_max_lag_for_feature(feature)  # From feature definition
    actual_data_used = check_data_window(panel, feature, t0)
    assert actual_data_used < t0 - HOLDING_PERIOD, f"Leak in {feature}"

# B. Test with deliberately lagged features
# If performance improves with extra lag → look-ahead present
config.time.FEATURE_MAX_LAG_DAYS = 126  # Double the lag
run_backtest_with_lagged_features()

# C. Inspect FwdRet calculation
fwd_ret_calc = panel['FwdRet_21']
# Ensure it's: Close[t+21] / Close[t] - 1
# NOT: Close[t+21] / Close[t-21] - 1 (would create wrong signal)
```

#### Thesis B: Feature Collinearity Masking True Signal
The strategy uses 96 features. Many may be redundant/noisy:
1. **Multicollinearity**: Correlated features amplify noise over signal
2. **Curse of dimensionality**: 96 features on 80 ETFs = severe overfitting risk
3. **Supervised binning on noise**: Bins fit to random fluctuations

#### Investigation Steps
```python
# A. Feature correlation matrix
feature_corr = panel[feature_cols].corr()
high_corr_pairs = feature_corr[abs(feature_corr) > 0.9].stack()
# Drop highly correlated features (keep one per group)

# B. PCA analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
principal_features = pca.fit_transform(panel[feature_cols].fillna(0))
# Check explained variance: if 90%+ in 5 components → heavy redundancy

# C. Test with MINIMAL feature set
minimal_features = ['Close%-21', 'Close%-63', 'Close%-126']  # Pure momentum
run_backtest_with_features(minimal_features)
# If this outperforms → feature bloat is the problem

# D. L1 regularization on feature selection
# Instead of IC threshold, use Lasso to auto-select features
```

#### Thesis C: Feature Normalization Issues
Cross-sectional features may not be properly standardized:
1. **Outliers dominating**: Single extreme ETF skews entire distribution
2. **Time-varying vol**: Features not vol-adjusted across regimes
3. **Scale mismatch**: Some features in % (returns) vs absolute (ADV)

#### Investigation Steps
```python
# A. Check feature distributions
for feat in feature_cols:
    print(f"{feat}: mean={panel[feat].mean():.4f}, "
          f"std={panel[feat].std():.4f}, "
          f"skew={panel[feat].skew():.2f}, "
          f"kurtosis={panel[feat].kurtosis():.2f}")
    # If skew > 5 or kurtosis > 20 → heavy outliers

# B. Test winsorization
panel_winsorized = panel.copy()
for feat in feature_cols:
    panel_winsorized[feat] = winsorize(panel[feat], limits=[0.01, 0.01])
run_backtest_with_panel(panel_winsorized)

# C. Test rank-based features
panel_rank = panel.copy()
for feat in feature_cols:
    panel_rank[feat] = panel.groupby('Date')[feat].rank(pct=True)
run_backtest_with_panel(panel_rank)
# If this improves → outliers were the problem
```

---

### Hypothesis 3: Supervised Binning Overfitting
**Severity**: 🔴 CRITICAL  
**Likelihood**: HIGH

#### Thesis
The supervised binning approach may be fitting to noise:
1. **Small sample per window**: 5-year window × 80 ETFs = 1260 × 80 = ~100k obs, but ~400 obs per ETF
2. **Bins fitted on forward returns**: Optimizes on-sample, fails out-of-sample
3. **No regularization**: Bins can perfectly separate good/bad in-sample
4. **High turnover**: Bins change between windows → strategy chases noise

#### Investigation Steps
```python
# A. Check bin stability across windows
bins_per_window = {}  # Store bin edges for each feature, each window
for window_id, (t_start, t_end) in enumerate(training_windows):
    bins_per_window[window_id] = model.get_bin_edges()

# Compare bin edges window-to-window
from scipy.stats import kendalltau
bin_stability = []
for feat in features:
    edges_t0 = bins_per_window[0][feat]
    edges_t1 = bins_per_window[1][feat]
    tau, p = kendalltau(edges_t0, edges_t1)
    bin_stability.append(tau)
    # If tau < 0.3 → bins are unstable (overfitting)

# B. Test with FIXED bins (no refitting)
# Fit bins on ENTIRE history once
model_fixed = train_alpha_model(panel, metadata, 
                                start_date='2011-01-01',
                                end_date='2025-11-01',
                                config=config)
# Use these bins for ALL periods (no refitting)
run_backtest_with_fixed_model(model_fixed)
# If this improves → overfitting to training windows

# C. Test simple ranking (no bins)
def score_simple_rank(panel, t0, config):
    cross_section = panel.loc[t0]
    momentum = cross_section['Close%-21']  # Simple 21-day momentum
    return momentum.rank()  # Just rank, no binning
run_backtest_with_scorer(score_simple_rank)

# D. Check IC stability
ic_per_window = []  # IC from diagnostics
ic_mean = np.mean(ic_per_window)
ic_std = np.std(ic_per_window)
# If std / mean > 2 → IC is noisy (overfitting)
```

#### Potential Fixes
1. **Remove supervised binning entirely**: Use simple momentum ranks
2. **Increase regularization**: Limit bins to 3-5 quantiles (not 10)
3. **Cross-validation**: Use k-fold CV within training window
4. **Ensemble bins**: Average predictions from multiple bin configurations

---

### Hypothesis 4: Universe Selection Bias
**Severity**: 🟡 MODERATE  
**Likelihood**: MEDIUM

#### Thesis
The ETF universe may be poorly suited for momentum:
1. **Liquidity-driven inclusion**: High-ADV ETFs are institutional favorites → crowded trades
2. **Thematic ETFs dominate**: ARK-style theme ETFs reverse quickly
3. **Leveraged ETF contamination**: Despite filters, some 2x/3x ETFs may remain
4. **Sector concentration**: If 60% of universe is tech/growth → momentum becomes sector bet

#### Investigation Steps
```python
# A. Analyze universe composition
universe = pd.read_parquet('D:/REPOSITORY/Data/crosssecmom2/universe_metadata.parquet')
print(universe['category'].value_counts())  # Check sector distribution
print(universe.groupby('cluster_id').size())  # Check clustering

# High-ADV bias check
adv_quantiles = universe['ADV_63'].quantile([0.25, 0.5, 0.75])
# If top quartile captures 90% of backtest periods → liquidity bias

# B. Test subsets
# Universe A: Only broad market ETFs (SPY, QQQ, IWM, etc.)
# Universe B: Exclude top 10 by ADV (remove most liquid)
# Universe C: Sector-balanced (equal weight per sector)

# C. Check short-specific patterns
short_positions_by_ticker = []  # From backtest results
most_shorted = pd.Series(short_positions_by_ticker).value_counts()
# If same 10 tickers shorted 80% of time → concentration risk

# D. Correlation to SPY
for ticker in universe.index:
    corr_to_spy = panel.loc[pd.IndexSlice[:, ticker], 'Close'].corr(
        panel.loc[pd.IndexSlice[:, 'SPY'], 'Close']
    )
    if abs(corr_to_spy) > 0.95:  # Near-duplicate of SPY
        print(f"{ticker}: redundant (ρ={corr_to_spy:.3f})")
```

#### Potential Fixes
1. **Narrow universe**: Focus on 20-30 most liquid, uncorrelated ETFs
2. **Sector-neutral construction**: Long/short within each sector
3. **Exclude thematic ETFs**: Remove ARKK-style names
4. **Dynamic universe**: Only trade ETFs with stable momentum (autocorr > 0)

---

### Hypothesis 5: Transaction Cost Model Error
**Severity**: 🟡 MODERATE  
**Likelihood**: LOW (but check)

#### Thesis
Despite accounting fixes, costs may still be mis-specified:
1. **Slippage underestimated**: ETF spreads wider than assumed
2. **Borrow costs variable**: Hard-to-borrow ETFs have 10%+ rates, not 5.5%
3. **Turnover amplification**: High leverage magnifies turnover costs
4. **Market impact**: Large notional positions move prices

#### Investigation Steps
```python
# A. Compare to zero-cost baseline
results_no_costs = run_backtest(config, costs=False)
results_with_costs = run_backtest(config, costs=True)
cost_drag = results_with_costs['ls_return'].mean() - results_no_costs['ls_return'].mean()
# If drag < 0.5% per period → costs are NOT the problem

# B. Actual turnover analysis
turnover_per_period = results['turnover'].mean()
print(f"Mean turnover: {turnover_per_period:.2%}")
# Momentum strategies: 30-50% is normal
# If turnover > 80% → excessive churn

# C. Check leverage × turnover interaction
gross_lev = results['gross_long'].mean() + results['gross_short'].mean()
effective_turnover = turnover_per_period * gross_lev
# This is what drives costs
# If effective_turnover > 200% → problem

# D. Hard-to-borrow diagnosis
# Check if shorted ETFs are illiquid/niche
shorted_tickers = get_all_short_positions()
for ticker in shorted_tickers:
    volume = get_avg_volume(ticker)
    short_interest = get_short_interest(ticker)  # From external source
    if short_interest > 20%:  # High short interest
        print(f"{ticker}: likely expensive to borrow (SI={short_interest}%)")
```

#### Potential Fixes
1. **Increase cost assumptions**: Test 15 bps (not 10 bps)
2. **Skip hard-to-borrow ETFs**: Filter out high short interest names
3. **Lower leverage**: Reduce gross to 2x (not 3.6x) to reduce turnover costs
4. **Hold longer**: Switch to 63-day rebalance (not 21-day)

---

### Hypothesis 6: Regime Mismatch
**Severity**: 🟡 MODERATE  
**Likelihood**: MEDIUM

#### Thesis
Strategy may work in trending markets but fail in range-bound/volatile regimes:
1. **2011-2025 regime mix**: Multiple market crashes, QE, zero rates, normalization
2. **Momentum works in trends**: But 2022 crash killed momentum factors
3. **Volatility spikes**: 2020 COVID, 2022 inflation → momentum breaks
4. **Correlation regime**: If all ETFs move together → no cross-sectional edge

#### Investigation Steps
```python
# A. Performance by market regime
# Classify periods: Bull (SPY > MA200), Bear (< MA200), High Vol (VIX > 25)
spy_returns = get_spy_returns()
vix = get_vix()
for period in results.index:
    if spy_returns[period] > spy_returns.rolling(200).mean():
        regime = 'bull'
    elif vix[period] > 25:
        regime = 'high_vol'
    else:
        regime = 'bear'
    results.loc[period, 'regime'] = regime

# Group performance by regime
results.groupby('regime')['ls_return'].agg(['mean', 'std', 'count'])
# If bear/high_vol have large negative returns → regime dependency

# B. Rolling correlation
etf_returns = panel.groupby('Date').apply(lambda x: x['Close'].pct_change())
avg_corr_per_date = etf_returns.corr().mean().mean()
# Plot avg_corr vs strategy return
# If corr > 0.8 when returns are negative → can't trade in high-corr regimes

# C. Test regime-adaptive approach
# Only trade when:
# - VIX < 25 (not too volatile)
# - SPY above MA200 (uptrend)
# - ETF correlation < 0.7 (dispersion present)
```

#### Potential Fixes
1. **Regime filter**: Only trade in favorable regimes (bull + low vol)
2. **Dynamic leverage**: Reduce exposure in high-vol periods
3. **Switch to mean reversion**: In range-bound regimes, flip to reversal
4. **Cash mode**: Sit in cash when regime is unfavorable

---

### Hypothesis 7: Data Quality Issues
**Severity**: 🟡 MODERATE  
**Likelihood**: LOW-MEDIUM

#### Thesis
Yahoo Finance data may have quality problems:
1. **Survivorship bias**: Only ETFs alive today are in universe
2. **Dividend adjustments**: Incorrect split/dividend adjustments
3. **Delisting events**: ETFs that closed had bad performance
4. **Intraday NAV issues**: Prices don't reflect true NAV

#### Investigation Steps
```python
# A. Check for gaps/spikes
for ticker in tickers:
    prices = panel.loc[pd.IndexSlice[:, ticker], 'Close']
    returns = prices.pct_change()
    
    # Look for suspicious moves
    outliers = returns[abs(returns) > 0.2]  # +20% in 1 day
    if len(outliers) > 0:
        print(f"{ticker}: {len(outliers)} suspicious moves")
        # Investigate: stock split? bad data?

# B. Cross-validate with alternative source
# Download same ETF data from Alpha Vantage or Polygon
# Compare Close prices: should match exactly (adjusted for splits)

# C. Check for missing data patterns
missing_by_ticker = panel.groupby('Ticker')['Close'].apply(lambda x: x.isna().sum())
if missing_by_ticker.max() > 10:  # > 10 missing days
    print(f"High missing data in: {missing_by_ticker.nlargest(10)}")

# D. Survivorship analysis
# Get list of ALL ETFs from 2011 (including delisted)
# If current universe only includes survivors → positive bias
```

#### Potential Fixes
1. **Use professional data**: Switch to Bloomberg/FactSet
2. **Apply robust estimators**: Winsorize extreme returns
3. **Exclude problematic tickers**: Drop ETFs with data issues
4. **Backfill with proxy**: Use related ETF for missing data

---

## II. DIAGNOSTIC DEEP DIVE

### A. Return Attribution Analysis
```python
# Decompose ls_return into components
attribution = {
    'long_asset_return': results['long_ret'].mean(),
    'short_asset_return': results['short_ret'].mean(),
    'gross_asset_return': results['long_ret'].mean() + results['short_ret'].mean(),
    'cash_return': results['cash_ret'].mean(),
    'financing_cost': -(results['borrow_cost'].mean() + results['margin_interest'].mean()),
    'transaction_cost': -results['transaction_cost'].mean(),
    'net_return': results['ls_return'].mean()
}

# Waterfall analysis
print("Return Attribution (per period):")
print(f"Long positions:    {attribution['long_asset_return']*100:>7.3f}% ✓")
print(f"Short positions:   {attribution['short_asset_return']*100:>7.3f}% ✗ PROBLEM")
print(f"Gross alpha:       {attribution['gross_asset_return']*100:>7.3f}%")
print(f"Cash interest:     {attribution['cash_return']*100:>7.3f}%")
print(f"Financing costs:   {attribution['financing_cost']*100:>7.3f}%")
print(f"Transaction costs: {attribution['transaction_cost']*100:>7.3f}%")
print(f"Net return:        {attribution['net_return']*100:>7.3f}%")

# Key insight: If short_asset_return < 0 (shorts appreciate) → model is inverted
```

### B. Position-Level Forensics
```python
# Analyze top 10 most-shorted ETFs
from collections import Counter
all_shorts = []
for period in results.index:
    short_positions = get_short_weights_at_date(period)  # From backtest
    all_shorts.extend(short_positions.index.tolist())

most_shorted = Counter(all_shorts).most_common(10)
print("\nTop 10 Most Shorted ETFs:")
for ticker, count in most_shorted:
    # Get actual performance of this ticker
    ticker_returns = panel.loc[pd.IndexSlice[:, ticker], 'Close'].pct_change(21)
    mean_ret = ticker_returns.mean()
    print(f"{ticker}: shorted {count} times, avg 21-day return: {mean_ret*100:>+6.2f}%")
    # If mean_ret > 0 → we shorted winners (BAD)

# Same for longs
all_longs = []
for period in results.index:
    long_positions = get_long_weights_at_date(period)
    all_longs.extend(long_positions.index.tolist())

most_longed = Counter(all_longs).most_common(10)
print("\nTop 10 Most Longed ETFs:")
for ticker, count in most_longed:
    ticker_returns = panel.loc[pd.IndexSlice[:, ticker], 'Close'].pct_change(21)
    mean_ret = ticker_returns.mean()
    print(f"{ticker}: longed {count} times, avg 21-day return: {mean_ret*100:>+6.2f}%")
    # If mean_ret > 0 → we longed winners (GOOD)
```

### C. Feature Importance Analysis
```python
# Which features drive short selection?
# For each training window, extract feature importances
feature_importances_per_window = []

for window_id in range(len(training_windows)):
    # Get bins for this window
    model = train_alpha_model(...)
    
    # For each feature, compute IC with forward returns
    training_data = panel.loc[training_window_dates]
    for feat in features:
        ic = training_data[feat].corr(training_data['FwdRet_21'])
        feature_importances_per_window.append({
            'window': window_id,
            'feature': feat,
            'ic': ic
        })

ic_df = pd.DataFrame(feature_importances_per_window)
mean_ic_by_feature = ic_df.groupby('feature')['ic'].mean().sort_values()

print("Features with NEGATIVE IC (inversely correlated):")
print(mean_ic_by_feature[mean_ic_by_feature < 0])
# These features are WRONG - they predict opposite of reality

print("\nFeatures with POSITIVE IC:")
print(mean_ic_by_feature[mean_ic_by_feature > 0])
# These are predictive

# KEY QUESTION: Are we using negative-IC features?
# If supervised binning uses them → bins will be inverted
```

### D. Temporal Analysis
```python
# Check if performance deteriorated over time
results['year'] = pd.to_datetime(results['date']).dt.year
annual_performance = results.groupby('year').agg({
    'ls_return': ['mean', 'std', 'count'],
    'long_ret': 'mean',
    'short_ret': 'mean',
    'turnover': 'mean'
})

print(annual_performance)
# Look for:
# 1. Is short_ret getting MORE negative over time? (model decay)
# 2. Is turnover increasing? (chasing noise)
# 3. Was there a regime shift (e.g., 2020 COVID)?

# Plot cumulative return by year
cumulative_by_year = {}
for year in results['year'].unique():
    year_data = results[results['year'] == year]
    cumulative_by_year[year] = (1 + year_data['ls_return']).prod() - 1

# If 2020-2022 destroyed the strategy → regime shift
```

---

## III. REMEDIATION ACTION PLAN

### Phase 1: Immediate Diagnostics (Week 1)
**Priority**: 🔴 URGENT

1. **Run momentum reversal test** (Hypothesis 1)
   - Check if bottom decile outperforms top decile
   - Compute return autocorrelations
   - If confirmed → strategy is fundamentally inverted

2. **Test minimal feature set** (Hypothesis 2)
   - Backtest with ONLY Close%-21, Close%-63, Close%-126
   - If this outperforms → feature bloat confirmed

3. **Position-level forensics** (Section II.B)
   - Identify which ETFs are consistently shorted
   - Check if they have positive returns
   - If yes → immediate smoking gun

4. **Check bin stability** (Hypothesis 3)
   - Compare bin edges across windows
   - If correlation < 0.3 → overfitting confirmed

**Deliverable**: 2-page executive summary with:
- Root cause hypothesis (ranked by likelihood)
- Key diagnostic charts
- Recommended next steps

### Phase 2: Rapid Prototyping (Week 2)
**Priority**: 🟡 HIGH

1. **Test inversions**:
   - Flip short signal (long bottom, short top)
   - If this works → confirm reversal hypothesis

2. **Simplify model**:
   - Remove supervised binning → use simple momentum ranks
   - Test with 3-month momentum (63-day) only
   - Hold for 63 days (not 21)

3. **Regime filtering**:
   - Only trade when VIX < 25 AND SPY above MA200
   - Expect 50% fewer trades, but better quality

4. **Universe refinement**:
   - Test with top 30 ETFs by ADV only
   - Exclude thematic/leveraged ETFs
   - Sector-neutral construction

**Deliverable**: 4 alternative model configurations with backtest results

### Phase 3: Model Rebuild (Weeks 3-4)
**Priority**: 🟡 MEDIUM

Based on Phase 1-2 findings, rebuild with:

**Option A: Momentum with Reversal Filter**
- Long: Top 20% by 6-month momentum, exclude if 1-month < 0
- Short: Bottom 20% by 6-month momentum, exclude if 1-month > 0
- Rationale: Filter out short-term reversals

**Option B: Pure Relative Strength**
- Rank ETFs by 12-month return
- Long top 10, short bottom 10
- Rebalance quarterly (63 days)
- Rationale: Simpler, less overfitting

**Option C: Sector-Neutral Pairs**
- Within each sector, long top 2, short bottom 2
- Dollar-neutral within sector
- Rationale: Avoid sector rotation effects

**Option D: Adaptive Regime Model**
- In bull/low-vol: Standard momentum
- In bear/high-vol: Mean reversion
- In ranging: Cash
- Rationale: Match strategy to regime

### Phase 4: Validation & Risk Review (Week 5)
**Priority**: 🟢 STANDARD

1. **Out-of-sample validation**: Test on 2021-2023 holdout
2. **Monte Carlo simulation**: Bootstrap confidence intervals
3. **Stress testing**: 2008, 2020 COVID scenarios
4. **Transaction cost sensitivity**: Test at 15 bps, 20 bps
5. **Risk metrics**: Sharpe, max DD, tail risk
6. **Robustness checks**: Different universes, lookbacks

---

## IV. RED FLAGS TO MONITOR

### Critical Indicators (Check Daily)
- **Short return < -0.5%**: Shorts appreciating rapidly
- **Win rate < 40%**: More losses than wins
- **Turnover > 100%**: Excessive churn
- **IC < 0**: Features inversely correlated with returns
- **Sharpe < 0**: Risk-adjusted losses

### Warning Signs (Check Weekly)
- **Long/short return correlation > 0.5**: Both sides moving together
- **Leverage drift**: Actual leverage >> target leverage
- **Concentration risk**: > 30% in single position
- **Data gaps**: Missing prices for > 5% of universe
- **Bin instability**: Kendall τ < 0.3 between windows

### Governance Failures (Check Monthly)
- **No IC tracking**: Can't see feature degradation
- **No regime awareness**: Trading in all conditions
- **No position-level analysis**: Don't know what we're shorting
- **No cost decomposition**: Can't separate alpha from execution
- **No benchmark**: Can't assess relative performance

---

## V. INVESTIGATION PRIORITIZATION MATRIX

| Hypothesis | Likelihood | Impact | Investigation Cost | Priority |
|------------|-----------|--------|-------------------|----------|
| 1. Momentum Reversal | HIGH | CRITICAL | LOW | 🔴 **P0** |
| 2A. Feature Leakage | MEDIUM | CRITICAL | LOW | 🔴 **P0** |
| 3. Supervised Binning Overfit | HIGH | CRITICAL | MEDIUM | 🔴 **P0** |
| 2B. Feature Collinearity | HIGH | HIGH | LOW | 🟡 **P1** |
| 6. Regime Mismatch | MEDIUM | HIGH | MEDIUM | 🟡 **P1** |
| 4. Universe Selection Bias | MEDIUM | MEDIUM | MEDIUM | 🟡 **P2** |
| 2C. Feature Normalization | MEDIUM | MEDIUM | LOW | 🟡 **P2** |
| 5. Transaction Costs | LOW | MEDIUM | LOW | 🟢 **P3** |
| 7. Data Quality | LOW | LOW | HIGH | 🟢 **P3** |

**Recommended sequence**:
1. Start with P0 items (can complete in 2-3 days)
2. If P0 doesn't solve it, move to P1 (1 week)
3. P2-P3 only if above inconclusive

---

## VI. RESOURCES & REFERENCES

### Code Locations
- Backtest results: `D:/REPOSITORY/Data/crosssecmom2/cs_momentum_results.csv`
- Feature panel: `D:/REPOSITORY/Data/crosssecmom2/cs_momentum_features.parquet`
- Universe metadata: `D:/REPOSITORY/Data/crosssecmom2/universe_metadata.parquet`
- Strategy code: `d:/REPOSITORY/morias/Quant/strategies/crosssecmom2/`

### Key Functions
- `run_walk_forward_backtest()`: Main backtest loop
- `train_alpha_model()`: Supervised binning
- `construct_portfolio()`: Long/short construction
- `evaluate_portfolio_return()`: Performance attribution
- `analyze_performance()`: Summary statistics

### External References
- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Asness, Moskowitz, Pedersen (2013): "Value and Momentum Everywhere"
- Israel & Moskowitz (2013): "The Role of Shorting, Firm Size, and Time on Market Anomalies"
- Novy-Marx (2012): "Is Momentum Really Momentum?"
- Daniel & Moskowitz (2016): "Momentum Crashes"

### Academic Insights
1. **Momentum works on stocks, not always ETFs**: ETFs may mean-revert faster
2. **Short side is harder**: Borrow costs, crowding, gamma squeeze risks
3. **Lookback period matters**: 12-month > 6-month > 3-month for persistence
4. **Skip recent month**: Reversal effects in last 1 month
5. **Regime-dependent**: Works in trends, fails in crashes/reversals

---

## VII. FINAL DIRECTIVE TO AI ASSISTANTS

### When investigating this strategy:

1. **DO NOT assume accounting bugs**: FIX 0-5 are verified correct
2. **START with position-level analysis**: What are we actually shorting?
3. **CHECK momentum persistence**: Is this a reversal universe?
4. **TEST simplifications first**: Occam's razor - simpler may be better
5. **MEASURE IC at every step**: Features must correlate with forward returns
6. **QUESTION supervised binning**: May be the root cause
7. **CONSIDER regime filters**: Don't force-trade in all conditions
8. **VALIDATE with alternative data**: Cross-check Yahoo Finance
9. **DOCUMENT every finding**: Update this file with results
10. **THINK independently**: Don't assume momentum must work on ETFs

### Success Criteria
- Short positions have **positive** forward returns (shorts depreciate)
- Win rate > 50%
- Sharpe ratio > 0.5
- Max drawdown < 30%
- IC > 0.02 consistently

### Escalation Protocol
If after Phase 1-2 diagnostics you cannot identify root cause:
1. Document all tests performed
2. List remaining hypotheses
3. Escalate to senior quant researcher
4. Consider external code review
5. May need to rebuild from scratch

---

**This is a scientific investigation, not an engineering bug hunt. Think deeply, test rigorously, and be willing to conclude that momentum may not work on this ETF universe.**

---

*Document Version: 1.0*  
*Author: CDO (via AI Assistant)*  
*Status: Active Investigation*  
*Last Updated: November 22, 2025*
