# Cross-Sectional Momentum Strategy (crosssecmom2)

> **Last Updated**: December 19, 2024  
> **Status**: Production-Ready with Full Cost Modeling, Regime Switching & Enhanced Features  
> **Strategy Type**: Cross-Sectional Momentum on 116 ETFs (2015-2025)

---

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Strategy Logic](#strategy-logic)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Cost Modeling](#cost-modeling)
- [Portfolio Construction](#portfolio-construction)
- [Regime-Based Switching](#regime-based-switching)
- [Walk-Forward Backtesting](#walk-forward-backtesting)
- [Testing](#testing)
- [Performance](#performance)
- [Debugging](#debugging)

---

## Overview

### What This Strategy Does

**crosssecmom2** implements a cross-sectional momentum strategy that:
1. **Trains models** on 5-year rolling windows using supervised binning
2. **Ranks ETFs** by momentum scores each month
3. **Goes long** top quintile, **shorts** bottom quintile
4. **Rebalances monthly** with realistic transaction and borrowing costs
5. **Enforces margin requirements** (50% default) on short positions

### Key Features

✅ **No Look-Ahead Bias**: Training, binning, and feature selection use only historical data  
✅ **Realistic Cost Modeling**: Transaction costs (commission + slippage) and borrowing costs  
✅ **Margin-Constrained Shorts**: Proper modeling of leverage limits (50% margin = 50% max short)  
✅ **Universe Management**: Handles ETF families, duplicates, theme clustering, caps  
✅ **Enhanced Feature Set**: 96 features including drawdowns, shocks, relative returns, correlations, asset flags, and macro/regime indicators
✅ **Multiple Modes**: Standard long/short, long-only, short-only, regime-based switching  
✅ **Regime Awareness**: Optional market regime detection for adaptive portfolio construction  
✅ **Reproducible**: Fixed random state for deterministic results  
✅ **Production-Ready**: Comprehensive testing, diagnostics, and validation

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required**:
- pandas>=1.5.0, numpy>=1.23.0, yfinance>=0.2.0
- scikit-learn>=1.2.0, scipy>=1.10.0, joblib>=1.2.0
- pyarrow>=10.0.0, numba>=0.56.0

**Optional**: cvxpy>=1.3.0 (for optimal portfolio construction with exact cap enforcement)

### 2. Prepare Data

You need one of these files:
- `etf_universe_names.csv`: Minimal (ticker, name)
- `etf_universe_full.csv`: Full metadata (ticker, name, category, avg_volume_3y, avg_close_3y)

Example minimal file:
```csv
ticker,name
SPY,SPDR S&P 500 ETF
QQQ,Invesco QQQ Trust
IWM,iShares Russell 2000 ETF
```

### 3. Run the Strategy

```bash
# Complete workflow (features → metadata → backtest → analysis)
python main.py --all --model supervised_binned

# Or step by step
python main.py --step feature_eng         # Generate features panel
python main.py --step build_metadata      # Build universe metadata
python main.py --step backtest --model supervised_binned
python main.py --step analyze             # Print statistics
```

### 4. Output Files

- `cs_momentum_features.parquet` - Feature panel (Date, Ticker)
- `universe_metadata.parquet` - ETF families, clusters, caps
- `backtest_results.parquet` - Monthly returns and positions
- `performance_report.txt` - Summary statistics

---

## Strategy Logic

### Walk-Forward Process

Each monthly rebalance:
1. **Train**: 5-year rolling window → fit supervised bins → select features (IC > threshold)
2. **Score**: Apply bins to current cross-section → rank tickers
3. **Construct**: Long top 15%, short bottom 15% (with caps and margin limits)
4. **Evaluate**: Measure 21-day forward return minus costs

### Time Structure

At each rebalance date `t0`:

```
t_train_start              t_train_end         t0              t0 + H
    |-------------------------|------------------|----------------|
    |   Training Window       |   GAP (21 days)  | Holding Period |
    |   (1260 days)           |                  | (21 days)      |
    |                         |                  |                |
    Features + FwdRet ----->  |                  |                |
                                                 |                |
                              Score → Portfolio → Evaluate FwdRet
```

- Training: t0 - 1260 days to t0 - 22 days (5-year window, 21-day gap)
- Scoring: t0 (current rebalance date)
- Holding: t0 to t0 + 21 days
- **No look-ahead**: Bins fitted on training window only, applied out-of-sample to t0

---

## Architecture

### File Structure

```
crosssecmom2/
├── config.py                           # All configuration parameters
├── regime.py                           # Market regime classification
├── universe_metadata.py                # ETF families, duplicates, clusters, caps
├── feature_engineering_refactored.py   # Panel generation (raw features)
├── alpha_models.py                     # Model training with supervised binning
├── portfolio_construction.py           # Portfolio construction with costs
├── walk_forward_engine.py             # Walk-forward backtest orchestration
├── main.py                            # Entry point
├── TEST_FIXES.py                      # Tests for basic correctness
├── TEST_ADVANCED_FIXES.py            # Tests for costs and advanced features
├── test_regime.py                     # Tests for regime system
└── README.md                          # This file
```

### Data Flow

```
YFinance
   ↓
feature_engineering_refactored.py
   ↓ (Date, Ticker) panel with Close, returns, momentum, volatility, ADV_63, FwdRet_21
universe_metadata.py
   ↓ families, duplicates, clusters, caps
walk_forward_engine.py
   ├→ alpha_models.py (train model per window)
   ├→ portfolio_construction.py (build portfolio with caps, costs)
   └→ Backtest results (monthly returns, positions, costs)
```

### Key Design Principles

1. **Panel Structure**: (Date, Ticker) MultiIndex for efficient cross-sectional operations
2. **Configuration-Driven**: No hardcoded paths, all parameters in config.py
3. **Model-Agnostic**: Any model that implements `train/score` interface
4. **Cost-Realistic**: Transaction costs, borrowing costs, margin requirements
5. **Testable**: Comprehensive test suites for correctness and advanced features

---

## Configuration

All parameters in `config.py` via `ResearchConfig` dataclass:

```python
from config import get_default_config
config = get_default_config()

# Key parameters
config.time.TRAINING_WINDOW_DAYS = 1260  # 5 years
config.time.HOLDING_PERIOD_DAYS = 21     # Monthly
config.portfolio.long_quantile = 0.85    # Top 15%
config.portfolio.short_quantile = 0.15   # Bottom 15%
config.portfolio.margin_regime = "reg_t_maintenance"  # 25%/30% margins
config.portfolio.short_borrow_rate = 0.055  # 5.5% annual
config.portfolio.margin_interest_rate = 0.055  # 5.5% annual
config.features.ic_threshold = 0.02      # Feature selection
```

See `config.py` for complete parameter list and validation rules.

---

## Accounting and Leverage Fixes (November 2025)

**Status**: ✅ All fixes (FIX 0-5) implemented and verified

### What Was Fixed

The strategy underwent comprehensive accounting and leverage fixes to ensure correctness:

1. **FIX 0 - Decimal Convention**: All returns stored as decimals (0.05 = 5%) throughout system
2. **FIX 1 - Margin Regime**: Leverage controlled by `margin_regime` setting, not deprecated parameters
3. **FIX 2 - Post-Cap Rescaling**: Caps enforced while preserving target exposures from margin regime
4. **FIX 3 - Active Margins**: All calculations use `get_active_margins()` from current regime
5. **FIX 4 - Capital Compounding**: Explicit capital state tracking with automatic verification
6. **FIX 5 - Design Cleanup**: Capital passed directly to constructors (cleaner architecture)

**Impact**: These fixes correct the accounting mechanics without changing the scientific methodology (features, scoring, ranking). Returns are now properly scaled, leverage matches margin settings, and capital compounds correctly.

**Documentation**: See `leverage_fixes_summary.md` and `FIXES_COMPLETE.md` for complete details.

**Performance Note**: The accounting is mathematically correct (verified), but the strategy currently underperforms (-69% return, 39% win rate) due to MODEL issues - shorts consistently go UP instead of DOWN, suggesting momentum reversal or feature/binning problems requiring scientific investigation.

---

## Cost Modeling

### Transaction Costs
`cost = (commission_bps + slippage_bps) × turnover / 10000`  
Turnover = 0.5 × Σ|w_t - w_{t-1}| per side

### Financing Costs (Phase 1 - November 2025)

**Separate rates for different costs**:
- `short_borrow_rate`: Annual fee to borrow shares for shorting (5.5%)
- `margin_interest_rate`: Annual interest on cash borrowed for longs (5.5%)

**Long positions**:
```python
borrowed_long = gross_long × (1 - margin_requirement)
cost = margin_interest_rate × borrowed_long × (days/365)
```

**Short positions**:
```python
borrowed_short = gross_short  # Always full notional
cost = short_borrow_rate × borrowed_short × (days/365)
```

**Cash interest** (earned on uninvested capital):
```python
cash_balance = 1.0 - (gross_long × margin_long + gross_short × margin_short)
interest = cash_rate × cash_balance × (days/365)
```

### Margin Regimes

Three options via `margin_regime` parameter:
- **reg_t_initial** (50%/50%) - Opening positions
- **reg_t_maintenance** (25%/30%) - Held positions ← DEFAULT
- **portfolio** (15%/15%) - Risk-based margin

**Max leverage formula** (dollar-neutral):
```python
max_position = capital / (margin_long + margin_short)
# Example: 1.0 / (0.25 + 0.30) = 1.82 each side = 3.64x gross
```

### Cash Ledger

Every period includes complete cash accounting via `result['cash_ledger']`:
- Capital deployment (margins posted, cash balance)
- Borrowing breakdown (long/short)
- Financing costs (interest earned vs. paid)
- Asset returns (long/short)
- Total P&L

Automatic verification: `margin_posted + cash_balance = 1.0` ✓

---

## Portfolio Construction

### Two Methods

#### 1. Simple Method (Fast)


```python
long_weights, short_weights = construct_portfolio_simple(
    scores=scores,
    universe_metadata=metadata,
    config=config,
    enforce_caps=True
)
```

**Algorithm**:
1. Select top/bottom quantiles
2. Equal-weight within each side
3. Scale to enforce caps (approximate)

**Pros**: Very fast, no optimization required  
**Cons**: Cap enforcement is approximate (scales globally)

#### 2. CVXPY Method (Optimal)

```python
long_weights, short_weights = construct_portfolio_cvxpy(
    scores=scores,
    universe_metadata=metadata,
    config=config
)
```

**Algorithm**:
```
maximize: Σ(score_i × weight_i)
subject to:
  |w_i| ≤ per_etf_cap_i                    (individual caps)
  Σ_{i ∈ cluster_k} |w_i| ≤ cluster_cap_k  (cluster caps)
  Σ(long) = 1.0                             (full long deployment)
  Σ|short| = margin                         (margin constraint)
```

**Pros**: Exact cap enforcement, optimal allocation  
**Cons**: Requires cvxpy installation, slower

### Caps System

**Per-ETF Caps**: Max weight per ticker (default 5%)

**Cluster Caps**: Max weight per theme cluster (default 10%)

**Example**:
```
Cluster: EQ_US_LARGECAP (SPY, IVV, VOO)
- All are highly correlated (duplicates)
- Cluster cap = 10%
- Per-ETF cap = 5%
- Result: Max 2 of these ETFs can be held (2 × 5% = 10%)
```

### Long-Only and Short-Only Modes

#### Long-Only Mode
```python
config.portfolio.long_only = True
config.portfolio.short_only = False
```

**Behavior**:
- Only long positions created
- Short positions skipped
- Uninvested capital earns cash_rate
- Transaction cost = cost_bps × long_turnover
- Borrowing cost = 0

#### Short-Only Mode
```python
config.portfolio.long_only = False
config.portfolio.short_only = True
```

**Behavior**:
- Only short positions created (limited by margin)
- Long positions skipped
- Transaction cost = cost_bps × short_turnover
- Borrowing cost = borrow_rate × gross_short × time

---

## Regime-Based Switching

Adaptive portfolio construction based on market conditions:

**Regimes**:
- **Bull**: Close > MA200 AND 63d return > +2% → Long-only  
- **Bear**: Close < MA200 AND 63d return < -2% → Short-only
- **Range**: Otherwise → Cash

```python
config.regime.use_regime = True
config.regime.market_ticker = 'SPY'
config.regime.ma_window = 200
config.regime.lookback_return_days = 63
config.regime.use_hysteresis = True  # Prevent rapid switching
config.regime.neutral_buffer_days = 21
```

**Implementation**: Regime computed once before backtest (shifted 1 day for no look-ahead). At each rebalance, lookup regime → map to mode (`long_only`, `short_only`, `cash`, or `ls`) → construct portfolio.

---

## Walk-Forward Backtesting

```python
from walk_forward_engine import run_walk_forward_backtest

results = run_walk_forward_backtest(
    panel_df=panel,
    universe_metadata=metadata,
    config=config,
    model_type='supervised_binned',
    verbose=True
)
```

**Per-Period Process**:
1. Train model on 5 years ending 22 days before t0
2. Score universe at t0 using trained model
3. Construct portfolio (top/bottom quantiles)
4. Evaluate returns + costs over 21-day hold
5. Step forward 21 days

**Results DataFrame**: `date`, `ls_return`, `long_ret`, `short_ret`, `txn_cost`, `borrow_cost`, `n_long`, `n_short`, `long_tickers`, `short_tickers`, `cash_ledger`

---

## Testing

**Run tests**: `python TEST_ADVANCED_FIXES.py`

Validates:
- Training restricted to core universe
- Universe filters applied correctly
- Long-only/short-only modes work
- Transaction costs calculated properly
- Borrowing costs on full notional
- CVXPY fallback to simple method
- Config validation catches errors
- Margin limits enforced

---

## Performance Analysis

```python
from walk_forward_engine import analyze_performance
stats = analyze_performance(results, config)
```

Returns: periods, win rate, mean return, Sharpe, max drawdown, costs breakdown.

**Note**: Run `python main.py --all --model supervised_binned` to generate real backtest (30-60 min for 10 years × 116 ETFs).

---

## Debugging & Common Issues

**Check feature panel**:
```python
panel = pd.read_parquet('cs_momentum_features.parquet')
assert panel.index.names == ['Date', 'Ticker']
assert 'FwdRet_21' in panel.columns
assert len([c for c in panel.columns if c.endswith('_Bin')]) == 0  # No global bins
```

**Check universe metadata**:
```python
metadata = pd.read_parquet('universe_metadata.parquet')
print(f"Core ETFs: {metadata['in_core_after_duplicates'].sum()}")
print(f"Clusters: {metadata['cluster_id'].nunique()}")
```

**Common issues**:
- **"Insufficient training data"**: Reduce `TRAINING_WINDOW_DAYS` or use earlier `START_DATE`
- **"No features passed IC threshold"**: Lower `ic_threshold` or add more candidates
- **"Borrowing cost seems wrong"**: Cost = rate × full_notional × days/365 (not margin-adjusted)
- **"Cap violations"**: Use CVXPY method (`pip install cvxpy`) for exact enforcement

---


results = run_walk_forward_backtest(
    ...,
    portfolio_method='cvxpy'  # Instead of 'simple'
)
```

### Issue 5: "Short positions are 100% when margin is 50%"

**Symptom**: Gross short exposure not limited by margin

**Status**: FIXED ✅

**Verification**:
```python
# Test that margin correctly limits shorts
python TEST_ADVANCED_FIXES.py  # See test_margin_limits_short_exposure()
```

---

## Support and Documentation

### Primary Documentation
- **This README.md**: Complete system documentation

### Code Documentation
- **config.py**: All parameters with docstrings
- **alpha_models.py**: Model interface and examples
- **portfolio_construction.py**: Portfolio methods and cost calculations
- **walk_forward_engine.py**: Backtest orchestration

### Test Files
- **TEST_FIXES.py**: Basic correctness validation
- **TEST_ADVANCED_FIXES.py**: Advanced features validation

### Getting Help

1. Check this README first
2. Review test files for examples
3. Check code docstrings for parameter details
4. Run with `verbose=True` for diagnostic output

---

## Appendix: Mathematical Details

### Information Coefficient (IC)

**Definition**: Spearman rank correlation between feature ranks and forward return ranks

**Formula**:
```
IC = correlation(rank(feature), rank(forward_return))
```

**Interpretation**:
- IC > 0.02: Feature is predictive (kept)
- IC < 0.02: Feature is noise (excluded)
- IC calculated on training window only

### Supervised Binning

**Algorithm**:
1. Fit decision tree: `tree.fit(feature_values, forward_returns)`
2. Extract split points from tree leaves
3. Apply: `binned_feature = np.digitize(feature, boundaries)`

**Why?**:
- Captures non-linear relationships
- More robust than quantile bins
- Adaptive to distribution

### Turnover Formula

**Per-side turnover**:
```
turnover_long = 0.5 × Σ|w_long[t] - w_long[t-1]|
turnover_short = 0.5 × Σ|w_short[t] - w_short[t-1]|
```

**Examples**:
- Full entry: Σ|w_t| = 1.0, turnover = 0.5 × 1.0 = 0.5 (one-way)
- Full exit: Σ|w_{t-1}| = 1.0, turnover = 0.5 × 1.0 = 0.5 (one-way)
- Full rebalance: turnover = 0.5 × (1.0 + 1.0) = 1.0 (round-trip)

### Sharpe Ratio

**Period Sharpe**:
```
Sharpe = mean(returns) / std(returns)
```

**Annualized Sharpe** (21-day periods):
```
Annual Sharpe = Sharpe × sqrt(252 / 21) = Sharpe × 3.464
```

---
.02--/