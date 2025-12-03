# Forensic Study Implementation Guide

## Overview

This guide explains how to run the forensic study module and provides details on:
- Data sources and preparation
- Performance optimizations
- Phase-by-phase execution
- Extending the module for additional analyses

---

## Quick Start

```python
from forensic_study import ForensicStudy, ForensicConfig
import pandas as pd

# Load your backtest results
results_df = pd.read_parquet("path/to/walk_forward_results.parquet")

# Load panel data
panel_df = pd.read_parquet("D:/REPOSITORY/Data/crosssecmom2/panel_features.parquet")

# Load universe metadata
metadata = pd.read_csv("D:/REPOSITORY/Data/crosssecmom2/universe_metadata.csv")

# Configure and run
config = ForensicConfig(
    output_dir="forensic_outputs",
    n_random_trials=1000,
    n_jobs=-1,  # Use all CPU cores
)

study = ForensicStudy(
    results_df=results_df,
    panel_df=panel_df,
    universe_metadata=metadata,
    forensic_config=config
)

# Run complete analysis
study.run_all()

# Or run specific phases
study.run_phase_0_integrity_audit()
study.run_phase_0b_null_hypothesis_tests()
study.run_phase_2_ranking_analysis()
study.run_statistical_power_analysis()
study.run_phase_5_counterfactual_analysis()
```

---

## Data Sources

### Required Data

| Data | Location | Format | Purpose |
|------|----------|--------|---------|
| Walk-forward results | `walk_forward_results.parquet` | Parquet | Strategy returns, holdings per period |
| Panel features | `D:/REPOSITORY/Data/crosssecmom2/panel_features.parquet` | Parquet | Per-ticker per-date features + forward returns |
| Universe metadata | `universe_metadata.csv` | CSV | ETF family, sector, inception date |

### Key Columns in Panel Data

```
Date (index level 0)
Ticker (index level 1)
Close, Volume, market_cap, ADV_63
FwdRet_21 (target: 21-day forward return)
Close%-21, Close%-42, ... (momentum features)
mom12_1, mom6_1, ... (standard momentum)
vol_adj_mom, ... (adjusted features)
```

### Key Columns in Results DataFrame

```
Index: Date (rebalance dates)
long_tickers: dict {ticker: weight}
short_tickers: dict {ticker: weight}
long_ret: float
short_ret: float
ls_return: float (long-short return)
```

### Diagnostics in results_df.attrs

```python
results_df.attrs['diagnostics'] = [
    {
        'date': pd.Timestamp,
        'feature_details': [
            {'feature': str, 'coefficient': float, 'ic': float}
        ],
        'model_fit': {
            'r_squared': float,
            'model_ic': float,
            'residual_std': float
        },
        'feature_flow': {...}
    },
    ...
]
```

---

## External Data to Download

### DXY (US Dollar Index)

For Phase 5 factor attribution (correlation with dollar moves):

```python
import yfinance as yf

# Download DXY
dxy = yf.download('DX-Y.NYB', start='2019-01-01')
dxy.to_parquet('D:/REPOSITORY/Data/crosssecmom2/external/DXY.parquet')
```

### VIX (if not already in panel)

```python
vix = yf.download('^VIX', start='2019-01-01')
vix.to_parquet('D:/REPOSITORY/Data/crosssecmom2/external/VIX.parquet')
```

### ETF Fund Flows (Approximate)

No free data source for actual flows. Proxy approach:
- Use Volume × Close as dollar volume proxy
- Compare to trailing average to detect unusual flow days

---

## Macro Event Dates

FOMC, CPI, NFP dates are hardcoded in `load_macro_event_dates()`:

```python
from forensic_study import load_macro_event_dates

events = load_macro_event_dates()
print(events['fomc'][:5])  # FOMC dates
print(events['cpi'][:5])   # CPI release dates
print(events['nfp'][:5])   # NFP dates
```

For more accurate dates, consider:
- FRED API for historical CPI/NFP dates
- Federal Reserve website for FOMC calendar

---

## Performance Optimizations

### 1. Parallelization (joblib)

Monte Carlo simulations (random baselines) are parallelized:

```python
from joblib import Parallel, delayed

def single_trial(seed):
    np.random.seed(seed)
    # ... simulation logic
    return sharpe

# Run 1000 trials across all cores
sharpes = Parallel(n_jobs=-1)(
    delayed(single_trial)(i) for i in range(1000)
)
```

**Configuration:**
```python
config = ForensicConfig(
    n_random_trials=1000,  # Reduce for faster testing
    n_jobs=-1,  # -1 = all cores, or specify number
)
```

### 2. Vectorized Operations (NumPy/Pandas)

Quintile analysis uses vectorized operations:

```python
# FAST: Vectorized quintile assignment
quintiles = pd.qcut(returns.rank(), 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
quintile_rets = returns.groupby(quintiles).mean()

# AVOID: Looping over rows
for i, row in df.iterrows():  # Don't do this!
    ...
```

### 3. Pre-Computed Data Caching

The `ForensicStudy` class pre-computes commonly used data:

```python
def _precompute_data(self):
    # Extracted once, used many times
    self.all_dates = self.panel_df.index.get_level_values('Date').unique()
    self.all_tickers = self.panel_df.index.get_level_values('Ticker').unique()
    self._parse_holdings()  # Extract holdings from results once
```

### 4. Memory-Efficient Chunking

For very large datasets, process in chunks:

```python
# Process one rebalance date at a time
for date in self.rebalance_dates:
    cross_section = self.panel_df.loc[date]  # Single date slice
    # ... analyze this period
    gc.collect()  # Free memory
```

### 5. Potential Numba/JIT Optimization

For tight numerical loops, consider Numba:

```python
from numba import jit

@jit(nopython=True)
def fast_ic_calculation(ranks, returns):
    n = len(ranks)
    sum_d_squared = 0.0
    for i in range(n):
        d = ranks[i] - returns[i]
        sum_d_squared += d * d
    return 1 - (6 * sum_d_squared) / (n * (n*n - 1))
```

---

## Phase-by-Phase Execution

### Phase 0: Integrity Audit (< 1 min)

Checks for look-ahead bias:
```python
audit = study.run_phase_0_integrity_audit()

# Check results
if not audit['passed']:
    print("⚠️ Critical integrity issues found!")
    print(audit['critical_failures'])
```

### Phase 0B: Null Hypothesis Tests (5-10 min)

Monte Carlo simulations:
```python
# Full run (1000 trials)
null_tests = study.run_phase_0b_null_hypothesis_tests()

# Fast run (100 trials for testing)
null_tests = study.run_phase_0b_null_hypothesis_tests(n_trials=100)
```

### Phase 2: Ranking Analysis (1-2 min)

Decompose ranking skill:
```python
ranking_df = study.run_phase_2_ranking_analysis()

# Key metrics
print(f"Avg Rank IC: {ranking_df['rank_ic'].mean():.4f}")
print(f"Monotonicity Rate: {ranking_df['is_monotonic'].mean()*100:.1f}%")
```

### Phase 4: Statistical Power (< 1 min)

Confidence intervals:
```python
power = study.run_statistical_power_analysis()

# Is the Sharpe statistically significant?
if power['ci_95_lower'] > 0:
    print("✓ Sharpe is statistically significant")
else:
    print("⚠️ Need more data for statistical significance")
```

### Phase 5: Counterfactual (1-2 min)

Upper bound analysis:
```python
counterfactual = study.run_phase_5_counterfactual_analysis()

# How close to perfect?
capture = counterfactual['perfect_foresight']['capture_ratio']
print(f"Strategy captures {capture*100:.1f}% of perfect foresight return")
```

---

## Extending the Module

### Adding a New Analysis

1. Add method to `ForensicStudy`:

```python
def run_my_custom_analysis(self) -> Dict:
    """My custom analysis."""
    print("\n[Custom] Running my analysis...")
    
    results = {}
    
    # Access data
    for i, diag in enumerate(self.diagnostics):
        # ... your analysis
        pass
    
    # Save results
    self._save_json(results, 'custom_analysis.json')
    return results
```

2. Add to `run_all()`:

```python
def run_all(self):
    ...
    self.run_my_custom_analysis()
    ...
```

### Adding External Data Source

1. Create data loader function:

```python
def load_fred_data(series_id: str, start_date: str) -> pd.DataFrame:
    """Load data from FRED."""
    from fredapi import Fred
    
    fred = Fred(api_key='YOUR_KEY')
    data = fred.get_series(series_id, observation_start=start_date)
    return data
```

2. Integrate into factor analysis:

```python
def _compute_factor_exposures(self):
    # Load external data
    dxy = pd.read_parquet(self.config.data_dir / 'external/DXY.parquet')
    
    # Compute correlation with strategy returns
    merged = self.results_df['ls_return'].to_frame().join(dxy['Close'].pct_change())
    correlation = merged.corr().iloc[0, 1]
    
    return {'dxy_correlation': correlation}
```

---

## Output Files

After running `study.run_all()`, you'll find:

```
forensic_outputs/
├── phase0_integrity_audit.json       # Look-ahead bias checks
├── phase0b_null_hypothesis_tests.json # Random baseline comparisons
├── phase2_ranking_analysis.csv        # Per-period ranking metrics
├── phase4_statistical_power.json      # CI and power analysis
├── phase5_counterfactual_analysis.json # Perfect foresight bounds
└── FORENSIC_SUMMARY.md                # Human-readable summary
```

---

## Troubleshooting

### "Forward return column not found"

Panel data must have `FwdRet_21` column:

```python
# Check available columns
print([col for col in panel_df.columns if 'Fwd' in col])
```

### "Not enough periods for analysis"

Need at least 10 rebalance periods:

```python
print(f"Periods: {len(results_df)}")
```

### Memory errors

Reduce random trials or process in chunks:

```python
config = ForensicConfig(
    n_random_trials=100,  # Reduce from 1000
    bootstrap_samples=500,  # Reduce from 1000
)
```

### Slow execution

1. Check parallelization is working:
```python
import os
print(f"Available cores: {os.cpu_count()}")
```

2. Use skip_slow mode for testing:
```python
study.run_all(skip_slow=True)
```

---

## Interpretation Guide

### Critical Pass/Fail Criteria

| Test | Pass Condition | Fail Implication |
|------|----------------|------------------|
| Integrity Audit | No look-ahead bias | Results are invalid |
| Random Portfolio p-value | p < 0.05 | Strategy is luck |
| Beats Momentum | Excess Sharpe > 0 | ML adds no value |
| 95% CI excludes 0 | CI_lower > 0 | Not significant |
| Inverse Strategy | Actual > Inverse | Model direction correct |

### Key Questions Answered

1. **Is there any signal?** → Phase 0B p-values
2. **Is ranking skill or monetization?** → Phase 2 quintile spreads
3. **Is it statistically significant?** → Phase 4 bootstrap CI
4. **What's the ceiling?** → Phase 5 perfect foresight capture ratio
5. **Is it robust?** → Phase 0 integrity audit

---

## Next Steps After Forensic Study

If all tests pass:
1. Proceed with LambdaMART for improved ranking
2. Add regime model for conditional execution
3. Implement vol-scaling for position sizing

If tests fail:
1. Review feature construction for look-ahead
2. Consider simpler momentum baseline
3. Collect more data for statistical power
4. Investigate failed periods for regime patterns
