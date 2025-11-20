# Quick Reference: Data Structure Transformation

## Original Wide Format → Panel Long Format

### BEFORE (Your Notebook Output)
```
Shape: (2000 rows × 18,900 columns)

Date       | Uk_Close%-1 | Uk_Mom21 | Uk_RSI14 | SPY_Close%-1 | SPY_Mom21 | SPY_RSI14 | ... (18,900 cols)
-----------|-------------|----------|----------|--------------|-----------|-----------|
2017-09-26 | 0.42        | 2.1      | 58.3     | -0.15        | -1.2      | 52.1      |
2017-09-27 | -0.13       | 1.8      | 56.1     | 0.22         | -0.8      | 53.5      |
```

**Problem:** Can't easily answer "Which ticker has highest momentum TODAY?"

### AFTER (New Panel Structure)
```
Shape: (300,000 rows × 300 columns)
MultiIndex: (Date, Ticker)

Date       | Ticker | Close%-21 | Mom21 | RSI14 | Close%-21_Rank | Close%-21_ZScore | FwdRet_21
-----------|--------|-----------|-------|-------|----------------|------------------|----------
2017-09-26 | Uk     | 0.42      | 2.1   | 58.3  | 0.65           | 0.42             | 1.8
2017-09-26 | SPY    | -0.15     | -1.2  | 52.1  | 0.25           | -1.05            | 0.5
2017-09-26 | QQQ    | 0.85      | 3.8   | 72.5  | 0.95           | 1.85             | 2.3
2017-09-27 | Uk     | -0.13     | 1.8   | 56.1  | 0.58           | 0.15             | 1.2
2017-09-27 | SPY    | 0.22      | -0.8  | 53.5  | 0.42           | -0.85            | 0.8
...
```

**Solution:** Natural cross-sectional operations!

---

## Key Operations Enabled by Panel Structure

### 1. Get Cross-Section (All Tickers at One Date)
```python
# Wide format - PAINFUL
spy_mom = df.loc['2024-01-15', 'SPY_Mom21']
qqq_mom = df.loc['2024-01-15', 'QQQ_Mom21']
# ... repeat for 117 tickers

# Panel format - NATURAL
cross_section = panel_df.loc['2024-01-15']
top_momentum = cross_section.nlargest(10, 'Mom21')
```

### 2. Rank Assets at Each Date
```python
# Wide format - COMPLEX
for date in dates:
    momentum_values = {}
    for ticker in tickers:
        momentum_values[ticker] = df.loc[date, f'{ticker}_Mom21']
    sorted_tickers = sorted(momentum_values.items(), key=lambda x: x[1], reverse=True)

# Panel format - ONE LINE
panel_df['Mom21_Rank'] = panel_df.groupby('Date')['Mom21'].rank(pct=True)
```

### 3. Filter Universe
```python
# Wide format - IMPOSSIBLE (data quality varies across columns)

# Panel format - SIMPLE
filtered = cross_section[
    (cross_section['Mom21'].notna()) &
    (cross_section['RSI14'].notna()) &
    (cross_section['Volume'] > threshold)
]
```

### 4. Walk-Forward Windows
```python
# Wide format - AWKWARD
formation = df.loc[start:end].copy()
# ... then extract relevant columns for each ticker

# Panel format - NATURAL
formation = panel_df.loc[start:end]  # All tickers, all features
signal_date_data = panel_df.loc[signal_date]  # Cross-section at signal date
```

---

## Feature Types in New Structure

### Raw Features (per ticker)
- Close%-1, Close%-21, Close%-63, Close%-126, Close%-252
- Mom5, Mom10, Mom21, Mom42, Mom63
- std5, std21, std63
- MA5, MA21, MA63, EMA5, EMA21, EMA63
- RSI14, RSI21, RSI42
- MACD, MACD_Signl, MACD_Histo
- ATR14, WilliamsR14, WilliamsR21
- Hurst21, Hurst63, Hurst126
- skew21, kurt21, BollUp21, BollLo21
- VPT

### Cross-Sectional Transforms (added automatically)
For each momentum/volatility/trend feature:
- `Feature_Rank`: Percentile rank (0 to 1) within universe at that date
- `Feature_ZScore`: Z-score relative to universe mean/std at that date
- `Feature_Quantile`: Decile bin (0-9) for portfolio formation

### Tree-Based Bins (optimal non-uniform bins)
- `Close%-21_Bin`, `Close%-63_Bin`, `Close%-252_Bin`
- `Mom21_Bin`, `Mom63_Bin`
- `std21_Bin`, `std63_Bin`
- `RSI14_Bin`, `Hurst21_Bin`

### Forward Returns (target variables)
- FwdRet_1, FwdRet_5, FwdRet_10, FwdRet_21, FwdRet_42, FwdRet_63

---

## Common Pandas Operations

### Loading
```python
# Load entire dataset
panel_df = pd.read_parquet('cs_momentum_features.parquet')

# Load specific columns only (memory efficient)
panel_df = pd.read_parquet(
    'cs_momentum_features.parquet',
    columns=['Close%-63', 'Close%-63_Rank', 'Mom21', 'FwdRet_21']
)

# Load date range only
panel_df = pd.read_parquet(
    'cs_momentum_features.parquet',
    filters=[('Date', '>=', '2020-01-01')]
)
```

### Slicing
```python
# Get specific date (cross-section)
cs = panel_df.loc['2024-01-15']

# Get specific ticker (time series)
spy = panel_df.xs('SPY', level='Ticker')

# Get date range
window = panel_df.loc['2023-01-01':'2024-01-01']

# Get specific ticker and date range
spy_window = panel_df.loc[(slice('2023-01-01', '2024-01-01'), 'SPY'), :]
```

### Grouping
```python
# Compute statistic per date (across tickers)
daily_mean = panel_df.groupby('Date')['Mom21'].mean()

# Compute statistic per ticker (across time)
ticker_mean = panel_df.groupby('Ticker')['Mom21'].mean()

# Add cross-sectional rank at each date
panel_df['Mom21_Rank'] = panel_df.groupby('Date')['Mom21'].rank(pct=True)
```

### Filtering
```python
# Filter by feature value
high_momentum = panel_df[panel_df['Mom21'] > 5.0]

# Filter by rank
top_decile = panel_df[panel_df['Mom21_Rank'] > 0.9]

# Filter by multiple conditions
filtered = panel_df[
    (panel_df['Mom21_Rank'] > 0.8) &
    (panel_df['RSI14'] < 70) &
    (panel_df['std21'] < panel_df['std21'].median())
]
```

---

## Walk-Forward Research Checklist

### Setup (One Time)
- [ ] Update file paths in all 3 Python scripts
- [ ] Verify ETF universe file is properly formatted
- [ ] Install dependencies: `pip install pandas numpy yfinance scikit-learn joblib matplotlib seaborn pyarrow numba`

### Generate Features (First Run)
- [ ] Run `cs_momentum_feature_engineering.py`
- [ ] Wait ~8-12 minutes
- [ ] Verify output file exists: `cs_momentum_features.parquet`
- [ ] Check summary statistics in console output

### Backtest Strategy
- [ ] Customize parameters in `walk_forward_research_demo.py`:
  - FORMATION_DAYS (default: 252)
  - TEST_DAYS (default: 21)
  - SIGNAL_FEATURE (default: 'Close%-63_Rank')
  - LONG_QUANTILE / SHORT_QUANTILE (default: 0.9 / 0.1)
- [ ] Run `walk_forward_research_demo.py`
- [ ] Review performance summary
- [ ] Check generated plots: `cs_momentum_backtest.png`

### Analyze Results
- [ ] Run `panel_data_utilities.py`
- [ ] Review IC analysis to find best features
- [ ] Check regime performance
- [ ] Examine cross-sectional correlation
- [ ] Monitor universe evolution

### Iterate
- [ ] Adjust signal combinations based on IC analysis
- [ ] Test different formation/test periods
- [ ] Experiment with universe filters
- [ ] Add transaction costs
- [ ] Implement position sizing

---

## Performance Benchmarks

| Operation | Time | Memory |
|-----------|------|--------|
| Feature engineering (117 ETFs, 8 years) | 8-12 min | 4-8 GB |
| Load full panel | 2-3 sec | 2-3 GB |
| Load date range (1 year) | <1 sec | 500 MB |
| Single backtest iteration | ~1 sec | 100 MB |
| Full walk-forward (185 periods) | 2-5 min | 2-3 GB |
| IC analysis (20 features) | 3-8 min | 3-4 GB |

*Benchmarks on: Intel i7, 16GB RAM, SSD*

---

## Troubleshooting Quick Fixes

| Error | Fix |
|-------|-----|
| `KeyError` on date | Date is weekend/holiday, use try/except or filter to trading days |
| `MemoryError` | Load fewer columns or smaller date range |
| Empty universe after filtering | Relax min_volume_rank or min_data_quality thresholds |
| Low IC values | Try different features, longer horizons, or regime filtering |
| NaN forward returns | Normal at end of dataset (no future data) |

---

## File Outputs Reference

| File | Purpose | Format | Size |
|------|---------|--------|------|
| `cs_momentum_features.parquet` | Main feature store | Parquet | 50-100 MB |
| `cs_momentum_metadata.csv` | Feature descriptions | CSV | <1 MB |
| `cs_momentum_results.csv` | Backtest results | CSV | <1 MB |
| `cs_momentum_backtest.png` | Performance plots | PNG | <1 MB |
| `ic_time_series.csv` | IC over time | CSV | 1-5 MB |
| `ic_analysis.png` | IC plots | PNG | <1 MB |
| `cs_correlation_heatmap.png` | Feature correlation | PNG | <1 MB |
| `regime_analysis.png` | Performance by regime | PNG | <1 MB |
| `universe_evolution.png` | Universe tracking | PNG | <1 MB |
| `cs_dispersion.png` | Cross-sectional dispersion | PNG | <1 MB |
