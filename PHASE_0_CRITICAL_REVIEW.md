# Phase 0 Critical Review and Recommendations

## Your Questions Answered:

### ❌ **Q1: Are you sure the results were good?**

**NO** - The results are **PROBLEMATIC** for several reasons:

1. **Universe Too Small**: Only **18 unique tickers** used across 47 periods
   - With 89 equity-classified ETFs, we should have ~22-23 available per period
   - Only seeing 18 suggests poor coverage

2. **Low Position Counts**: Only 3-4 positions per side
   - With 89 equity ETFs, 25th/75th percentile should give ~22 longs and ~22 shorts
   - Getting only 4 positions means **insufficient universe size after filters**

3. **USO Misclassification**: 
   - `USO` (United States Oil Fund) classified as NON-EQUITY ("Commodities Focused")
   - But appears in backtests as long position (Period 2, 3)
   - **ISSUE**: Filter may not be working correctly during backtest!

4. **XLB Missing**:
   - `XLB` (Materials sector ETF) classified as NON-EQUITY ("Natural Resources")
   - Should probably be equity (it's a sector ETF like XLF, XLI, etc.)

### ✅ **Q2: Should we re-run with cvxpy properly installed?**

**YES** - cvxpy was not available during Phase 0, so portfolio construction likely used:
- Simple equal-weight fallback OR
- Naive heuristic method

Proper cvxpy optimization should:
- Better balance risk/return
- Respect position caps more efficiently
- Potentially improve Sharpe from 0.33 → 0.5+

### ✅ **Q3: Should we check ETF classification before running?**

**YES** - Several issues found:

**INCORRECTLY EXCLUDED** (should be equity):
1. `USO` - United States Oil Fund (commodity exposure but trades like equity)
2. `XLB` - Materials Select Sector SPDR (sector ETF, definitely equity)
3. `XME` - SPDR S&P Metals & Mining ETF (sector ETF)

**CORRECTLY EXCLUDED** (bonds/commodities):
- `GLD`, `SLV`, `IAU` (physical gold/silver)
- `AGG`, `BND`, `TLT` (bond ETFs)
- `GBTC` (Bitcoin trust)

### ✅ **Q4: Should we widen the buckets to get more positions?**

**YES** - Current config is **WRONG** for small universe:

**Current Settings** (from config.py line 96-97):
```python
long_quantile: float = 0.75   # Top 10% for long portfolio
short_quantile: float = 0.25  # Bottom 10% for short portfolio
```

**Comment says "Top 10%" but code says 0.75 quantile = Top 25%!**
- 0.75 quantile = top 25% (above 75th percentile)
- 0.25 quantile = bottom 25% (below 25th percentile)

**Math**:
- If 23 ETFs available per period:
  - Top 25%: 23 * 0.25 = 5.75 → 6 positions
  - We're only getting 4 → universe must be even smaller (16-18 after all filters)

**Recommendation**: Change to **top/bottom 33%** (0.67/0.33 quantiles):
- With 23 ETFs: 23 * 0.33 = 7.6 → 8 positions per side
- With 18 ETFs: 18 * 0.33 = 5.9 → 6 positions per side (better than 4)

---

## Critical Issues Summary

| Issue | Severity | Impact | Fix Required |
|-------|----------|--------|--------------|
| **USO in backtest but marked non-equity** | 🔴 HIGH | Filter not working correctly | Fix family classification OR fix filter logic |
| **Only 18 tickers used (should be ~23)** | 🔴 HIGH | Poor diversification | Check eligibility filters |
| **Only 3-4 positions per side** | 🔴 HIGH | High concentration risk | Widen quantiles to 0.67/0.33 |
| **cvxpy not installed** | 🟡 MEDIUM | Suboptimal portfolio | Re-run with cvxpy |
| **XLB, XME misclassified** | 🟡 MEDIUM | Excluding valid sector ETFs | Add to equity keywords |

---

## Root Cause Analysis

### Why Only 18 Tickers?

**Hypothesis**: Multiple filters compounding:
1. **Equity filter**: 116 → 89 ETFs (-27)
2. **Core universe filter**: Removes duplicates
3. **ADV filter**: 30th percentile liquidity requirement
4. **Data quality filter**: 80% non-NaN features
5. **History requirement**: 504 days training + 63 days lag = 567 days history

**Combined effect**: 89 equity ETFs → ~18-23 eligible per period → only 18 actually used

### Why USO in Results?

**Two possibilities**:
1. USO was eligible BEFORE Phase 0 equity filter was added (old results file?)
2. Equity filter is not being applied correctly during backtest

**Evidence**: Need to check if `cs_momentum_results.csv` is from:
- ✅ New Phase 0 run (Nov 24, 2025) with equity filter
- ❌ Old baseline run (before equity filter)

---

## Recommended Actions (Priority Order)

### 🔴 **CRITICAL: Before Re-Running**

1. **Fix Family Classifications**
   ```python
   # config.py - Add to equity_family_keywords:
   'Natural Resources',  # Includes XLB (Materials)
   'Commodities Focused',  # Includes USO (oil ETF)
   ```
   
   **OR** better: Create explicit inclusion list for sector ETFs
   ```python
   sector_etfs = ['XLE', 'XLF', 'XLI', 'XLK', 'XLB', 'XLP', 'XLU', 'XLV', 'XLY']
   # Force include these regardless of family
   ```

2. **Widen Quantile Buckets**
   ```python
   # config.py line 96-97
   long_quantile: float = 0.67   # Top 33% for long portfolio
   short_quantile: float = 0.33  # Bottom 33% for short portfolio
   ```

3. **Verify Results File is Fresh**
   - Check timestamp on `cs_momentum_results.csv`
   - If before Phase 0 implementation → results are from BASELINE, not Phase 0!

### 🟡 **HIGH PRIORITY: Re-Run Backtest**

4. **Install cvxpy** (already done ✓)

5. **Re-run Phase 0 with fixes**:
   ```bash
   cd D:\REPOSITORY\morias\Quant\strategies\crosssecmom2
   python main.py --step backtest --model supervised_binned
   ```

6. **Compare Results**:
   - Phase 0 (fixed) vs Phase 0 (original)
   - Should see: 20-25 unique tickers, 6-8 positions per side

### 🟢 **VALIDATION**

7. **Print Universe Info During Backtest**
   - Add logging to show eligible universe size at each rebalance
   - Verify equity filter is working: should see "23 tickers" consistently

8. **Sanity Check**:
   - All positions should be from 89-ETF equity list
   - If USO appears → verify it's intentionally included (via sector list)

---

## Expected Results After Fixes

| Metric | Phase 0 (Current) | Phase 0 (Fixed) | Change |
|--------|-------------------|-----------------|--------|
| **Unique Tickers** | 18 | 25-30 | Better coverage |
| **Positions/Side** | 3-4 | 6-8 | Better diversification |
| **Win Rate** | 55.32% | 52-58% | Should hold |
| **Sharpe** | 0.33 | 0.4-0.6 | Better with cvxpy + diversity |
| **Max DD** | -6.10% | -5% to -8% | More positions = more stable |

---

## Decision Tree

```
[Check cs_momentum_results.csv timestamp]
    │
    ├─ Before Phase 0 code changes (old baseline results)
    │   └─ ❌ INVALID - Results are from BASELINE, not Phase 0!
    │       └─ Discard analysis, re-run Phase 0 properly
    │
    └─ After Phase 0 code changes (Nov 24, 2025)
        │
        ├─ USO appears in results
        │   └─ ❌ Equity filter not working correctly
        │       └─ Fix filter logic before re-running
        │
        └─ Only 18 tickers used
            └─ ⚠️ Universe too small
                └─ Fix classifications + widen quantiles
```

---

## Conclusion

**Phase 0 Results May Be Invalid** for these reasons:

1. **Timestamp uncertainty**: Need to verify results are from Phase 0 run, not baseline
2. **Filter inconsistency**: USO appears but should be excluded (unless filter not working)
3. **Insufficient positions**: 3-4 positions suggests configuration problems

**Before Proceeding to Phase 1**:
- ✅ Fix ETF classifications (add Natural Resources, sector ETFs)
- ✅ Widen quantiles to 0.67/0.33 (top/bottom 33%)
- ✅ Verify cvxpy installed
- ✅ Re-run Phase 0 backtest
- ✅ Validate results: 25+ unique tickers, 6-8 positions/side
- ✅ Compare new vs old results

**If new results show Win Rate < 51%**: Equity-only hypothesis may be wrong, need to investigate cross-asset approach.

**If new results maintain 52%+ Win Rate**: Proceed to Phase 1 with confidence.
