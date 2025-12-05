# Alpha Feature Selection Architecture v4

### Step 0 — Feature Classification
- Alpha Features are any feature that is not classified as a risk feature (see feature_risk.md)

## Pipeline Overview

```
thousands of features
    ↓ Hard Gates (5 binary pass/fail)
    ↓ ~800-1,200 survive
    ↓ Composite Scoring (5 metrics)
    ↓ Top 200 features by composite scoring
    ↓ Family Assignment
    ↓ Within-Family Redundancy (ρ ≤ 0.70, max 10/family)
    ↓ Cross-Family Redundancy (ρ ≤ 0.80)
    ↓ Forced Inclusions
    ↓ 50-100 final features → Ensemble Model
```

---

## Hard Gates

Must pass ALL. Failure in any = reject.

| Gate | Threshold |
|------|-----------|
| Data Quality | Coverage ≥ 85%, no degenerate std, <1% outliers beyond 10σ |
| Global IC | Mean |IC| ≥ 2.0% |
| Statistical Significance | t-stat ≥ 2.5 (Newey-West) |
| Sign Stability | IC sign matches global sign in ≥ 3/4 consecutive segments |
| Residual IC | IC ≥ 0.75% after orthogonalizing vs Close%-63 |
| Long-Tail Excess | Top decile excess return > 0% |

---

## Scoring Layer

All metrics percentile-normalized (0-100) within surviving features.

| Metric | Weight | Direction |
|--------|--------|-----------|
|1. Global IC | 30% | Higher = better |
|2. Long-Tail Excess | 25% | Higher = better |
|3. Spread Sharpe (Q5-Q1) | 20% | Higher = better | \
|4. Spread Sharpe = Mean(Q5 - Q1) / Std(Q5 - Q1)
|5. Turnover | 15% | Lower = better (invert percentile) |
|6. Stress IC | 10% | Higher = better |
  Stress Period: any period where benchmark (VT) return < 0%


**Composite Score** = Σ (Percentile × Weight)

**Selection:** Top 200 by composite score before redundancy

---

## Diagnostics Only

Logged but no selection impact:

- Bootstrap stability
- IC decay / autocorrelation
- Mutual information
- Rolling IC timeseries
- Per-fold t-statistics
- Winner skewness

---

## Redundancy Filters

| Filter | Threshold | Constraint |
|--------|-----------|------------|
| Within-Family | ρ ≤ 0.70 | Max 10 per family |
| Cross-Family | ρ ≤ 0.75 | None |

**Method:** Hierarchical clustering within family; keep highest-scoring per cluster. Pairwise elimination across families.

---

## Forced Inclusions

Always include regardless of scoring:

- Close%-63
- Close%-21  
- Close%-126

Subject to cross-family filter but not within-family elimination against each other.

---


## Final Pool Checks

| Check | Target |
|-------|--------|
| Max pairwise correlation | < 0.75 |
| Mean pairwise correlation | < 0.35 |
| Condition number | < 30 |
| All families represented | Yes |
| Core momentum included | Yes |

---

## Quick Reference

```
HARD GATES:
  Data quality: coverage ≥ 85%
  Global IC ≥ 2%
  t-stat ≥ 2.5
  Sign stability ≥ 3/4 segments
  Residual IC ≥ 0.75%
  Long-tail excess > 0%

SCORING:
  Global IC:        30%
  Long-tail:        25%
  Spread Sharpe:    20%
  Turnover (inv):   15%
  Stress IC:        10%

SELECTION: Top 200 by score before redundancy

REDUNDANCY:
  Within-family: ρ ≤ 0.70, max 10
  Cross-family:  ρ ≤ 0.75

```


*End of Document*