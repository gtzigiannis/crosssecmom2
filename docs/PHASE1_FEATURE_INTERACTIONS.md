# Phase 1: Exhaustive Interaction Feature Engineering

**Date**: November 24, 2025  
**Status**: Implementation In Progress  
**Objective**: Generate ALL mathematically valid interaction features systematically

---

## Philosophy

**NOT heuristic-based**: We do NOT handpick features based on "gut feel"  
**EXHAUSTIVE**: Perform combinatronics and let statistical tests decide  
**SYSTEMATIC**: Use mathematical rules to ensure validity (no Inf, minimize NaN)

---

## Base Feature Categories

From current `cs_momentum_features.parquet` (96 features):

### 1. Momentum Features (13 features)
- `Close%-1`, `Close%-2`, `Close%-3`, `Close%-5`, `Close%-10`
- `Close%-21`, `Close%-42`, `Close%-63`, `Close%-126`, `Close%-252`
- `Close_Mom5`, `Close_Mom10`, `Close_Mom21`, `Close_Mom42`, `Close_Mom63`

**Mathematical Property**: Can be positive or negative (zero-centered)

### 2. Volatility Features (6 features)
- `Close_std5`, `Close_std10`, `Close_std21`, `Close_std42`, `Close_std63`, `Close_std126`

**Mathematical Property**: Always positive (>= 0)

### 3. Oscillator Features (6 features)
- `Close_RSI14`, `Close_RSI21`, `Close_RSI42`
- `Close_WilliamsR14`, `Close_WilliamsR21`, `Close_WilliamsR63`

**Mathematical Property**: Bounded [0, 100] for RSI, [-100, 0] for Williams%R

### 4. Trend Features (14 features)
- Moving Averages: `Close_MA5`, `Close_MA10`, `Close_MA21`, `Close_MA42`, `Close_MA63`, `Close_MA126`, `Close_MA200`
- Exponential MA: `Close_EMA5`, `Close_EMA10`, `Close_EMA21`, `Close_EMA42`, `Close_EMA63`, `Close_EMA126`
- Bollinger Bands: `Close_BollUp21`, `Close_BollLo21`, `Close_BollUp50`, `Close_BollLo50`

**Mathematical Property**: Price-level (same scale as Close), always positive

### 5. MACD Family (7 features)
- `Close_MACD`, `Close_MACD_Signl`, `Close_MACD_Histo`, `Close_MACD_HistSl`
- `Close_MACD_Mom`, `Close_MACD_Xover`, `Close_MACD_SignDir`

**Mathematical Property**: Zero-centered, can be positive or negative

### 6. Higher Moments (8 features)
- Skewness: `Close_skew21`, `Close_skew42`, `Close_skew63`, `Close_skew126`
- Kurtosis: `Close_kurt21`, `Close_kurt42`, `Close_kurt63`, `Close_kurt126`

**Mathematical Property**: Unbounded, can be positive or negative

### 7. Hurst Exponent (3 features)
- `Close_Hurst21`, `Close_Hurst63`, `Close_Hurst126`

**Mathematical Property**: Bounded [0, 2], typically [0.3, 0.7]

### 8. Drawdown Features (2 features)
- `Close_DD20`, `Close_DD60`

**Mathematical Property**: Always negative (<= 0), represents loss

### 9. Shock Features (1 feature)
- `Close_Ret1dZ` (1-day return / 60-day vol)

**Mathematical Property**: Z-score, zero-centered

### 10. Liquidity Features (2 features)
- `ADV_63` (Average Dollar Volume)
- `Close_ATR14` (Average True Range)

**Mathematical Property**: Always positive (> 0)

### 11. Macro Features (6 features)
- `vix_level`, `vix_z_1y` (VIX level and z-score)
- `yc_slope` (10Y - 2Y yield)
- `short_rate` (3-month T-bill)
- `credit_proxy_20` (HYG - LQD spread)
- `crash_flag`, `meltup_flag`, `high_vol`, `low_vol` (regime indicators)

**Mathematical Property**: Mixed (VIX > 0, slopes can be +/-)

### 12. Relative Features (6 features)
- `Rel5_vs_VT`, `Rel20_vs_VT`, `Rel60_vs_VT`
- `Rel5_vs_Basket`, `Rel20_vs_Basket`, `Rel60_vs_Basket`

**Mathematical Property**: Zero-centered (relative returns)

### 13. Correlation Features (2 features)
- `Corr20_VT`, `Corr20_BNDW`

**Mathematical Property**: Bounded [-1, +1]

### 14. Asset Type Flags (4 features)
- `is_equity`, `is_bond`, `is_real_asset`, `is_sector`

**Mathematical Property**: Binary {0, 1}

### 15. Lagged Returns (5 features)
- `Close_lag1`, `Close_lag2`, `Close_lag3`, `Close_lag5`, `Close_lag10`

**Mathematical Property**: Zero-centered

---

## Interaction Types

### Type 1: Multiplicative Interactions (Products)

**Formula**: `feature_A × feature_B`

**Purpose**: Capture joint effects

**Generation Rules**:
1. **Momentum × Momentum** (all pairs): Capture momentum interactions across horizons
   - Example: `Close%-21 × Close%-63` = short-term × medium-term momentum
   - Count: C(13, 2) = 78 combinations

2. **Momentum × Volatility** (all pairs): Risk-adjusted momentum
   - Example: `Close%-21 × Close_std21` = momentum with volatility scaling
   - Count: 13 × 6 = 78 combinations

3. **Momentum × Macro** (all pairs): Regime-sensitive momentum
   - Example: `Close%-63 × vix_z_1y` = momentum in high/low volatility
   - Count: 13 × 6 = 78 combinations

4. **Momentum × Oscillator** (all pairs): Overbought/oversold momentum
   - Example: `Close%-21 × Close_RSI14` = momentum with mean-reversion signal
   - Count: 13 × 6 = 78 combinations

5. **Volatility × Macro** (all pairs): Volatility regime interactions
   - Example: `Close_std21 × vix_level` = idiosyncratic vol × market vol
   - Count: 6 × 6 = 36 combinations

6. **Momentum × Relative Returns** (all pairs): Cross-sectional momentum strength
   - Example: `Close%-21 × Rel20_vs_VT` = momentum vs market
   - Count: 13 × 6 = 78 combinations

**Total Multiplicative**: ~426 features

### Type 2: Ratio Interactions (Division)

**Formula**: `feature_A / feature_B` where `feature_B > 0` always

**Purpose**: Risk-adjusted returns, relative scaling

**Generation Rules**:
1. **Momentum / Volatility** (all pairs): Sharpe-like ratios
   - Example: `Close%-21 / Close_std21` = return per unit risk
   - Count: 13 × 6 = 78 combinations

2. **Momentum / ADV** (momentum / liquidity): Return per liquidity unit
   - Example: `Close%-63 / ADV_63` = momentum adjusted for liquidity
   - Count: 13 × 1 = 13 combinations

3. **Volatility / Volatility** (different horizons): Vol regime shifts
   - Example: `Close_std21 / Close_std63` = short-term vs long-term vol
   - Count: C(6, 2) = 15 combinations

**Total Ratio**: ~106 features

### Type 3: Difference Interactions (Subtraction)

**Formula**: `feature_A - feature_B`

**Purpose**: Acceleration, regime shifts, spread changes

**Generation Rules**:
1. **Momentum Acceleration** (adjacent horizons): Rate of change in momentum
   - Example: `Close%-21 - Close%-63` = momentum acceleration
   - Count: 12 pairs (1-2, 2-3, 3-5, 5-10, 10-21, 21-42, 42-63, 63-126, 126-252)

2. **Volatility Changes** (adjacent horizons): Vol regime shifts
   - Example: `Close_std21 - Close_std63` = vol increasing/decreasing
   - Count: 5 pairs (5-10, 10-21, 21-42, 42-63, 63-126)

3. **Oscillator Spreads** (RSI - Williams%R adjusted): Divergence signals
   - Example: `Close_RSI14 - (100 + Close_WilliamsR14)` = RSI vs Williams
   - Count: 9 combinations (3 RSI × 3 Williams)

**Total Difference**: ~26 features

### Type 4: Polynomial Transformations

**Formula**: `feature^n` where n ∈ {2, 3}

**Purpose**: Capture non-linear effects (convexity, tail behavior)

**Generation Rules**:
1. **Momentum Squared**: Emphasize extreme movers
   - Example: `(Close%-21)^2` = convex momentum
   - Count: 13 features

2. **Momentum Cubed**: Preserve sign, emphasize tails
   - Example: `(Close%-21)^3` = asymmetric tail emphasis
   - Count: 13 features

3. **Volatility Squared**: Vol-of-vol proxy
   - Example: `(Close_std21)^2` = variance
   - Count: 6 features

**Total Polynomial**: ~32 features

### Type 5: Regime-Conditional Features

**Formula**: `feature × regime_indicator`

**Purpose**: Let model learn regime-specific feature behavior

**Generation Rules**:
1. **Momentum × Regime Flags** (all momentum × all regime flags):
   - Example: `Close%-21 × high_vol` = momentum in high volatility
   - Count: 13 momentum × 4 regime flags = 52 combinations

2. **Volatility × Regime Flags**:
   - Example: `Close_std21 × crash_flag` = volatility during crashes
   - Count: 6 volatility × 4 regime flags = 24 combinations

**Total Regime-Conditional**: ~76 features

---

## Total Feature Count

| Category | Count |
|----------|-------|
| Base Features | 96 |
| Multiplicative Interactions | 426 |
| Ratio Interactions | 106 |
| Difference Interactions | 26 |
| Polynomial Transformations | 32 |
| Regime-Conditional Features | 76 |
| **GRAND TOTAL** | **762** |

---

## Implementation Rules

### 1. NaN Handling
- **Propagate NaN**: If either operand is NaN, result is NaN
- **Track NaN %**: Log warning if feature has > 50% NaN

### 2. Inf Handling
- **Division by zero**: Replace with NaN (not Inf)
- **Overflow**: Clip extreme values at ±1e10

### 3. Feature Naming Convention
- Multiplicative: `{base1}_x_{base2}`
- Ratio: `{base1}_div_{base2}`
- Difference: `{base1}_minus_{base2}`
- Polynomial: `{base}_sq` or `{base}_cb`
- Regime-conditional: `{base}_in_{regime}`

### 4. Validation Checks
- No Inf values in final panel
- No features with > 80% NaN
- All numeric features are float32

---

## Mathematical Justification

### Why Multiplicative?
- **Joint effects**: Momentum × Volatility captures "strong momentum with low risk"
- **Regime sensitivity**: Momentum × VIX captures "momentum in calm vs turbulent markets"

### Why Ratio?
- **Risk adjustment**: Mom / Vol is Sharpe-like ratio (return per unit risk)
- **Relative scaling**: Vol / Vol ratio detects regime shifts

### Why Difference?
- **Acceleration**: Mom21 - Mom63 = momentum is accelerating/decelerating
- **Spread changes**: Captures divergence in related indicators

### Why Polynomial?
- **Non-linearity**: Momentum^2 emphasizes extreme movers (both +/- extremes matter)
- **Tail behavior**: Momentum^3 preserves sign but emphasizes tails

### Why Regime-Conditional?
- **Let model learn**: Model decides if momentum behaves differently in high-vol vs low-vol
- **NOT for directionality**: Still ranks cross-sectionally (does NOT switch to long-only)

---

## Next Steps

1. ✅ Document interaction types and counts
2. ⏳ Implement `generate_interaction_features()` function
3. ⏳ Integrate into `run_feature_engineering()`
4. ⏳ Test with full backtest
5. ⏳ Phase 2: Let statistical tests (IC + MI) filter to 15-25 features

---

**Last Updated**: November 24, 2025  
**Author**: Morias Lab Research  
**Version**: 1.0 (Initial Design)
