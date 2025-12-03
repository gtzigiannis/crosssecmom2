# Forensic Analysis Plan for Cross-Sectional Momentum Strategy

## Executive Summary

This document outlines a comprehensive forensic study to diagnose the strategy's performance characteristics, identify sources of alpha (and alpha decay), and understand the risks from the perspective of a Chief Risk Officer. The study is structured in phases, from basic data extraction to advanced causal analysis.

**Critical Question:** Before analyzing "why" the strategy performed as it did, we must first establish "is there any real signal at all?"

---

## Phase 0: Backtest Integrity Audit (RUN FIRST)

### 0.1 Look-Ahead Bias Audit

**Objective:** Verify no future information leaked into features or signals.

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 0.1.1 | Feature timestamp audit: For each feature, verify it uses only data available at t | Audit report |
| 0.1.2 | Target leakage check: Confirm FwdRet_21 is computed from Close[t+1:t+22], not Close[t:t+21] | Code review |
| 0.1.3 | Panel alignment check: Verify panel.loc[t0] contains only data known at market close on t0 | Spot checks |

### 0.2 Execution Timing Audit

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 0.2.1 | Rebalance timing: Verify signal is generated at close of t0, trade executed at open of t0+1 | Code review |
| 0.2.2 | Holding period alignment: Confirm returns are from t0+1 to t0+21 (not t0 to t0+20) | Calculation audit |
| 0.2.3 | Transaction cost timing: Costs applied at entry AND exit? | Cost model review |

### 0.3 Data Snooping Audit

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 0.3.1 | Parameter count: How many hyperparameters were tuned? | Parameter inventory |
| 0.3.2 | Trials count: How many backtest runs were executed during development? | Estimation |
| 0.3.3 | Bonferroni correction: Adjusted p-value for multiple testing | Statistical adjustment |
| 0.3.4 | OOS validation: Was any true holdout period used? | Design review |

---

## Phase 0B: Is There Any Signal? (Null Hypothesis Tests)

### 0B.1 Random Feature Baseline

**Objective:** Compare strategy to random feature selection (same pipeline, random inputs).

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 0B.1.1 | Run backtest with 15 randomly selected features per window (100 trials) | Distribution of random Sharpe |
| 0B.1.2 | Compute p-value: P(random Sharpe >= actual Sharpe) | Statistical significance |
| 0B.1.3 | Plot actual Sharpe vs. random distribution | Histogram overlay |

### 0B.2 Random Portfolio Baseline

**Objective:** Compare strategy to random ETF selection (same universe, random picks).

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 0B.2.1 | Each period, pick 10 random longs (equal weight), run 1000 trials | Random portfolio distribution |
| 0B.2.2 | Compute percentile of actual strategy in random distribution | Skill vs. luck quantification |
| 0B.2.3 | Random long-short: 10 random longs, 10 random shorts | L/S random baseline |

### 0B.3 Simple Momentum Baseline

**Objective:** Compare to classic momentum (no ML, no feature selection).

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 0B.3.1 | 12-1 month momentum: Rank by ret[-252:-21], long top 10, short bottom 10 | Baseline performance |
| 0B.3.2 | 6-1 month momentum: Rank by ret[-126:-21] | Alternative lookback |
| 0B.3.3 | Compare Sharpe: ML strategy vs. simple momentum vs. VT | Performance ladder |

**Key Question:** Does the complex ML pipeline add any value over simple momentum sorting?

---

## Phase 1: Feature-Level Forensics

### 1.1 Feature Selection & Model Coefficients per Window

**Objective:** Understand which features the model selected at each rebalance, their coefficients (importance), and their Information Coefficients (IC).

**Data Sources:**
- `results_df.attrs['diagnostics']` contains per-window diagnostics
- `feature_details` list in each diagnostics entry has: `feature`, `coefficient`, `ic`, `abs_coef`, `coef_sign`
- `model_fit` dict has: `r_squared`, `model_ic`, `residual_std`, `n_samples`, `n_features`

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 1.1.1 | Extract feature names, coefficients, and IC for each window | CSV: `feature_coefficients_by_window.csv` |
| 1.1.2 | Compute feature stability metrics (selection frequency, IC variance across windows) | CSV: `feature_stability_analysis.csv` |
| 1.1.3 | Identify features with high selection frequency + stable IC vs. one-off selections | Ranked table |
| 1.1.4 | Correlate feature importance with realized period returns | Scatter plots |
| 1.1.5 | Track coefficient sign stability (does feature flip from positive to negative?) | Time series chart |

**Key Questions:**
- Are there "workhorse" features that appear consistently and contribute reliably?
- Are there features that worked historically but have decayed?
- Do coefficient signs flip (suggesting unstable relationship with target)?

---

### 1.2 Feature IC Decay Analysis

**Objective:** Determine if predictive power (IC) is decaying over time.

**Data Sources:**
- `results_df.attrs['attribution']['ic_decay']` DataFrame
- Contains: `period`, `avg_ic`, `median_ic`, `top_ic`, `n_features`
- `.attrs['ic_trend']` has: `slope`, `intercept`, `r_squared`, `p_value`, `interpretation`

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 1.2.1 | Plot IC time series (avg, median, top) over windows | Time series chart |
| 1.2.2 | Compute rolling 12-month IC trend | Trend overlay |
| 1.2.3 | Test for structural break in IC (Chow test or regime detection) | Statistical test |
| 1.2.4 | Decompose IC by feature bucket (momentum, value, quality, macro) | Grouped time series |

---

## Phase 2: Portfolio Holdings Forensics

### 2.1 Position-Level Attribution

**Objective:** Understand which ETFs were selected, when, and how they performed relative to alternatives.

**Data Sources:**
- `results_df['long_tickers']` - Dict of long positions {ticker: weight}
- `results_df['short_tickers']` - Dict of short positions {ticker: weight}
- `universe_metadata` - ETF families, categories, sectors
- `panel_df` - Historical prices and features for all ETFs

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 2.1.1 | Extract all ETFs held long/short in each period | CSV: `holdings_by_period.csv` |
| 2.1.2 | Compute realized return for each held ETF vs. median/mean of non-held ETFs | Comparison table |
| 2.1.3 | Compare Top-5 picks vs. Bottom-5 picks (by score) in each period | Selection quality analysis |
| 2.1.4 | Count "correct picks" (held ETF beat universe median) vs. "wrong picks" | Hit rate by period |
| 2.1.5 | Identify consistently selected ETFs (high selection frequency) | Concentration analysis |
| 2.1.6 | **Rank accuracy:** Spearman correlation between predicted score and realized return, per period OOS | Rank IC time series |
| 2.1.7 | **Top-quintile vs. Bottom-quintile spread:** Average return of top 20% minus bottom 20% | Quintile spread chart |
| 2.1.8 | **Monotonicity check:** Do quintiles 1-5 line up in order? (Q1 > Q2 > Q3 > Q4 > Q5) | Monotonicity score |
| 2.1.9 | **Score dispersion per period:** std(scores) across ETFs | Dispersion time series |
| 2.1.10 | **Conditional performance:** Returns when score dispersion is high vs. low | Confidence-conditional Sharpe |
| 2.1.11 | **Extreme score analysis:** Performance of top-3 and bottom-3 scored ETFs only | Extreme picks performance |

**Key Questions:**
- Did the algorithm pick the right ETFs most of the time?
- How large was the performance gap between selected and non-selected ETFs?
- Are there ETFs that were always/never selected (why)?
- **Is the model ranking correctly, even if portfolio construction is suboptimal?**
- **Does high model confidence predict better performance?**

---

### 2.2 Sector/Theme Attribution

**Objective:** Understand if performance came from specific sectors or themes.

**Data Sources:**
- `universe_metadata['family']` - ETF family/theme classification
- `results_df.attrs['attribution']['sector_attribution']` DataFrame

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 2.2.1 | Compute return contribution by sector/theme | Ranked table |
| 2.2.2 | Plot sector exposure over time (heatmap of weights by sector × date) | Heatmap |
| 2.2.3 | Identify sector concentration risk (top 3 sectors as % of total) | Concentration metric |
| 2.2.4 | Decompose returns into sector selection + intra-sector selection | Brinson attribution |

---

### 2.3 Long vs. Short Side Attribution

**Objective:** Separate long-side skill from short-side skill.

**Data Sources:**
- `results_df['long_ret']` and `results_df['short_ret']`
- `results_df.attrs['attribution']['long_short_attribution']`

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 2.3.1 | Cumulative return chart: Long, Short, L/S, VT | Time series |
| 2.3.2 | Sharpe ratio comparison (Long, Short, L/S) | Performance table |
| 2.3.3 | Win rate comparison (% positive periods) | Comparison bar chart |
| 2.3.4 | Correlation of long returns with VT (is long just beta?) | Scatter + regression |
| 2.3.5 | Drawdown analysis for long vs. short side separately | Max DD comparison |

**Key Questions:**
- Does alpha come from long picks, short picks, or both?
- Is the long side just market beta in disguise?
- Did shorts add or subtract value (accounting for borrow costs)?

---

## Phase 3: Benchmark Comparison Forensics

### 3.1 Daily and Monthly Comparison vs. VT

**Objective:** Rigorous comparison against buy-and-hold VT benchmark.

**Data Sources:**
- `results_df['ls_return']` - Strategy period returns
- `benchmark_daily_returns` from `load_benchmark_returns()`
- `benchmark_period_returns` from `load_benchmark_returns()`

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 3.1.1 | Cumulative return chart: Strategy vs. VT (aligned dates) | Time series |
| 3.1.2 | Rolling 12-month excess return over VT | Rolling excess chart |
| 3.1.3 | Information Ratio (IR) = mean(excess return) / std(tracking error) | Scalar metric |
| 3.1.4 | Upside/Downside capture ratio | Performance attribution |
| 3.1.5 | Monthly return correlation with VT | Beta estimation |
| 3.1.6 | Calendar heatmap of excess returns | Visual pattern detection |

---

### 3.2 Regime-Conditional Performance

**Objective:** Understand when the strategy works (and when it doesn't).

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 3.2.1 | Split performance by VIX regime (high/med/low volatility) | Conditional Sharpe |
| 3.2.2 | Split by market regime (bull/bear/range based on VT returns) | Regime performance table |
| 3.2.3 | Correlate strategy returns with macro factors (credit spreads, yield curve) | Factor analysis |
| 3.2.4 | Event study: performance during drawdowns > 10% | Crisis alpha analysis |

---

## Phase 4: Chief Risk Officer Concerns

### 4.1 Drawdown & Tail Risk Analysis

**Objective:** Understand worst-case scenarios and loss distribution.

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 4.1.1 | Max drawdown analysis (depth, duration, recovery time) | Drawdown table |
| 4.1.2 | Drawdown decomposition (what positions caused the drawdown?) | Attribution during DD |
| 4.1.3 | VaR and CVaR at 95% and 99% confidence | Risk metrics |
| 4.1.4 | Left-tail analysis: what happened in worst 5% of periods? | Extreme loss study |
| 4.1.5 | Correlation of strategy returns with VT during market stress | Crisis correlation |
| 4.1.6 | Time to recovery analysis after drawdowns | Recovery chart |

**CRO Questions:**
- What is the worst case scenario (max monthly loss)?
- How correlated are we to the market during crashes?
- How long does it take to recover from drawdowns?

---

### 4.2 Concentration & Liquidity Risk

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 4.2.1 | Position concentration: Herfindahl index over time | Concentration time series |
| 4.2.2 | Sector concentration: top 3 sectors as % of total | Sector risk |
| 4.2.3 | Liquidity analysis: ADV of held positions | Capacity constraint |
| 4.2.4 | Days to liquidate: (position size / ADV) analysis | Liquidity risk |
| 4.2.5 | Cap violations tracking: how often do constraints bind? | Constraint analysis |

---

### 4.3 Model Risk & Stability

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 4.3.1 | Feature turnover: what % of features change each window? | Stability metric |
| 4.3.2 | Coefficient stability: correlation of coefficients across windows | Model consistency |
| 4.3.3 | Out-of-sample R² vs. in-sample R² | Overfitting detection |
| 4.3.4 | Prediction error analysis: histogram of (predicted - realized) | Error distribution |
| 4.3.5 | Feature selection sensitivity: what happens with ±10% IC threshold? | Robustness test |
| 4.3.6 | **Minimum detectable effect:** Given ~80 windows, what Sharpe is statistically distinguishable from 0? | Power calculation |
| 4.3.7 | **Confidence intervals on all metrics:** Is the Sharpe ratio significantly different from VT's? | Bootstrapped CIs |

**CRO Questions:**
- Is the model stable or does it completely change every month?
- Are we overfitting to in-sample data?
- How sensitive are results to feature selection parameters?
- **Do we have enough data to draw statistically valid conclusions?**

---

### 4.4 Turnover & Transaction Cost Sensitivity

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 4.4.1 | Turnover time series (by long, short, total) | Time series chart |
| 4.4.2 | Transaction cost as % of gross returns | Cost efficiency |
| 4.4.3 | Sensitivity: rerun with 2x and 0.5x transaction costs | Robustness test |
| 4.4.4 | Turnover vs. performance correlation (does high turnover hurt?) | Scatter analysis |
| 4.4.5 | Borrow cost breakdown (which shorts are expensive?) | Cost attribution |

---

## Phase 5: Alpha Source Identification

### 5.1 Factor Attribution (Expanded)

**Objective:** Decompose returns into known factor exposures. If "alpha" is explained by factor tilts, it's not alpha.

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 5.1.1 | Regress strategy returns on Fama-French factors (Mkt-RF, SMB, HML, MOM) | Factor loadings |
| 5.1.2 | Compute residual alpha after controlling for factors | Unexplained alpha |
| 5.1.3 | Time-varying factor exposures (rolling regression) | Exposure time series |
| 5.1.4 | Factor contribution to returns | Attribution breakdown |
| 5.1.5 | **ETF flow factor:** Regress on aggregate ETF flows (from yfinance volume data) | Flow exposure |
| 5.1.6 | **Sector momentum factor:** Are you just riding sector rotation? | Sector momentum beta |
| 5.1.7 | **Dollar factor (DXY):** Regress on DXY (currency exposure for international ETFs) | Currency beta |
| 5.1.8 | **Volatility factor:** Are you systematically long/short vol? (VIX beta) | Vol exposure |
| 5.1.9 | **Macro surprise factor:** Performance around FOMC, CPI, NFP dates | Event-conditional returns |

**Data Requirements:**
- DXY (US Dollar Index) - **Add to yfinance download and store in raw data**
- FOMC meeting dates, CPI release dates, NFP release dates - **Download from FRED or manual list**
- ETF flow data (approximated from volume × price changes)

---

### 5.2 Timing vs. Selection

**Objective:** Is skill from picking the right assets or from market timing?

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 5.2.1 | Compute average position size over time | Exposure chart |
| 5.2.2 | Conditional exposure analysis: does cash position increase before drawdowns? | Timing skill |
| 5.2.3 | Selection skill: excess return of picks vs. random selection | Stock picking alpha |

---

### 5.3 Counterfactual Analysis ("What Would Have Worked?")

**Objective:** Understand the upper bound on performance and whether the model has any directional skill.

**Analysis Tasks:**
| # | Task | Output |
|---|------|--------|
| 5.3.1 | **Perfect foresight portfolio:** What if you knew next-period returns? | Maximum achievable Sharpe |
| 5.3.2 | **Perfect foresight feature:** Which single feature would have worked best (in hindsight)? | Best feature analysis |
| 5.3.3 | **Inverse strategy:** Performance if you did the opposite of predictions | Anti-signal check |

**Key Insight:** If the inverse strategy outperforms, your model is consistently wrong (which is actually useful—flip the sign!).

---

## Phase 6: Implementation Checklist

### Outputs to Generate

| Category | Output File | Format |
|----------|-------------|--------|
| Integrity | `backtest_audit_report.md` | Report |
| Null Tests | `random_baseline_comparison.csv` | CSV |
| Null Tests | `momentum_baseline_comparison.csv` | CSV |
| Features | `feature_coefficients_by_window.csv` | CSV |
| Features | `feature_stability_analysis.csv` | CSV |
| Features | `ic_decay_analysis.png` | Chart |
| Holdings | `holdings_by_period.csv` | CSV |
| Holdings | `selection_quality_analysis.csv` | CSV |
| Holdings | `quintile_analysis.csv` | CSV |
| Holdings | `rank_accuracy_per_period.csv` | CSV |
| Holdings | `confidence_conditional_performance.csv` | CSV |
| Holdings | `sector_attribution.csv` | CSV |
| Long/Short | `long_short_decomposition.csv` | CSV |
| Long/Short | `cumulative_returns_chart.png` | Chart |
| Benchmark | `vt_comparison_daily.csv` | CSV |
| Benchmark | `excess_return_analysis.csv` | CSV |
| Risk | `drawdown_analysis.csv` | CSV |
| Risk | `tail_risk_metrics.csv` | CSV |
| Risk | `concentration_analysis.csv` | CSV |
| Risk | `statistical_power_analysis.csv` | CSV |
| Model | `model_stability_metrics.csv` | CSV |
| Costs | `transaction_cost_sensitivity.csv` | CSV |
| Factors | `factor_attribution.csv` | CSV |
| Factors | `currency_exposure.csv` | CSV |
| Factors | `macro_event_performance.csv` | CSV |
| Counterfactual | `perfect_foresight_analysis.csv` | CSV |
| Counterfactual | `inverse_strategy_results.csv` | CSV |

---

## Priority Implementation Order

**CRITICAL PATH (Run First - Invalidates Everything If Failed):**
1. **Phase 0:** Backtest integrity audit (look-ahead, execution timing, data snooping)
2. **Phase 0B:** Null hypothesis tests (random baselines, simple momentum)

**If Phase 0/0B pass, proceed:**
3. **Phase 2.1.6-2.1.8:** Ranking skill (is the model ranking correctly?)
4. **Phase 2.1.9-2.1.11:** Confidence analysis (does model know when it's right?)
5. **Phase 5.3:** Counterfactual analysis (perfect foresight, inverse strategy)
6. **Phase 5.1:** Factor attribution (is alpha real or hidden factor exposure?)
7. **Phase 4.3.6-4.3.7:** Statistical power (do we have enough data?)

**Secondary analyses (refine understanding):**
8. **Phase 1.1:** Feature coefficients and IC per window
9. **Phase 2.3:** Long vs. Short attribution
10. **Phase 3.1:** VT benchmark comparison
11. **Phase 4.1:** Drawdown and tail risk

---

## Script Implementation Requirements

A Python script `forensic_analysis.py` should be created that:
1. Loads the latest backtest results from `results_df` with its `.attrs`
2. Loads panel data and universe metadata
3. Implements each analysis task as a separate function
4. Generates all output files in a dedicated `forensic_outputs/` directory
5. Creates a summary report `forensic_summary.md` with key findings

---

## Key Metrics Summary Table

| Metric | Description | Target | Critical? |
|--------|-------------|--------|-----------|
| Random Baseline p-value | P(random Sharpe >= actual) | < 0.05 | **YES** |
| Momentum Baseline Excess | ML Sharpe - Momentum Sharpe | > 0 | **YES** |
| Rank IC (OOS) | Spearman(score, realized return) | > 0.05 | **YES** |
| Quintile Monotonicity | Q1 > Q2 > Q3 > Q4 > Q5? | Yes | **YES** |
| Inverse Strategy Sharpe | Performance of flipped signals | < 0 | **YES** |
| Win Rate | % of periods with positive L/S return | > 55% | No |
| Information Ratio | Mean excess return / Tracking error | > 0.5 | No |
| Max Drawdown | Largest peak-to-trough decline | < 25% | No |
| IC Decay Slope | Trend in average IC over time | > -0.001 | No |
| Feature Stability | % of features appearing in >50% of windows | > 30% | No |
| Factor Alpha | Residual after FF4 regression | > 0% annualized | No |

---

## Future Work: Not Yet Implemented

The following analyses are planned but not yet implemented in the current forensic study module:

### F.1 Holding Period Sensitivity

**Objective:** Determine if 21-day holding period is optimal or arbitrary.

| # | Task | Output |
|---|------|--------|
| F.1.1 | Re-run backtest with holding periods: 7, 14, 21, 28, 42 days | Performance comparison table |
| F.1.2 | Plot Sharpe ratio vs. holding period | Sensitivity curve |
| F.1.3 | Analyze turnover vs. holding period trade-off | Cost-adjusted returns |
| F.1.4 | Test for statistical difference between holding periods | p-values for pairwise comparisons |

### F.2 Alpha Decay Rate Analysis

**Objective:** Measure how quickly the signal loses predictive power after formation.

| # | Task | Output |
|---|------|--------|
| F.2.1 | Compute cumulative IC at days 1, 3, 5, 7, 10, 14, 21 post-formation | IC decay curve |
| F.2.2 | Fit exponential decay model: IC(t) = IC(0) * exp(-λt) | Decay rate λ, half-life |
| F.2.3 | Compare decay rate in high vs. low volatility periods | Regime-conditional decay |
| F.2.4 | Analyze "stale signal" periods where α already exhausted | Entry timing optimization |

### F.3 Performance by ETF Liquidity Tier

**Objective:** Understand if alpha concentrates in illiquid (harder to trade) ETFs.

| # | Task | Output |
|---|------|--------|
| F.3.1 | Tier ETFs by ADV_63: Top 25%, Middle 50%, Bottom 25% | Liquidity buckets |
| F.3.2 | Compute L/S return by liquidity tier | Performance attribution |
| F.3.3 | Analyze % of picks from each tier over time | Liquidity drift analysis |
| F.3.4 | Estimate realistic slippage/market impact by tier | Execution feasibility |
| F.3.5 | Compute "executable alpha" after realistic transaction costs | Net-of-friction returns |

### F.4 Kill Switch Criteria

**Objective:** Define quantitative rules for when to halt or reduce strategy exposure.

| # | Task | Output |
|---|------|--------|
| F.4.1 | Define drawdown threshold (e.g., -15% trailing 63 days) | Drawdown trigger |
| F.4.2 | Define IC degradation threshold (e.g., 5 consecutive IC < 0) | Signal quality trigger |
| F.4.3 | Define regime indicator threshold (e.g., VIX > 30) | Market condition trigger |
| F.4.4 | Backtest kill switch rules: What would performance look like? | Historical simulation |
| F.4.5 | Define position reduction ladder (50%, 75%, 100% hedge) | Graduated response protocol |

### F.5 Hidden Look-Ahead Detection

**Objective:** Deep audit for subtle look-ahead bias in feature engineering.

| # | Task | Output |
|---|------|--------|
| F.5.1 | Trace each feature's computation to raw data sources | Data lineage diagram |
| F.5.2 | Check for inadvertent use of same-day close in t0 features | Timestamp audit |
| F.5.3 | Verify panel lag: Is panel.loc[t0] truly known at t0 close? | Lag verification |
| F.5.4 | Check index rebalancing overlap with our rebalance dates | Index reconstitution bias |
| F.5.5 | Audit any external data sources for publication lag | Data availability audit |

---

## Additional CRO Questions to Address

1. **What is the capacity of this strategy?** (How much AUM before we move markets?)
2. **What is the correlation with other strategies in the fund?** (Diversification benefit)
3. **What happens if a key data source fails?** (Operational risk)
4. **What is the sensitivity to holding period?** (21 days is arbitrary - what about 14 or 42?)
5. **What is the decay rate of alpha after formation?** (Signal half-life analysis)
6. **What are the worst-case borrow costs in stressed markets?** (Short squeeze risk)
7. **How does performance vary by ETF AUM/liquidity tier?** (Execution feasibility)
8. **What is the impact of late rebalancing?** (Implementation shortfall)
9. **What regime indicators would trigger strategy shutdown?** (Kill switch criteria)
10. **What is the lookback bias in feature engineering?** (Any inadvertent lookahead?)

---

*Document created: December 3, 2025*
*Updated: December 3, 2025 - Added Phases 0, 0B, expanded 2.1, 4.3, 5.1, 5.3, Future Work F.1-F.5*
*Author: AI Assistant (Claude)*
*Strategy: crosssecmom2*
