## Risk Feature Selection Pipeline (with Targets Defined)

### 0. Setup

- Scope: `role = "risk_control"` features (volatility, beta, liquidity, volatility-regime, etc.).
- Rebalancing: portfolio rebalances **monthly**, but risk is evaluated at **short horizons** (daily / weekly) so it can react to spikes and improve estimation.
- Time segmentation: split full history into **4 equal-length contiguous segments** (S1–S4) for stability checks.

#### Risk targets

**CRITICAL: All risk targets are computed PER TICKER (per-asset), not on a flattened panel.**
This ensures that forward-looking windows don't cross ticker boundaries.

For *volatility / tail* risk features:

- `RV_1d(t, ticker)`: next-day realized volatility for that asset
  - `RV_1d(t) = |r_{t+1}| × sqrt(252)` for that ticker's return series.
- `RV_5d(t, ticker)`: 5-day forward realized volatility for that asset
  - `RV_5d(t) = sqrt(mean(r_{t+i}^2 for i=1..5)) × sqrt(252)` for that ticker.

For *tail events*:

- Estimate rolling volatility `σ_20d(t)` from that ticker's last 20 days.
- Define binary tail event (per-ticker):
  - `TailEvent_1d(t) = 1[ r_{t+1} < -k · σ_20d(t) ]`, with `k ≈ 2.5`.
  - Note: LEFT tail only (large negative returns).

For *liquidity* features:

- Define `LiqTarget(t)` (choose one):
  - Bid–ask spread (if available), or
  - `log(ADV_20d)` = log(20-day average daily dollar volume), or
  - 20-day turnover (volume / shares outstanding).

Vol / regime-of-vol features are scored vs `RV_1d`, `RV_5d`, `TailEvent_1d`.  
Liquidity features are scored vs `LiqTarget` (not vs returns or RV directly).

---

### Step 1 – Data Quality & Cleaning

For all risk features:

- Enforce minimum coverage (e.g. ≥ 80–90% of (date, asset) cells).
- Drop near-constant / degenerate features (very low variance).
- Winsorize or clip extreme outliers to avoid instability.
- Basic structural sanity (no impossible jumps due to data errors).

---

### Step 1.5 – Sign Stability Hard Gate

Risk control features must have a **consistent directional relationship** with their risk targets across time. A feature that flips sign across market regimes is unreliable as a risk control.

For each risk feature and its appropriate target (`RV_1d`/`RV_5d` for vol features, `LiqTarget` for liquidity):

1. **Global relevance gate**
   - Compute global (pooled) Spearman correlation between feature and target.
   - If |global_corr| < 0.05, reject (too weak/noisy to be a risk control).

2. **Define reference sign**
   - Reference sign = sign of the global correlation (positive or negative).

3. **Segment analysis**
   - Split history into 4 equal contiguous segments (S1–S4).
   - Compute correlation in each segment.

4. **Sign-stability veto**
   - For each segment:
     - If segment correlation has the **opposite sign** to the reference AND |segment_corr| > 0.05 (noise band), **reject the feature**.
   - Accept only features where all segments are either:
     - Same sign as global, or
     - Very close to zero (within the ±0.05 noise band).

5. **Only survivors proceed** to risk relevance scoring (Step 2).

---

### Step 2 – Risk Relevance Scoring (vs Risk Targets)

For each candidate risk feature:

1. **Choose appropriate target(s)** based on family:
   - Volatility / vol-regime features → `RV_1d`, `RV_5d`, `TailEvent_1d`.
   - Liquidity features → `LiqTarget`.

2. **Compute risk relevance metrics** (per feature):

   - **Vol/variance relevance**
     - Corr or rank-corr with `RV_1d` and `RV_5d` (average them into a single score), or simple regression `R²`.
   - **Tail discrimination** (for vol / tail features)
     - Binary classification vs `TailEvent_1d` using the feature as a score.
     - Measure: ROC AUC (higher AUC = better at identifying tail days).
   - **Stability across time**
     - Recompute the above metrics in each of the 4 segments S1–S4.
     - Stability components:
       - Sign consistency (same sign as global in ≥ 3/4 segments).
       - Magnitude stability (segment-wise metrics not wildly different).

3. **Convert to percentiles** across all risk features:

   - `VolScore`  = percentile of |vol/variance relevance|.
   - `TailScore` = percentile of AUC / tail discrimination (0 for liquidity-only features if not used).
   - `StabScore` = percentile of stability measure.

4. **Combine into a single `RiskScore`** (weights can be tuned):

   - Example:
     - `RiskScore = 0.4 · VolScore + 0.4 · TailScore + 0.2 · StabScore`.
   - For liquidity features, you can replace `VolScore` / `TailScore` by relevance to `LiqTarget`, keeping the same structure.

---

### Step 3 – Redundancy Filter (Ordered by RiskScore)

Within each risk family (vol, beta, liquidity, regime-of-vol, etc.):

- Sort features by `RiskScore` from highest to lowest.
- Traverse in that order and build a selected set:
  - For each new candidate feature, check redundancy vs already-selected ones:
    - Very similar behaviour vs risk targets (e.g. very close correlations to `RV_1d/5d` and `TailEvent_1d`), and/or
    - Very high feature-value correlation (|ρ| extremely high) with a higher-scored feature.
  - If redundant, drop; otherwise, keep.

Result: within each family you retain non-redundant features with high `RiskScore`.

---

### Step 4 – Final Selection by RiskScore

- For each family, keep the top `K` non-redundant features by `RiskScore` (e.g. `K = 3–5`).
- Optionally enforce an overall cap (e.g. 10–30 risk features in total).
- These selected risk features feed the **risk model** (vol forecasts, tail-risk indicators, liquidity constraints), separate from the alpha-feature selection and alpha scoring.
