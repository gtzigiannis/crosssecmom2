# Feature Selection Pipeline

**Goal: Find features whose signal is PERSISTENT and REAL, not noise that passes statistical tests.**

## Core Principles

- No hand-crafted economic priors.
- Cross-temporal univariate stability (Newey-West significance + direction consistency + block-level persistence).
- Parent pre-filtering for interactions (drop if both parents are clearly bad).
- Redundancy reduction.
- K-fold **non-overlapping block** CV with LassoCV (single model, no ensemble complexity).

---

## 1. Per-Formation Walk-Forward Step

For each walk-forward step with:

- `formation_period` (ending at `T_form`),
- `training_period` (immediately after formation),
- `test_period` (OOS),

run the following pipeline.

---

### 1.1 Stage A – Univariate Cross-Temporal Stability

Goal: identify features with persistent sign, sufficient IC magnitude, statistical significance, and block-level persistence across time blocks within the formation period. Base and interaction features are treated separately with different thresholds.

#### 1.1.1 Base Features

Let `B` be the set of base (non-interaction) features.

1. Determine which global blocks fall entirely inside the current `formation_period`.
2. For each base feature `b ∈ B`:
   - Restrict to the subset of blocks belonging to the formation period.
   - Compute:
     - `mean_IC_b` = mean of `IC[b, k]` over formation blocks.
     - `sign_global_b` = `sign(mean_IC_b)`.
     - **Direction Consistency**: `frac_same_sign_b` = fraction of blocks where `sign(IC[b, k]) == sign_global_b`.
     - Newey-West t-stat on the weighted mean IC.
3. Across all base features, apply FDR correction (Benjamini-Hochberg) to p-values from Newey-West t-stats.
4. **Block-Level Persistence Filter** (Gate 4):
   - For each feature, count blocks where:
     - `sign(IC_block) == sign(mean_IC)`, AND
     - `|IC_block| >= IC_block_min` OR `|t_block| >= t_block_min`
   - Feature must pass at least `m_base` blocks (capped at `n_blocks` if fewer blocks available).
5. Define **base feature survivors** `S_A_base`:

   A base feature `b` belongs to `S_A_base` if all of:

   - FDR-corrected p-value ≤ `alpha_base` (from Newey-West).
   - `abs(mean_IC_b) ≥ IC_min_base`.
   - `frac_same_sign_b ≥ direction_consistency_min` (e.g. 0.6 = 60% of blocks have same sign as global mean).
   - Passes block-level persistence filter (≥ `m_base` blocks with correct sign and sufficient magnitude).

#### 1.1.2 Interaction Features - Parent Pre-Filtering

Let `I` be the set of interaction features, each of the form `i = A × B` with parents `A` and `B` in the base feature set.

Stage A for interactions uses **two levels** of filtering:

1. Parent quality test (pre-filter).
2. Interaction quality test (stricter thresholds).

**Define "parent clearly bad"** as:

- Parent `P` has:
  - FDR-corrected p-value > `alpha_parent_loose` (e.g. 0.50), **and**
  - `abs(mean_IC_P) < IC_min_parent_loose` (e.g. 0.01).

For each interaction `i = A × B`:

1. Get Stage A stats for the parents `A` and `B` using the same formation blocks.
2. **If both parents are clearly bad** → **drop the interaction immediately**.
   - This reduces noise (false positives) and computation time.
3. For remaining interactions, compute interaction Stage A stats:
   - `mean_IC_i`, direction consistency, Newey-West t-stat.
4. Across all such interactions, apply FDR correction.
5. Define **interaction survivors** `S_A_int` using **stricter thresholds** than base features:
   - FDR-corrected p-value ≤ `alpha_int` with `alpha_int < alpha_base`.
   - `frac_same_sign_i ≥ direction_consistency_min_int` (stricter than base).
   - `abs(mean_IC_i) ≥ IC_min_int` with `IC_min_int > IC_min_base`.

6. Stage A survivors: `S_A = S_A_base ∪ S_A_int`

---

### 1.2 Redundancy Reduction (After Stage A, Before Multivariate)

Goal: reduce redundancy among Stage A survivors so that multivariate selection operates on a compact, non-collinear set.

1. Apply bucket-aware correlation-based redundancy filter.
2. The resulting set is: `S_AR` (Stage A survivors after Redundancy filtering).

`S_AR` is the input feature set for multivariate selection.

---

### 1.3 Stage B – K-Fold Non-Overlapping Block CV with LassoCV

Goal: identify features that are robustly important in a multivariate model using K-fold **non-overlapping block** cross-validation with LassoCV.

**Key change**: Each fold fits LassoCV only on that block's data (NOT leave-one-out). This provides truly independent stability assessment.

#### 1.3.1 K-Fold Block Partition

On the combined `formation_period ∪ training_period`:

1. Partition the time axis into `K` contiguous blocks (non-overlapping), preserving order:
   - `K = 5`, each block ≈ `n_samples / K` samples.

2. For each block `k ∈ {1, …, K}`:
   - Fit LassoCV **only on block k's data** (not leave-one-out).
   - This ensures each block provides an independent feature selection signal.

#### 1.3.2 LassoCV Feature Selection

For each block `k = 1..K`:

1. Build `X_block_k`, `y_block_k` using only block k's samples and features `S_AR`.
2. Fit LassoCV on `(X_block_k, y_block_k)`:
   - Cross-validation within the block to select optimal regularization.
3. Record selected features: `selected_k[j] = 1` if coefficient ≠ 0 for feature `j`.

After all K blocks, compute **selection frequency** for each feature `j`:

- `π_j = (1 / K) * Σ_{k=1..K} selected_k[j]`

Define **stable feature set**:

- `S_stable = { j ∈ S_AR : π_j ≥ π_threshold }` (e.g. `π_threshold = 0.6` means selected in ≥60% of blocks)

Define: `S_final = S_stable`

---

### 1.4 Final Per-Window Model Fit and OOS Test

For the current formation–training–test step:

1. **Final feature set for prediction**: Use `S_final` from Stage B.

2. **Training**:
   - Build `X_train`, `y_train` using only the `training_period` and features in `S_final`.
   - Fit the final predictive model (Ridge) on `(X_train, y_train)`.
   - **No per-window IC re-ranking/truncation** - features in `S_final` are used directly without IC gating.

3. **Test**:
   - Build `X_test` on the `test_period` using features in `S_final`.
   - Apply the trained model to generate predictions and trading signals.
   - Compute performance metrics (returns, Sharpe, drawdown, etc.) on the test period.

4. **Walk-forward**:
   - Move the formation–training–test window forward in time.
   - Repeat the entire process.

---

## 2. Key Properties

- **No economic priors** - everything is data-driven.
- **Four-gate filtering** - FDR significance, IC floor, direction consistency, and block-level persistence.
- **Block-level persistence** - features must show signal in multiple independent time blocks, not just pooled.
- **Parent pre-filtering** - interactions dropped if both parents are clearly bad (reduces noise and computation).
- **Stricter thresholds for interactions** - interactions are more likely to be spurious.
- **Non-overlapping block LassoCV** - each block provides independent stability signal (no overlapping training sets).
- **No per-window IC re-ranking** - once features pass Stage A + redundancy + Stage B, they're used directly.
- **No ensemble complexity** - single model family (LassoCV for selection, Ridge for final fit).

---

## 3. Configuration Parameters

### Stage A - Univariate

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha_base` | FDR threshold for base features | 0.10 |
| `alpha_int` | FDR threshold for interactions (stricter) | 0.05 |
| `direction_consistency_min` | Min fraction of blocks with same sign (base) | 0.60 |
| `direction_consistency_min_int` | Min fraction for interactions (stricter) | 0.80 |
| `IC_min_base` | Min absolute mean IC for base features | 0.02 |
| `IC_min_int` | Min absolute mean IC for interactions | 0.03 |
| `alpha_parent_loose` | FDR threshold for "clearly bad" parent | 0.50 |
| `IC_min_parent_loose` | IC threshold for "clearly bad" parent | 0.01 |

### Stage A - Block-Level Persistence (Gate 4)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `block_persistence_enabled` | Enable block-level persistence filter | True |
| `block_persistence_m_base` | Min blocks feature must pass | 3 (capped at n_blocks) |
| `block_persistence_ic_min` | Min |IC| per block | 0.01 |
| `block_persistence_t_min` | Min |t-stat| per block | 1.5 |

### Stage B - Multivariate

| Parameter | Description | Default |
|-----------|-------------|---------|
| `kfold_n_splits` | Number of CV blocks | 5 |
| `kfold_selection_threshold` | Min selection frequency (π) to survive | 0.60 |
| `kfold_use_block_only` | Fit only on block data (non-overlapping) | True |
| `kfold_max_features` | Max features after LassoCV selection | 50 |

### Per-Window Pipeline

| Parameter | Description | Default |
|-----------|-------------|---------|
| `skip_per_window_ic_ranking` | Skip IC re-ranking in training windows | True |

---

## 4. Time-Decay Weighting Configuration

**Time-decay weighting is OFF by default.** Our goal is to find features whose signal is **persistent and real** across time. Using **uniform weights** (all time points weighted equally) forces the pipeline to demonstrate that features survive across multiple time periods with consistent direction and magnitude.

### Configuration

In `config.py`, `FeaturesConfig`:

```python
use_time_decay_weights: bool = False      # OFF by default
formation_halflife_days: int = 252        # Half-life (only used if enabled)
```

---

## 5. Performance & Parallelism Guidelines

### Critical Rules

1. **Vectorize everything** - Use numpy/pandas vectorized operations instead of Python loops.
2. **Parallel inner loops, sequential outer walk**:
   - **Outer loop (sequential)**: Monthly window shifts walk forward in time
   - **Inner loop (parallel)**: Feature computation, IC calculation, model fitting use `joblib.Parallel`
3. **Why sequential outer loop?** Each walk-forward step depends on the previous window's end date.

### Memory Efficiency

- Use `float32` for feature matrices
- Delete intermediate arrays with `del` + `gc.collect()`

---

## 6. Implementation Status (December 2025)

### ✅ Completed

| Feature | Location | Notes |
|---------|----------|-------|
| Direction consistency | `feature_selection.py:formation_fdr()` | Requires 60% same sign |
| Block-level persistence | `feature_selection.py:formation_fdr()` | Gate 4: m_base blocks with correct sign + magnitude |
| Parent pre-filtering | `feature_selection.py:formation_interaction_screening()` | Stage 0 drops if BOTH parents bad |
| Non-overlapping block LassoCV | `feature_selection.py:kfold_lasso_cv_selection()` | `use_block_only=True` fits on block data only |
| Skip per-window IC ranking | `feature_selection.py:per_window_pipeline_v3()` | `skip_per_window_ic_ranking=True` |
| Config parameters | `config.py:FeatureConfig` | All params with defaults |

### Pipeline Flow (V3)

```
Input Features
    │
    ├─► Base Features ─────────────────────────────────────────┐
    │                                                          │
    └─► Interactions ──► Stage 0: Parent Pre-Filter ──────────┴─► formation_fdr()
                         (drop if both parents bad)                  │
                                                                     │
                         Four-Gate Filter:                           │
                         1. FDR significant (p < 0.10)               │
                         2. |IC| ≥ 0.02                              │
                         3. Direction consistency ≥ 60%              │
                         4. Block persistence (≥ m_base blocks)      │
                                                                     │
                                                                     ▼
                         Interaction Screening ◄──────────── Approved Base Features
                         (stricter thresholds)                       │
                                                                     │
                                                                     ▼
                                                              Merged Pool
                                                                     │
                                                                     ▼
                         Per-Window Pipeline (per_window_pipeline_v3):
                         1. [SKIPPED] Soft IC Ranking
                         2. Redundancy Filter (correlation)
                         3. Standardization
                         4. K-Fold LassoCV (non-overlapping blocks)
                         5. Ridge Refit
                                                                     │
                                                                     ▼
                                                              Final Features
```

### Formation FDR Summary Output

```
[Formation FDR] SUMMARY:
  Features in:  309
  Gate 1 - FDR significant:     188
  Gate 2 - + IC floor (≥0.02): 188
  Gate 3 - + Direction (≥60%): 186
  Gate 4 - + Block persist (≥2): 178
  Rejected (all gates): 131
```

### K-Fold LassoCV Summary Output

```
[K-Fold LassoCV] Starting 5-fold time-series CV (block-only (non-overlapping))...
[K-Fold LassoCV] Block 1: 14 features selected (α=0.013511, n_train=3665)
[K-Fold LassoCV] Block 2: 1 features selected (α=1.087696, n_train=3665)
...
[K-Fold LassoCV] Features with π ≥ 60%: 1 (from 5 valid blocks)
[K-Fold LassoCV] Final: 12 features (bounds: [12, 50])
```
