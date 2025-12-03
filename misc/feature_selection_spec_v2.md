# crosssecmom2 – Feature Selection & Meta-Periods Specification (for Copilot)

You are working in my `crosssecmom2` repository. Implement the complete feature-selection pipeline described below. Do not assume any prior design. Treat this document as the single source of truth.

Goals:
- Use a principled but computationally realistic feature-selection process.
- Keep feature engineering strictly X-only; all label-dependent logic lives in feature selection.
- Use a filter pipeline: **global FDR → per-window IC → stability → correlation redundancy → ElasticNetCV**, with **optional XGBoost**.
- Use three **meta-periods** (Formation / Validation / Test) correctly in a walk-forward process.
- Never use the **Test** period in any label-dependent design or tuning.
- Log diagnostics and timings to understand aggressiveness, stability, and bottlenecks.

Save this file as `feature_selection_spec.md` and follow it when refactoring or adding code.

---

## 1. Meta-periods and walk-forward usage

Configure three meta-periods in `config.py` by date ranges:

1. **Formation period** (earliest history)  
   Purpose:
   - Define the initial raw feature library (from feature engineering).
   - Perform global univariate IC + FDR on raw features to build an **Approved Raw Feature Universe** `F_approved_raw`.
   - Fix “structural” hyperparameters that do not require Validation search (FDR level, stability folds, correlation threshold, ElasticNetCV candidate grid, XGBoost base settings).

2. **Validation period** (middle history)  
   Purpose:
   - Run an **outer walk-forward** (WF) over Validation rebalance dates.
   - Evaluate a **small grid** of pipeline configurations (time-decay, IC thresholds, feature budget, XGBoost on/off).
   - Choose the configuration that delivers the best and most stable out-of-sample IC (and optionally returns) on Validation.
   - Optionally run a full Validation backtest with the frozen pipeline for QA; do not treat it as final OOS.

3. **Test period** (most recent history)  
   Purpose:
   - Run a final, untouched **outer walk-forward backtest** using the frozen pipeline.
   - Use Test as the only source of true OOS performance metrics (Sharpe, drawdown, turnover, etc.).
   - Test data must not be used in any label-based transformation, FDR, binning, or hyperparameter tuning.

Outer walk-forward (Validation and Test):

- For each rebalance date `t0`:
  - Define an outer training window `[train_start, train_end]` according to config.
  - Impose a 21-day **embargo**:
    - training data must satisfy `t + 21 < t0` (no overlap between label horizon and future formation of `t0`).
  - Train feature-selection and models on this window.
  - Score at `t0`.
  - Compute P&L from `t0` to `t0 + 21` using existing portfolio-construction logic.

---

## 2. Time-decay weights

For each outer training window `[train_start, train_end]`:

- For each date `t` in the window:
  - `age_t = (train_end - t)` in calendar days.
  - `lambda = ln(2) / half_life`, where `half_life` is a configurable scalar (e.g. 42–84 days).
  - `w_t = exp(-lambda * age_t)`.
- Each sample `(i,t)` gets `sample_weight = w_t`.

Use `{w_t}` for:

- Weighted mean IC aggregation over time.
- Newey–West t-stats on IC time series.
- `sample_weight` in ElasticNetCV and XGBoost.

`half_life` is a tunable hyperparameter in the Validation grid.

---

## 3. Feature engineering (X-only, raw features)

Feature engineering is X-only and can see the full history (Formation + Validation + Test) because it does not use labels.

1. **Raw continuous features**  
   - Starting from OHLCV, compute all rolling/statistical/technical transforms (RSI, EMAs, volatility, skew, kurtosis, etc.).
   - All transforms must be **left-closed**:
     - The feature at date `t` uses only information up to and including `t`.

2. **Global variance threshold (X-only)**  
   - Before any label-based logic, run a variance-threshold filter on all raw features across the full history:
     - compute variance of each raw feature over all dates and ETFs,
     - drop any feature with variance below a small configurable threshold.
   - The survivors form `features_raw_var_filtered`.

From this point on, only `features_raw_var_filtered` are candidates for label-based selection.

---

## 4. Formation: global IC + FDR on raw features

Using only Formation data and `features_raw_var_filtered`, build a global **Approved Raw Feature Universe** `F_approved_raw`.

Inputs:

- `features_raw_var_filtered(i,t)` restricted to Formation dates.
- `y(i,t)` restricted to Formation dates.
- Weights `{w_t}` computed over Formation.

For each raw feature `f`:

1. Compute daily Spearman IC over Formation:
   - For each date `t` in Formation, compute `rho_t^f = SpearmanIC( f(i,t), y(i,t) )` across ETFs.
2. Compute:
   - Weighted mean IC `rho_w_f` using `{w_t}`.
   - Newey–West t-stat `t_NW_f` on `{rho_t^f}` with `{w_t}`.
3. Convert `t_NW_f` to a p-value for `H0: IC_f = 0`.

Across features:

4. Apply FDR (e.g. Benjamini–Hochberg) to p-values.
5. Choose FDR level from config (e.g. 5–10%).
6. Define:
   - `F_approved_raw` = set of raw features whose p-values pass the FDR threshold.

Persist:

- `F_approved_raw` (list of feature names).
- For each raw feature: `rho_w_f`, `t_NW_f`, p-value, FDR decision.

All subsequent per-window selection starts from `F_approved_raw`. There is no binning at this stage.

---

## 5. Per-window feature-selection pipeline (Validation and Test)

For each outer training window `[train_start, train_end]` in Validation and Test:

Input:

- `y(i,t)` in `[train_start, train_end]` satisfying `t + 21 < t0`.
- Raw features `f(i,t)` for all `f ∈ F_approved_raw` in that window.
- Weights `{w_t}` computed for this window.

The per-window pipeline is:

1. Per-window univariate IC filter (Section 5.1).
2. Per-window stability across K folds (Section 5.2).
3. Supervised binning + representation choice for a small top subset (Section 5.3).
4. Correlation-based redundancy (Section 5.4).
5. Robust standardization with median/MAD (Section 5.5).
6. ElasticNetCV (core multivariate selector) (Section 5.6).
7. Optional XGBoost with early stopping (Section 5.7).
8. Final feature set for this window (Section 5.8).

### 5.1 Per-window univariate IC filter

For each `f ∈ F_approved_raw`:

1. For each date `t` in the outer training window:
   - compute `rho_t^f = SpearmanIC( f(i,t), y(i,t) )`.
2. Compute:
   - weighted mean IC `rho_w_f` using `{w_t}`,
   - Newey–West t-stat `t_NW_f` on `{rho_t^f}`.

Apply per-window IC thresholds (tunable via Validation grid):

- Keep `f` only if:
  - `abs(rho_w_f) > theta_IC_window`,
  - `abs(t_NW_f) > T_min_window`.

The survivors form `F_IC_window`.

Diagnostics per window:

- `n_start = |F_approved_raw|`.
- `n_after_ic = |F_IC_window|`.
- Summary stats of `rho_w_f` for survivors (mean, median, quantiles).

### 5.2 Per-window stability across K folds

To avoid one-regime features, enforce temporal stability inside the training window.

1. Split the outer training window into `K` contiguous folds by time (e.g. `K = 3` or `4`), fixed in Formation.
2. For each `f ∈ F_IC_window`:
   - For each fold `k`:
     - compute fold-level weighted mean IC `rho_w_f_k` using `{rho_t^f}` and `{w_t}` restricted to dates in fold `k`.
   - Compute:
     - sign-consistency = fraction of folds where `sign(rho_w_f_k) == sign(rho_w_f)`,
     - median magnitude = median of `abs(rho_w_f_k)` across folds.
3. Apply stability thresholds (fixed in Formation):
   - sign-consistency ≥ `p_min` (e.g. 0.8),
   - median `abs(rho_w_f_k)` ≥ `theta_IC_fold` (e.g. ~0.015–0.02).

Survivors form `F_stable_window`.

Diagnostics per window:

- `n_after_stability = |F_stable_window|`.
- Distribution of median `abs(rho_w_f_k)` among survivors.

### 5.3 Supervised binning + representation choice (top subset only)

**Important**: Supervised binning is **NEVER** done in Formation. It happens per walk-forward window using only that window's training data. This ensures no future information leaks into the binning process.

Supervised binning is applied only to a **small top subset** of features per window, and fitted **once per outer window**, not per inner CV fold.

Steps:

1. From `F_stable_window`, rank features by `abs(rho_w_f)` (from 6.1).
2. Take top `K_bin` features (configurable, e.g. 20–50). Call this subset `F_bin_candidates_window`.
3. For each `f ∈ F_bin_candidates_window`:
   - Fit a shallow `DecisionTreeRegressor` (or similar) on this outer training window:
     - X = `f(i,t)`; y = `y(i,t)`; sample weights `{w_t}`.
     - Keep depth and min_samples_leaf small (config).
   - Use this tree to define **supervised bins** (leaves).
   - Construct a binned feature `f_bin_sup(i,t)` for all `(i,t)` in the window.
   - Compute daily IC for `f_bin_sup`:
     - `rho_t_bin_sup = SpearmanIC( f_bin_sup(i,t), y(i,t) )`.
   - Compute weighted mean IC `rho_w_bin_sup` using `{w_t}`.

4. Representation choice per feature family `(f_raw, f_bin_sup)` in this window:
   - Keep whichever representation has higher `abs(rho_w)` in this window.
   - Drop the other representation.
   - **Simplified logic**: Always choose the representation with stronger IC; no epsilon threshold.

5. For features in `F_stable_window` but not in `F_bin_candidates_window`:
   - keep them as raw only (no supervised binning).

Let the resulting set of features (raw and supervised-binned) be `F_repr_window`.

Implementation notes:

- Supervised binning is **per outer window** and **after FDR + IC + stability**, so the number of features is small.
- Do not refit bins per inner fold in Validation; bins are fixed per outer window.

Diagnostics per window:

- `n_bin_candidates = |F_bin_candidates_window|`.
- For each candidate: `rho_w_f` vs `rho_w_bin_sup`, chosen representation.
- `n_after_repr = |F_repr_window|`.

### 5.4 Per-window Multicollinearity Filter

Use our existing Multicollinearity Filter to remove redundant features via correlation clustering in the outer training window.

1. Build the matrix of surviving features `(i,t)` and compute the feature–feature correlation matrix:
   - use the existing redundancy code’s correlation measure  
2. use MI as the selection criterion

Diagnostics per window:
- `n_after_corr`,
- cluster size distribution,
- which features were dropped as redundant vs which were kept.


### 5.5 Robust standardization via median/MAD

Before ElasticNetCV and correlation modeling, standardize features in a robust, leakage-safe way.

For each outer training window:

1. Build `X_corr_window` as the matrix of features in `F_corr_window`.
2. For each feature column `j`:
   - compute median: `med_j` over all rows `(i,t)` in the training window,
   - compute MAD: `mad_j = median(|x_ij - med_j|)`; add a small epsilon to avoid division by zero.
3. Define standardized matrix `Z_window`:
   - `z_ij = (x_ij - med_j) / (mad_j + eps)`.

Use `Z_window` for:

- ElasticNetCV (Section 5.6).
- XGBoost if desired (Section 5.7), or you may use non-standardized features; for simplicity, reuse `Z_window`.

For scoring at `t0`:

- Extract features at `t0` for the selected features,
- Standardize them using `med_j`, `mad_j` learned from the training window.

No standardization is done using Validation or Test data outside the training window.

### 5.6 ElasticNetCV (core multivariate selector)

ElasticNet is the core multivariate selector. Use `ElasticNetCV` with time-aware splits.

1. Input:
   - features: `Z_window` (rows `(i,t)` in the training window, columns = `F_corr_window`),
   - label: `y(i,t)` in the window,
   - `sample_weight = w_t` per row.
2. Use `ElasticNetCV` with:
   - a small, fixed list of candidate `(alpha, l1_ratio)` pairs (defined in config, decided in Formation),
   - **TimeSeriesSplit Configuration**:
     - `n_splits=3`: Creates 3 train/test folds with expanding training windows
     - `test_size=63`: Each test fold is 63 days (~3 months, matching rebalancing period)
     - `gap=21`: 21-day embargo between train and test (respects standard embargo)
     - This ensures proper time structure and no data leakage
3. Let ElasticNetCV choose the best `(alpha, l1_ratio)` based on validation MSE (or similar).
4. Refit a final ElasticNet model on the full training window with that pair.
5. Define ElasticNet-selected features:
   - `ENet_set` = features with non-zero coefficients (or `|coef|` above a tiny threshold),
   - you may optionally cap `ENet_set` by `K1` highest-|coef| features if needed to keep things small.

Diagnostics per window:

- `n_elasticnet_nonzero = |ENet_set|`.
- Distribution of coefficients.
- Time taken by ElasticNetCV (recorded for profiling).

### 5.7 Optional XGBoost with early stopping

XGBoost is an optional second selector, controlled by:

- `use_xgboost_for_selection: bool` in config.

If `use_xgboost_for_selection` is `False`:
- Skip XGBoost entirely for this window.

If `use_xgboost_for_selection` is `True`:

1. Use `Z_window` (or `X_corr_window`) as predictors; `y(i,t)` as label; `sample_weight = w_t`.
2. **Train/Val Split** (aligned with TimeSeriesSplit):
   - Training: days [0, 252] (expanding window, ~80% of typical ~315-day window)
   - Validation: days [273, 336] (21-day gap, 63-day validation)
   - This matches the last fold of our 3-fold TimeSeriesSplit in Section 5.6
3. Set conservative XGBoost defaults (fixed in Formation):
   - `max_depth`, `learning_rate`, regularization terms.
4. Train with:
   - `n_estimators` set to a relatively large value (e.g. 1000),
   - `early_stopping_rounds` set (e.g. 50),
   - using the validation split to trigger early stopping.
5. After training, extract feature importances (e.g. gain-based).
6. Define:
   - `XGB_set` = top `K2` features by importance (configurable; small, e.g. 10–20).

Diagnostics per window:

- `n_xgb_used = |XGB_set|`.
- Importance distribution.
- Time spent in XGBoost.

---

### 5.8 Scoring at t0 (OOS Prediction)

**Goal**: Generate predictions at t0 (rebalance date) for the forward period [t0, t0+21].

**Process**:

1. **Final Model Training**:
   - Train final ElasticNet (and optionally XGBoost) on full training window [train_start, train_end]
   - Use selected features from Section 5.6/5.7
   - Apply same time-decay weights used during cross-validation

2. **Feature Preparation at t0**:
   - Extract feature values at t0 for all assets in universe
   - Apply same transformations used during training:
     - Use stored binning boundaries (for binned features)
     - Apply robust standardization using training median/MAD
   - **Critical**: All transformations use parameters fitted on training data only

3. **Prediction**:
   - If using ElasticNet only: `scores = model.predict(X_t0)`
   - If using XGBoost: `scores = model.predict(X_t0)`
   - If using ensemble: `scores = w_enet * enet_pred + w_xgb * xgb_pred`
     - Default weights: w_enet=0.6, w_xgb=0.4 (can tune in Validation)

4. **Post-Processing**:
   - Rank-transform scores to percentiles (optional, for interpretability)
   - Apply any portfolio construction constraints (sector neutral, etc.)

5. **Output**:
   - Asset scores at t0
   - Store for P&L evaluation over [t0, t0+21]

**Timing**: This entire process should take <1 second per window (feature extraction is fast, prediction is instantaneous).

---

### 5.9 Final selected features per window

Combine ElasticNet and XGBoost selections into a final set for this window.

Hyperparameter:

- `F_final_window` = maximum number of final features (tuned in Validation).

Case 1: `use_xgboost_for_selection = False`

- Let `union = ENet_set`.
- If `|union| ≤ F_final_window`, keep all features in `union`.
- If `|union| > F_final_window`, keep top `F_final_window` features by `|coef|`.

Case 2: `use_xgboost_for_selection = True`

- Let `union = ENet_set ∪ XGB_set`.
- If `|union| ≤ F_final_window`, keep all features in `union`.
- If `|union| > F_final_window`:
  - For each feature in `union`, define a combined score:
    - rank by |ElasticNet coefficient| (features not in `ENet_set` get worst rank),
    - rank by XGBoost importance (features not in `XGB_set` get worst rank),
    - combine normalized ranks (e.g. sum).
  - Sort by combined score and keep top `F_final_window`.

These final selected features define the scoring model for this window.
The final scores are then used by the existing portfolio-construction code.

Diagnostics per window:

- `n_final = |final_feature_set|`.
- Composition of final set (ENet-only, XGB-only, both).
- Time spent in each step (IC, stability, binning, redundancy, scaling, ENetCV, XGB).

---

## 6. Validation-period tuning (Bayesian Optimization)

Validation is used to tune only a **small set of critical hyperparameters**.

Hyperparameters fixed in Formation (not tuned in Validation):

- FDR level in Formation.
- Stability config: `K`, `p_min`, `theta_IC_fold`.
- Correlation threshold `corr_thresh`.
- Candidate `(alpha, l1_ratio)` grid for ElasticNetCV.
- XGBoost structural settings (max_depth, base learning_rate, regularization).

Hyperparameters tuned in Validation (small grid):

- `half_life` (e.g. two values: {42, 63}).
- `theta_IC_window` (e.g. two values: {0.02, 0.03}).
- `M_window` (e.g. two values: {30, 50}).
- `F_final_window` (e.g. two values: {10, 15}).
- `use_xgboost_for_selection` (two values: {False, True}).

This yields up to 2×2×2×2×2 = 16 pipeline variants.

For each pipeline variant:

1. Run an outer walk-forward over Validation `t0` dates:
   - Optionally subsample `t0` (e.g. use every 2nd or 3rd rebalance date) to control compute.
2. For each `t0` and its training window:
   - Run the full per-window pipeline described in Section 5 with this variant’s hyperparameters.
   - Score Validation OOS (at `t0`).
3. On Validation OOS:
   - Compute daily cross-sectional IC between scores and `y(i,t0)`.
   - Optionally compute P&L metrics (Sharpe, etc.) over Validation.
4. For each variant:
   - Aggregate mean OOS IC over all Validation `t0` dates.
   - Examine stability of IC over time (variance, drawdowns).

Select the variant with the best and most stable Validation OOS IC (and acceptable P&L profile).

Freeze:

- `half_life`.
- `theta_IC_window`.
- `M_window`.
- `F_final_window`.
- `use_xgboost_for_selection`.

---

## 7. Final walk-forward backtests (Validation + Test)

Once hyperparameters are frozen:

1. Implement a function to run the **frozen pipeline** for an outer training window:
   - Input: training window dates, raw features (restricted to `F_approved_raw`), label `y(i,t)`.
   - Output: final feature set, ElasticNet model, optional XGBoost model, and scores at `t0`.

2. Run full WF backtests on:
   - Validation:
     - as a QA check with frozen pipeline,
     - do not retune based on this.
   - Test:
     - as the final OOS evaluation.

For each `t0` in Validation and Test:

- Define `[train_start, train_end]` and enforce embargo.
- Run the frozen pipeline on the training window.
- Score at `t0`.
- Pass scores into portfolio-construction.
- Compute P&L for the holding period and aggregate metrics.

Only Test-period results should be used as “final OOS”.

---

## 8. Diagnostics, stability, and representation monitoring

Implement logging to capture the behaviour and stability of the pipeline.

### 8.1 Per-window diagnostics

For each outer training window:

- Feature-count path:
  - `n_start`, `n_after_ic`, `n_after_stability`, `n_bin_candidates`, `n_after_repr`, `n_after_corr`, `n_elasticnet_nonzero`, `n_final`.
- IC stats:
  - mean/median/quantiles of `rho_w_f` at IC filter stage.
  - mean/median of `rho_w` for final selected features.
- Binning/repr:
  - for each supervised-binned feature: `rho_w_raw` vs `rho_w_bin_sup`, final representation.
- **Feature Importance Diagnostics**:
  - ElasticNet coefficients (magnitude and sign) for selected features
  - If using XGBoost: feature importance scores (gain-based)
  - Top 5 features by absolute coefficient/importance
  - Proportion of raw vs binned features in final selection
  - Stability of feature importances: correlation of coefficients across consecutive windows
- Model scores:
  - IC between final scores and `y(i,t)` on the training window (`t < t0`).
- Time spent in each step:
  - `time_ic`, `time_stability`, `time_binning_repr`, `time_corr`, `time_scaling`, `time_enet`, `time_xgb`.

### 8.2 Across-window stability

Track:

- **Feature turnover** between consecutive windows:
  - `turnover = 1 - |F_t ∩ F_{t+1}| / |F_t ∪ F_{t+1}|`.
- **ElasticNet coefficient stability**:
  - For features present in both windows, compute correlation between coefficient vectors.
- **Selection frequency**:
  - For each feature, how many windows it appears in at:
    - post-IC stage,
    - post-redundancy,
    - final selection.

### 8.3 Representation-choice monitoring (optional diagnostic)

Optionally, to assess robustness of supervised binning vs raw representation across meta-periods:

- Choose a subset of important feature families (e.g. those with consistently high IC).
- For each such family and for each meta-period (Formation, Validation):
  - Fit supervised bins per outer window.
  - Compute `rho_w_raw` and `rho_w_bin_sup`.
  - Record which representation is better (raw vs bin) in that window.
- Summarise:
  - proportion of windows where raw vs bin is preferred,
  - how often the preferred representation changes between Formation and Validation.

This monitoring is for analysis only and does **not** alter the pipeline’s representation decisions, which are made per window as described in Section 6.3.

### 8.4 Logging format

Store diagnostics in one or more Parquet/CSV files, e.g.:

- `feature_selection_windows.parquet`:
  - one row per `(period, t0)`, columns for all per-window metrics and timings.
- `feature_selection_features.parquet`:
  - one row per feature, columns for selection frequencies, average IC, etc.
- `representation_diagnostics.parquet` (optional):
  - per family, per period representation statistics.

---

## 9. Timing and bottleneck identification

Instrument each major step with simple timers:

- Global:
  - time for feature engineering,
  - time for Formation FDR.
- Per window:
  - IC filter,
  - stability,
  - supervised binning + representation,
  - correlation redundancy,
  - standardization (median/MAD),
  - ElasticNetCV,
  - XGBoost (if used).

Include timings in the logged diagnostics (Section 8) so bottlenecks can be identified and optimised.

---

## 9. Memory Management and Performance Optimization

### 9.1 Memory Efficiency

**Data Types**:
- Use `float32` instead of `float64` for all feature matrices (2x memory reduction)
- Store feature matrices as `np.ndarray` with dtype=np.float32
- Exception: Keep target variable as float64 for numerical stability in IC computation

**Intermediate Data Cleanup**:
- Clear large intermediate matrices after each window:
  - Raw feature matrix after binning/transformation
  - Correlation matrices after redundancy filter
  - CV fold data after ElasticNetCV
- Use explicit `del` and `gc.collect()` after each window completes
- Only persist:
  - Selected features list
  - Fitted model (ElasticNet/XGBoost)
  - Binning boundaries for selected features
  - Scaling parameters (median/MAD)

**Memory Monitoring**:
- Log peak memory usage per window (using `psutil`)
- Set memory alert threshold (e.g., 8GB per worker)
- If memory exceeds threshold, reduce K_bin or disable XGBoost for that window

### 9.2 Parallelization

**Safe Parallelization Opportunities**:
1. **IC Computation** (Section 5.1): Parallelize across features using `joblib`
   - Each feature's IC computed independently
   - ~4-8 workers typical

2. **Stability Folds** (Section 5.2): Already parallelized in `TimeSeriesSplit`
   - CV folds are independent
   - ElasticNetCV uses `n_jobs=-1`

3. **Supervised Binning** (Section 5.3): Parallelize across top K_bin features
   - Each feature's bins fitted independently
   - Use `joblib.Parallel` with `backend='loky'`

4. **Walk-Forward Windows**: Already parallelized at top level
   - Each rebalance window processed independently
   - Current implementation uses loky backend

**Not Parallelizable**:
- Sequential filters (IC → stability → redundancy) within a window
- Bayesian optimization (inherently sequential)

**Configuration**:
- Set `OMP_NUM_THREADS=1` to avoid nested parallelism
- Use `joblib.Parallel(n_jobs=4, backend='loky')` for feature-level parallelization
- Outer walk-forward parallelization: 4-8 windows concurrently (depending on RAM)

### 9.3 Expected Performance

**Per-Window Breakdown**:
- IC filter: ~2s (parallelized)
- Stability: ~1s (3-fold CV)
- Binning: ~0.5s (50 features, parallelized)
- Redundancy: ~0.3s (correlation matrix)
- ElasticNetCV: ~3s (3-fold TimeSeriesSplit)
- XGBoost (optional): ~5s (with early stopping)
- **Total**: ~10-15s per window

**Full Backtest** (25 windows):
- Serial: 25 × 12s = 300s (~5 minutes)
- Parallel (4 workers): ~75-90s (~1.5 minutes)

**Memory Footprint**:
- Per window: ~500MB-1GB (with float32)
- Parallel (4 workers): ~2-4GB total
- Safe for machines with 8GB+ RAM

---

## 9. Memory Management and Performance Optimization

### 9.1 Memory Efficiency

**Data Types**:
- Use `float32` instead of `float64` for all feature matrices (2x memory reduction)
- Store feature matrices as `np.ndarray` with dtype=np.float32
- Exception: Keep target variable as float64 for numerical stability in IC computation

**Intermediate Data Cleanup**:
- Clear large intermediate matrices after each window:
  - Raw feature matrix after binning/transformation
  - Correlation matrices after redundancy filter
  - CV fold data after ElasticNetCV
- Use explicit `del` and `gc.collect()` after each window completes
- Only persist:
  - Selected features list
  - Fitted model (ElasticNet/XGBoost)
  - Binning boundaries for selected features
  - Scaling parameters (median/MAD)

**Memory Monitoring**:
- Log peak memory usage per window (using `psutil`)
- Set memory alert threshold (e.g., 8GB per worker)
- If memory exceeds threshold, reduce K_bin or disable XGBoost for that window

### 9.2 Parallelization

**Safe Parallelization Opportunities**:
1. **IC Computation** (Section 5.1): Parallelize across features using `joblib`
   - Each feature's IC computed independently
   - ~4-8 workers typical

2. **Stability Folds** (Section 5.2): Already parallelized in `TimeSeriesSplit`
   - CV folds are independent
   - ElasticNetCV uses `n_jobs=-1`

3. **Supervised Binning** (Section 5.3): Parallelize across top K_bin features
   - Each feature's bins fitted independently
   - Use `joblib.Parallel` with `backend='loky'`

4. **Walk-Forward Windows**: Already parallelized at top level
   - Each rebalance window processed independently
   - Current implementation uses loky backend

**Not Parallelizable**:
- Sequential filters (IC → stability → redundancy) within a window
- Bayesian optimization (inherently sequential)

**Configuration**:
- Set `OMP_NUM_THREADS=1` to avoid nested parallelism
- Use `joblib.Parallel(n_jobs=4, backend='loky')` for feature-level parallelization
- Outer walk-forward parallelization: 4-8 windows concurrently (depending on RAM)

### 9.3 Expected Performance

**Per-Window Breakdown**:
- IC filter: ~2s (parallelized)
- Stability: ~1s (3-fold CV)
- Binning: ~0.5s (50 features, parallelized)
- Redundancy: ~0.3s (correlation matrix)
- ElasticNetCV: ~3s (3-fold TimeSeriesSplit)
- XGBoost (optional): ~5s (with early stopping)
- **Total**: ~10-15s per window

**Full Backtest** (25 windows):
- Serial: 25 × 12s = 300s (~5 minutes)
- Parallel (4 workers): ~75-90s (~1.5 minutes)

**Memory Footprint**:
- Per window: ~500MB-1GB (with float32)
- Parallel (4 workers): ~2-4GB total
- Safe for machines with 8GB+ RAM

---

## 10. Future Enhancements and Implementation Notes

**Future Enhancement - Grouped ElasticNet**: Consider using grouped lasso (e.g., `GroupLasso` from `celer` package) to handle feature families (e.g., all momentum features, all volatility features) as groups. This can improve interpretability and stability by encouraging selection/rejection of entire feature groups. **Status**: Deferred to future sprint; will implement after validating core pipeline performance.

---

### 10.1 Implementation notes

- If you can disentangle feature selection from alpha_models.py without breaking the code, implement in a dedicated module (e.g. `feature_selection.py`) with clear functions for:
  - Formation IC+FDR on raw features,
  - per-window IC and stability,
  - supervised binning and representation choice for top features,
  - correlation redundancy,
  - robust scaling (median/MAD),
  - ElasticNetCV and optional XGBoost,
  - Validation grid evaluation and Test backtesting,
  - diagnostics and timing.
- Ensure that:
  - No label-based operation uses Test data.
  - Embargo is respected for all training slices.
  - Supervised bins are fitted **once per outer window** and reused for that window’s evaluation; no re-fitting per inner fold.
  - The Validation grid remains small and runtime controlled.

implement this specification end-to-end, keeping modularity and performance in mind.
