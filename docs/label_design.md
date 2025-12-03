# Target Redesign Specification (Cross-Sectional, Risk-Adjusted Label)

The goal is to redesign the target variable so that the models learn **cross-sectional, risk-adjusted, regime-aware** momentum rather than raw 21-day drift.

Do not remove existing functionality. Preserve the current raw 21-day return target for comparison. Add new labels and wire them into the existing walk-forward and feature-selection pipeline behind configuration flags.

## I. High-Level Design

We want three related target variants for horizon `H = 21` days:

1. `y_raw_21d`: raw H-day forward return, existing logic preserved.
2. `y_cs_21d`: cross-sectionally demeaned H-day forward return.
3. `y_resid_z_21d`: risk-adjusted residual z-score (preferred label):
   - Cross-sectionally demeaned.
   - Optionally residualised against simple risk controls (beta, vol, liquidity, structure).
   - Winsorised and z-scored per date.

The existing pipeline (interaction screening, Formation FDR, redundancy, soft ranking, LassoCV, WF engine) must continue to work, but use `config.target_column` to decide which label to use.

## II. Config Changes

Edit `config.py` in the `crosssecmom2` module. Add a dedicated TargetConfig or equivalent section.

1. Add core parameters:
   - `target_horizon_days: int = 21`
   - `target_column_raw: str = "y_raw_21d"`
   - `target_column_cs: str = "y_cs_21d"`
   - `target_column_resid_z: str = "y_resid_z_21d"`
   - `target_column: str = "y_resid_z_21d"` (the column actually used by WF / models; allow user to switch between raw, cs, resid_z)

2. Cross-sectional demeaning / grouping:
   - `target_demean_mode: str = "global"` (options: `"global"`, `"by_asset_class"`, `"by_sector"`; default `"global"`)
   - If non-global modes are used later, they should group by an existing structural column (e.g. `asset_class`, `sector`).

3. Risk adjustment toggles:
   - `target_use_risk_adjustment: bool = True` (enable/disable regression residualisation)
   - `target_risk_control_columns: list[str] = []` (default empty = no risk regression)
   - `target_winsorization_limits: tuple[float,float] = (0.01, 0.99)` for per-date winsorisation of residuals before z-scoring.

4. Add any other helper parameters you need (e.g. minimum group size for regression), but keep defaults safe and backwards-compatible.

## III. New Module: Label Engineering

Create a new module `label_engineering.py` inside the crosssecmom2 package. This module is responsible for computing all target columns on the long panel that other modules use. 

### III.1. Module responsibilities

1. Compute raw forward returns for the given horizon (`y_raw_21d`).
2. Compute cross-sectional demeaned returns (`y_cs_21d`) as a function of `target_demean_mode`.
3. Regress `y_cs_21d` on risk controls and use residuals.
4. Winsorise and z-score residuals per date to produce `y_resid_z_21d`.
5. Return a DataFrame with all three target columns added to the original panel.

### III.2. Inputs and outputs

Assume there is a central “panel” DataFrame (MultiIndex [date, asset] or equivalent) that already contains:

- Price/return information needed to compute H-day forward returns.
- Already-computed features (beta, vol, liquidity, structure flags) that can be used as risk controls.

Define a main function:

- `def compute_targets(panel: pd.DataFrame, config: Config) -> pd.DataFrame:`
  - `panel` has at least: prices/returns needed for H-day forward return, plus any columns listed in `config.target_risk_control_columns`.
  - Returns a new DataFrame with additional columns: `y_raw_21d`, `y_cs_21d`, `y_resid_21d`, `y_resid_z_21d`.

Do not hard-code column names for risk controls. Use `config.target_risk_control_columns`; if it is empty, skip the regression step.

## IV. Label Computation Logic

Implement the following logic in `compute_targets` (or helper functions it calls).

### IV.1. Raw H-day forward return (`y_raw_21d`)

1. Identify the existing column and logic currently used to compute the 21-day forward return (e.g. `ret_21d`, `forward_return`, etc.) by searching the repo.
2. Refactor that logic into a reusable helper function if necessary.
3. Create a new column (or alias) called `y_raw_21d` that holds that H-day forward return:
   - Either reuse the existing column,
   - Or compute anew using the existing implementation.

Do not change the semantics of the existing raw target; just rename/expose it as `y_raw_21d`.

### IV.2. Cross-sectional demeaning (`y_cs_21d`)

For each date `t` (and universe subset depending on `target_demean_mode`):

1. Identify the group of assets to demean over:
   - If `target_demean_mode == "global"`: group = all assets active on date `t`.
   - If `"by_asset_class"`: group by an existing `asset_class` or similar field.
   - If `"by_sector"`: group by `sector` or equivalent (only if present; otherwise fallback to global).

2. Within each group `G` at date `t`:
   - Compute group mean of `y_raw_21d`: `mean_G_t`.
   - Define `y_cs_21d = y_raw_21d - mean_G_t`.

Implementation requirements:
- Use vectorised groupby operations; avoid Python loops over individual assets.
- Handle missing values gracefully; skip assets with NaN `y_raw_21d` in the group statistics but keep them in the panel.

### IV.3. Risk adjustment (residual label `y_resid_21d`)

This step must be driven by config and robust to `target_risk_control_columns` being empty.

1. If `config.target_use_risk_adjustment` is `False` or `target_risk_control_columns` is empty:
   - Set `y_resid_21d = y_cs_21d`.
   - Skip regression and proceed to winsorisation/z-scoring.

2. If risk adjustment is enabled and there are control columns:
   - For each date `t`, and each group (consistent with `target_demean_mode`):
     - Form a design matrix `X_t` with columns from `panel[target_risk_control_columns]` for assets in the group with non-null data.
     - The response is `y_cs_21d_t` (demeaned return) for those assets.
     - Fit a linear model per group per date:
       - Prefer a numerically robust solver (e.g. numpy least squares or ridge with small alpha).
       - Do not overcomplicate; the number of controls is small and the cross-section is large.

   - For each asset in the group, compute fitted value `X_t * beta_t` and residual:
     - `y_resid_21d = y_cs_21d - fitted`.

   - Store `y_resid_21d` in the panel.

Implementation requirements:
- Use groupby on date (and optionally asset_class/sector) and apply a function that:
  - Builds X and y from `y_cs_21d` and the configured control columns,
  - Fits coefficients,
  - Computes residuals;
  - Returns a Series of residuals aligned to the group index.
- If a group has too few assets or ill-conditioned X (e.g. fewer rows than columns), fall back to `y_resid_21d = y_cs_21d` for that group and log a warning.

### IV.4. Winsorisation and per-date z-scoring (`y_resid_z_21d`)

Using `y_resid_21d`:

1. For each date `t` (global grouping, not by asset_class/sector):
   - Compute lower and upper quantiles based on `target_winsorization_limits`, e.g. 1% and 99%.
   - Clip residuals at those quantiles: `y_resid_clipped`.

2. For each date `t`:
   - Compute mean and std of `y_resid_clipped` across the active universe:
     - `mu_t = mean_i(y_resid_clipped_{i,t})`
     - `sigma_t = std_i(y_resid_clipped_{i,t})` (use sample std, avoid zero division; if `sigma_t` is very small, set z-score to 0 for that date).

   - Define:
     - `y_resid_z_21d = (y_resid_clipped - mu_t) / sigma_t`

3. Store both `y_resid_21d` and `y_resid_z_21d` in the panel.

This label is what the model will use when `config.target_column = "y_resid_z_21d"`.

## V. Integration with Existing Pipeline

### V.1. Data flow and module wiring

Identify the main place where the panel DataFrame is constructed and where the current target is computed (likely in `data_manager.py` or `feature_engineering.py`).

1. After all basic features and the raw target are available, call `target_engineering.compute_targets` to add the new columns.
2. Ensure this is done once per data build, before any Formation/Training/Test splits and before interaction screening or feature selection.

Do not re-compute targets inside the walk-forward loop. Targets must be precomputed for the entire history.

### V.2. Make target usage config-driven

Search for all places where the target is referenced directly (examples: `ret_21d`, `forward_return`, `target`, etc.) in:

- `feature_selection.py`
- `alpha_models.py`
- `walk_forward_engine.py`
- Any other relevant module.

Refactor them so that:

- They use `config.target_column` as the column name for y.
- Where needed, they can still access `y_raw_21d` or `y_cs_21d` for diagnostics, but the main model fitting and IC computations use `config.target_column`.

Examples of changes (conceptual, do not insert literal code here):
- In Formation FDR IC: compute IC between features and `panel[config.target_column]`.
- In per-window stability: use `config.target_column`.
- In LARS/Ridge fitting: use `config.target_column` as y.
- In walk-forward engine: pass the target column name through so all modules use the same label.

Ensure that tests that assume the old target name are updated to use `config.target_column` or explicitly test both `y_raw_21d` and `y_resid_z_21d`.

## VI. Tests

Add a new test module `test_target_engineering.py` (or equivalent) under the crosssecmom2 tests.

### VI.1. Forward-return correctness and no leakage

On a small synthetic panel (few dates, few assets):

- Construct deterministic prices such that the 21-day forward return is known.
- Run `compute_targets`.
- Assert:
  - `y_raw_21d` matches the expected forward returns.
  - There is no leakage: `y_raw_21d` at date t only depends on prices up to t+H, not beyond.

### VI.2. Cross-sectional demeaning

On synthetic data with non-zero cross-sectional mean:

- After computing `y_cs_21d`, verify that for each date (and group if using by_asset_class/sector):
  - The cross-sectional mean of `y_cs_21d` is approximately 0 (within numerical tolerance).

### VI.3. Risk adjustment (if controls are provided)

Create a synthetic panel with a known linear relationship between a control column and `y_cs_21d`:

- Include one control column `X` with strong linear effect on `y_cs_21d`.
- Set `config.target_risk_control_columns = ["X"]` and `target_use_risk_adjustment = True`.
- After computing `y_resid_21d`, verify that:
  - The cross-sectional correlation between `y_resid_21d` and `X` is close to 0.

Also test the fallback path where `target_risk_control_columns = []` yields `y_resid_21d == y_cs_21d`.

### VI.4. Winsorisation and z-scoring

On synthetic residuals:

- Verify:
  - Extremes are clipped at the configured quantiles.
  - For each date, `y_resid_z_21d` has mean ~0 and std ~1 (within tolerance), except for edge cases where there are too few assets.

### VI.5. Integration sanity check

Add or extend an integration test (e.g. in `test_walk_forward_integration.py`) to:

- Build a minimal panel, compute targets, run a one-window WF using `config.target_column = "y_resid_z_21d"`.
- Assert:
  - The pipeline runs end-to-end without errors.
  - At least one rebalance date is processed and scores are produced.
  - No hard-coded references to the old target column remain.

## VII. Performance and robustness

- Ensure `compute_targets` is implemented in a vectorised manner where possible (groupby, apply), and avoid excessive Python loops.
- Be defensive about NaNs: if a label cannot be computed for a row (e.g. missing prices), leave it as NaN but do not crash.
- Log informative messages when falling back from risk-adjusted residuals to simpler `y_cs_21d` due to insufficient group size or ill-conditioned X.

## VIII. Summary

Implement the above as a self-contained “target engineering” layer that:

- Computes `y_raw_21d`, `y_cs_21d`, `y_resid_21d`, `y_resid_z_21d`.
- Makes the target used by the selection and model pipeline configurable via `config.target_column`.
- Leaves existing raw-target-based behaviour intact when `config.target_column` is set to the old name.
- Integrates cleanly with the existing feature-selection and walk-forward engine without changing their core logic.
