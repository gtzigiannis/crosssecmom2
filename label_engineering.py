"""
Label Engineering Module for Cross-Sectional Momentum Strategy.

This module computes target labels for the cross-sectional momentum pipeline:
- y_raw_21d: Raw H-day forward return (preserved from existing logic)
- y_cs_21d: Cross-sectionally demeaned H-day forward return
- y_resid_21d: Risk-adjusted residual (optional regression on controls)
- y_resid_z_21d: Risk-adjusted residual z-score (preferred label)

The module is called once per data build, after feature engineering but before
any walk-forward splits. All downstream code uses config.target.target_column.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ResearchConfig, TargetConfig

logger = logging.getLogger(__name__)


def compute_targets(
    panel: pd.DataFrame, 
    config: ResearchConfig,
    raw_return_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute all target columns for the cross-sectional momentum pipeline.
    
    This is the main entry point for label engineering. It computes:
    1. y_raw_21d: Raw H-day forward return
    2. y_cs_21d: Cross-sectionally demeaned return
    3. y_resid_21d: Risk-adjusted residual (if controls specified)
    4. y_resid_z_21d: Winsorised and z-scored residual
    
    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with MultiIndex [date, asset] or columns ['date', 'asset'].
        Must contain the raw return column and any risk control columns.
    config : ResearchConfig
        Configuration object containing TargetConfig settings.
    raw_return_col : str, optional
        Name of existing raw return column. If None, uses FwdRet_{horizon}.
        
    Returns
    -------
    pd.DataFrame
        Panel with additional columns: y_raw_21d, y_cs_21d, y_resid_21d, y_resid_z_21d
    """
    tc = config.target
    
    # Make a copy to avoid modifying input
    panel = panel.copy()
    
    # Ensure we have date column accessible
    if isinstance(panel.index, pd.MultiIndex):
        panel = panel.reset_index()
    
    # Normalize column names: Date -> date, Ticker -> asset for internal processing
    col_renames = {}
    if 'Date' in panel.columns and 'date' not in panel.columns:
        col_renames['Date'] = 'date'
    if 'Ticker' in panel.columns and 'asset' not in panel.columns:
        col_renames['Ticker'] = 'asset'
    if col_renames:
        panel = panel.rename(columns=col_renames)
    
    # Step 1: Compute/alias raw forward return
    panel = _compute_raw_target(panel, tc, raw_return_col)
    
    # Step 2: Cross-sectional demeaning
    panel = _compute_cs_demeaned(panel, tc)
    
    # Step 3: Risk adjustment (regression residuals)
    panel = _compute_risk_adjusted(panel, tc)
    
    # Step 4: Winsorisation and z-scoring
    panel = _compute_zscore(panel, tc)
    
    # Restore original column names (date -> Date, asset -> Ticker)
    reverse_renames = {}
    if 'date' in panel.columns and 'Date' not in panel.columns:
        reverse_renames['date'] = 'Date'
    if 'asset' in panel.columns and 'Ticker' not in panel.columns:
        reverse_renames['asset'] = 'Ticker'
    if reverse_renames:
        panel = panel.rename(columns=reverse_renames)
    
    # Restore MultiIndex if Date and Ticker columns exist
    if 'Date' in panel.columns and 'Ticker' in panel.columns:
        panel = panel.set_index(['Date', 'Ticker'])
    
    logger.info(
        f"Target engineering complete. Active target: {tc.target_column}. "
        f"Non-null targets: {panel[tc.target_column].notna().sum():,}"
    )
    
    return panel


def _compute_raw_target(
    panel: pd.DataFrame, 
    tc: TargetConfig,
    raw_return_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Step 1: Compute or alias raw H-day forward return.
    
    If raw_return_col is provided, uses that column.
    Otherwise, looks for FwdRet_{horizon} column.
    Creates y_raw_21d as the canonical raw target column.
    """
    horizon = tc.target_horizon_days
    
    # Determine source column
    if raw_return_col is not None:
        src_col = raw_return_col
    else:
        src_col = f"FwdRet_{horizon}"
    
    if src_col not in panel.columns:
        raise ValueError(
            f"Raw return column '{src_col}' not found in panel. "
            f"Available columns: {list(panel.columns)[:20]}..."
        )
    
    # Create y_raw_{horizon}d column
    panel[tc.target_column_raw] = panel[src_col].copy()
    
    n_valid = panel[tc.target_column_raw].notna().sum()
    logger.info(f"Raw target '{tc.target_column_raw}' computed. Valid values: {n_valid:,}")
    
    return panel


def _compute_cs_demeaned(
    panel: pd.DataFrame, 
    tc: TargetConfig,
) -> pd.DataFrame:
    """
    Step 2: Cross-sectional demeaning.
    
    For each date (and optionally group), subtract the cross-sectional mean.
    This removes common market drift and isolates relative performance.
    
    Demean modes:
    - 'global': Demean across all assets per date
    - 'by_asset_class': Demean within asset class groups per date
    - 'by_sector': Demean within sector groups per date
    - 'none': Skip demeaning (y_cs = y_raw)
    """
    raw_col = tc.target_column_raw
    cs_col = tc.target_column_cs
    
    if tc.target_demean_mode == "none":
        panel[cs_col] = panel[raw_col].copy()
        logger.info(f"Demeaning skipped (mode='none'). {cs_col} = {raw_col}")
        return panel
    
    # Determine grouping columns
    if tc.target_demean_mode == "global":
        group_cols = ["date"]
    elif tc.target_demean_mode == "by_asset_class":
        if "asset_class" not in panel.columns:
            logger.warning(
                "asset_class column not found. Falling back to global demeaning."
            )
            group_cols = ["date"]
        else:
            group_cols = ["date", "asset_class"]
    elif tc.target_demean_mode == "by_sector":
        if "sector" not in panel.columns:
            logger.warning(
                "sector column not found. Falling back to global demeaning."
            )
            group_cols = ["date"]
        else:
            group_cols = ["date", "sector"]
    else:
        raise ValueError(f"Unknown demean_mode: {tc.target_demean_mode}")
    
    # Compute cross-sectional mean per group
    group_means = panel.groupby(group_cols)[raw_col].transform("mean")
    
    # Demean
    panel[cs_col] = panel[raw_col] - group_means
    
    # Verify demeaning worked (mean should be ~0 per group)
    mean_abs_bias = panel.groupby("date")[cs_col].mean().abs().mean()
    logger.info(
        f"Cross-sectional demeaning complete (mode='{tc.target_demean_mode}'). "
        f"Mean absolute bias: {mean_abs_bias:.2e}"
    )
    
    return panel


def _compute_risk_adjusted(
    panel: pd.DataFrame, 
    tc: TargetConfig,
) -> pd.DataFrame:
    """
    Step 3: Risk adjustment via regression residuals.
    
    For each date (and group), regress y_cs on risk control columns and use residuals.
    This removes exposure to known risk factors (beta, volatility, etc.).
    
    If no controls specified or risk adjustment disabled, y_resid = y_cs.
    """
    cs_col = tc.target_column_cs
    resid_col = tc.target_column_resid
    
    # Check if risk adjustment is enabled and controls are specified
    controls = tc.target_risk_control_columns or []
    
    if not tc.target_use_risk_adjustment or len(controls) == 0:
        panel[resid_col] = panel[cs_col].copy()
        logger.info(
            f"Risk adjustment skipped (enabled={tc.target_use_risk_adjustment}, "
            f"n_controls={len(controls)}). {resid_col} = {cs_col}"
        )
        return panel
    
    # Verify control columns exist
    missing_controls = [c for c in controls if c not in panel.columns]
    if missing_controls:
        logger.warning(
            f"Missing risk control columns: {missing_controls}. "
            f"Falling back to {resid_col} = {cs_col}"
        )
        panel[resid_col] = panel[cs_col].copy()
        return panel
    
    # Perform risk adjustment per date
    def regress_residuals(group: pd.DataFrame) -> pd.Series:
        """Compute regression residuals for a single date group."""
        y = group[cs_col].values
        X = group[controls].values
        
        # Filter valid observations
        valid_mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        n_valid = valid_mask.sum()
        
        # Need more observations than regressors
        min_obs = tc.target_min_regression_samples
        if n_valid < max(min_obs, len(controls) + 1):
            # Not enough data for regression, return original
            return pd.Series(group[cs_col].values, index=group.index)
        
        y_valid = y[valid_mask]
        X_valid = X[valid_mask]
        
        # Add intercept
        X_valid = np.column_stack([np.ones(n_valid), X_valid])
        
        try:
            # Use least squares with regularization for stability
            # Ridge-like: (X'X + lambda*I)^-1 X'y
            lambda_reg = 1e-6
            XtX = X_valid.T @ X_valid
            XtX += lambda_reg * np.eye(XtX.shape[0])
            Xty = X_valid.T @ y_valid
            beta = np.linalg.solve(XtX, Xty)
            
            # Compute fitted values and residuals
            fitted = X_valid @ beta
            residuals = np.full(len(y), np.nan)
            residuals[valid_mask] = y_valid - fitted
            
            return pd.Series(residuals, index=group.index)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Regression failed for date group: {e}")
            return pd.Series(group[cs_col].values, index=group.index)
    
    # Apply regression per date
    logger.info(f"Computing risk-adjusted residuals using controls: {controls}")
    residuals = panel.groupby("date", group_keys=False).apply(
        regress_residuals, include_groups=False
    )
    panel[resid_col] = residuals.values
    
    # Check correlation between residuals and controls
    for ctrl in controls:
        corr = panel[[resid_col, ctrl]].dropna().corr().iloc[0, 1]
        logger.info(f"Residual-control correlation ({ctrl}): {corr:.4f}")
    
    return panel


def _compute_zscore(
    panel: pd.DataFrame, 
    tc: TargetConfig,
) -> pd.DataFrame:
    """
    Step 4: Optional winsorisation and per-date z-scoring.
    
    For each date:
    1. If target_enable_winsorization=True: Winsorise residuals at configured quantiles
    2. Compute z-score: (x - mean) / std
    
    This produces a standardised target with ~N(0,1) distribution per date.
    """
    resid_col = tc.target_column_resid
    zscore_col = tc.target_column_resid_z
    
    enable_winsorization = getattr(tc, 'target_enable_winsorization', False)
    lower_q, upper_q = tc.target_winsorization_limits
    
    def zscore_with_optional_winsorization(group: pd.Series) -> pd.Series:
        """Optionally winsorise and z-score a single date group."""
        values = group.values.copy()
        valid_mask = ~np.isnan(values)
        
        if valid_mask.sum() < tc.target_min_regression_samples:
            # Not enough observations for meaningful z-score
            return pd.Series(np.full(len(values), np.nan), index=group.index)
        
        valid_values = values[valid_mask]
        
        # Optional winsorisation
        if enable_winsorization:
            lower_bound = np.nanquantile(valid_values, lower_q)
            upper_bound = np.nanquantile(valid_values, upper_q)
            valid_values = np.clip(valid_values, lower_bound, upper_bound)
        
        # Z-score
        mu = np.nanmean(valid_values)
        sigma = np.nanstd(valid_values, ddof=1)
        
        if sigma < tc.target_min_std_for_zscore:
            # Near-zero variance, can't compute meaningful z-score
            return pd.Series(np.full(len(values), np.nan), index=group.index)
        
        result = np.full(len(values), np.nan)
        result[valid_mask] = (valid_values - mu) / sigma
        
        return pd.Series(result, index=group.index)
    
    # Apply per date
    zscore_values = panel.groupby("date", group_keys=False)[resid_col].apply(
        zscore_with_optional_winsorization
    )
    panel[zscore_col] = zscore_values.values
    
    # Summary statistics
    mean_zscore = panel[zscore_col].mean()
    std_zscore = panel[zscore_col].std()
    winsor_status = "enabled" if enable_winsorization else "disabled"
    logger.info(
        f"Z-score computation complete (winsorization {winsor_status}). "
        f"Global mean: {mean_zscore:.4f}, std: {std_zscore:.4f}"
    )
    
    return panel


def get_target_columns(tc: TargetConfig) -> List[str]:
    """Return list of all target columns that will be created."""
    return [
        tc.target_column_raw,
        tc.target_column_cs,
        tc.target_column_resid,
        tc.target_column_resid_z,
    ]


def validate_targets(panel: pd.DataFrame, tc: TargetConfig) -> dict:
    """
    Validate computed targets and return diagnostics.
    
    Returns
    -------
    dict
        Dictionary with validation results and summary statistics.
    """
    results = {}
    
    for col in get_target_columns(tc):
        if col not in panel.columns:
            results[col] = {"status": "missing"}
            continue
        
        series = panel[col]
        n_total = len(series)
        n_valid = series.notna().sum()
        
        results[col] = {
            "status": "ok",
            "n_total": n_total,
            "n_valid": n_valid,
            "pct_valid": n_valid / n_total * 100,
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "q01": series.quantile(0.01),
            "q99": series.quantile(0.99),
        }
    
    # Check cross-sectional properties
    cs_col = tc.target_column_cs
    if cs_col in panel.columns and "date" in panel.columns:
        cs_means = panel.groupby("date")[cs_col].mean()
        results["cs_demeaning_check"] = {
            "mean_of_daily_means": cs_means.mean(),
            "max_abs_daily_mean": cs_means.abs().max(),
        }
    
    # Check z-score properties
    z_col = tc.target_column_resid_z
    if z_col in panel.columns and "date" in panel.columns:
        z_means = panel.groupby("date")[z_col].mean()
        z_stds = panel.groupby("date")[z_col].std()
        results["zscore_check"] = {
            "mean_of_daily_means": z_means.mean(),
            "mean_of_daily_stds": z_stds.mean(),
        }
    
    return results


if __name__ == "__main__":
    # Simple test with synthetic data
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic panel
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    assets = [f"ETF_{i}" for i in range(20)]
    
    rows = []
    for date in dates:
        for asset in assets:
            rows.append({
                "date": date,
                "asset": asset,
                "FwdRet_21": np.random.randn() * 0.02 + 0.001,  # ~2% vol, slight drift
                "beta": np.random.uniform(0.5, 1.5),
                "vol_21d": np.random.uniform(0.1, 0.3),
            })
    
    panel = pd.DataFrame(rows)
    
    # Create config
    from config import get_default_config
    config = get_default_config()
    
    # Override target settings
    config.target.target_horizon_days = 21
    config.target.target_risk_control_columns = ["beta", "vol_21d"]
    config.target.target_use_risk_adjustment = True
    
    # Compute targets
    panel = compute_targets(panel, config, raw_return_col="FwdRet_21")
    
    # Validate
    validation = validate_targets(panel, config.target)
    
    print("\n" + "="*60)
    print("TARGET VALIDATION RESULTS")
    print("="*60)
    for col, stats in validation.items():
        print(f"\n{col}:")
        if isinstance(stats, dict):
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("Sample of computed targets:")
    print("="*60)
    print(panel[["date", "asset"] + get_target_columns(config.target)].head(10))
