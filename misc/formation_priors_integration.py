"""
Integration of Economic Priors with Feature Selection Pipeline
==============================================================

This module shows how to integrate economic prior filtering into the
existing formation_fdr and per_window_pipeline.

The key insight: Apply economic priors BEFORE FDR control to:
1. Reduce multiple testing burden (fewer tests = less FDR penalty)
2. Ensure only economically justified features enter selection
3. Enforce sign consistency with financial theory
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from economic_priors import (
    EconomicPriorFilter, 
    apply_economic_prior_to_formation,
    get_prior_for_feature,
    ExpectedSign
)

logger = logging.getLogger(__name__)


def formation_fdr_with_priors(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
    half_life: int = 126,
    fdr_level: float = 0.10,
    n_jobs: int = 4,
    config = None,
    use_economic_priors: bool = True,
    require_sign_match: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Enhanced Formation FDR with economic prior filtering.
    
    New pipeline order:
    1. Compute daily IC for all features
    2. ** NEW: Apply economic prior filter **
    3. Apply FDR only to features that passed priors
    4. Apply short-lag protection if enabled
    
    This dramatically reduces the feature space before statistical testing,
    which:
    - Reduces multiple testing penalty
    - Ensures economically sensible features
    - Improves signal-to-noise ratio
    """
    from feature_selection import (
        compute_daily_ic_series,
        compute_time_decay_weights,
        _compute_newey_west_vectorized,
        protect_short_lag_features,
        log_memory_usage,
    )
    from statsmodels.stats.multitest import multipletests
    from scipy import stats
    import time
    import gc
    
    logger.info("=" * 80)
    logger.info("Starting Formation FDR with Economic Priors")
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    log_memory_usage("Formation start")
    
    start_time = time.time()
    stage_times = {}
    
    # =========================================================================
    # STEP 1: NaN handling
    # =========================================================================
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Dropping {len(nan_cols)} columns with NaN")
        X = X.drop(columns=nan_cols)
        
    X = X.astype(np.float32)
    y = y.astype(np.float64)
    
    # =========================================================================
    # STEP 2: Compute daily IC for ALL features
    # =========================================================================
    stage_start = time.time()
    print(f"[Formation FDR] Computing daily IC series for {X.shape[1]} features...")
    ic_daily = compute_daily_ic_series(X, y, dates)
    stage_times['daily_ic'] = time.time() - stage_start
    
    # Compute weighted mean IC
    unique_dates = ic_daily.index
    train_end = unique_dates.max()
    weights = compute_time_decay_weights(unique_dates, train_end, half_life)
    ic_values = ic_daily.values
    ic_weighted = np.average(ic_values, axis=0, weights=weights)
    
    # Build feature → IC mapping
    feature_ic_dict = dict(zip(X.columns, ic_weighted))
    
    # =========================================================================
    # STEP 3: ** NEW ** Apply Economic Prior Filter
    # =========================================================================
    prior_approved_features = list(X.columns)  # Default: all features
    prior_stats = None
    
    if use_economic_priors:
        stage_start = time.time()
        print(f"\n[Economic Priors] Applying economic prior filter...")
        
        prior_filter = EconomicPriorFilter(
            require_sign_match=require_sign_match,
            allow_unprioried_features=False,
            min_prior_confidence=0.3,
        )
        
        passed_ics = prior_filter.filter_features(feature_ic_dict, verbose=True)
        prior_approved_features = list(passed_ics.keys())
        prior_stats = prior_filter.stats.copy()
        
        stage_times['economic_priors'] = time.time() - stage_start
        
        print(f"[Economic Priors] {len(prior_approved_features)}/{X.shape[1]} features passed")
        
        if len(prior_approved_features) == 0:
            raise ValueError("No features passed economic prior filter!")
    
    # =========================================================================
    # STEP 4: Run FDR only on prior-approved features
    # =========================================================================
    # Filter to only prior-approved features
    X_filtered = X[prior_approved_features]
    ic_daily_filtered = ic_daily[prior_approved_features]
    
    stage_start = time.time()
    print(f"\n[Formation FDR] Computing Newey-West t-stats for {len(prior_approved_features)} features...")
    
    # Recompute weighted IC for filtered features
    ic_values_filtered = ic_daily_filtered.values
    ic_weighted_filtered = np.average(ic_values_filtered, axis=0, weights=weights)
    
    t_nw_values, n_dates_per_feature = _compute_newey_west_vectorized(
        ic_daily_filtered, weights, max_lags=5
    )
    
    # Convert to p-values
    p_values = np.where(
        n_dates_per_feature > 2,
        2 * (1 - stats.t.cdf(np.abs(t_nw_values), df=n_dates_per_feature - 1)),
        1.0
    )
    
    # Build diagnostics
    diagnostics_df = pd.DataFrame({
        'feature': prior_approved_features,
        'ic_weighted': ic_weighted_filtered,
        't_nw': t_nw_values,
        'p_value': p_values,
        'n_dates': n_dates_per_feature,
    })
    
    # Add prior info to diagnostics
    if use_economic_priors:
        diagnostics_df['has_prior'] = True
        diagnostics_df['prior_sign'] = diagnostics_df['feature'].apply(
            lambda f: get_prior_for_feature(f).expected_sign.value 
            if get_prior_for_feature(f) else 'none'
        )
    
    stage_times['newey_west'] = time.time() - stage_start
    
    # Apply FDR control
    stage_start = time.time()
    print(f"[Formation FDR] Applying FDR control at level {fdr_level}...")
    reject, pvals_corrected, _, _ = multipletests(
        diagnostics_df['p_value'],
        alpha=fdr_level,
        method='fdr_bh'
    )
    
    diagnostics_df['fdr_reject'] = reject
    diagnostics_df['p_value_corrected'] = pvals_corrected
    stage_times['fdr_control'] = time.time() - stage_start
    
    approved_features = diagnostics_df[diagnostics_df['fdr_reject']]['feature'].tolist()
    n_approved_raw = len(approved_features)
    
    # =========================================================================
    # STEP 5: Short-lag protection if enabled
    # =========================================================================
    if config is not None and hasattr(config.features, 'enable_short_lag_protection'):
        fc = config.features
        if fc.enable_short_lag_protection:
            approved_features, _ = protect_short_lag_features(
                approved_features=approved_features,
                all_features=prior_approved_features,
                diagnostics_df=diagnostics_df,
                protect_count=fc.short_lag_protect_fdr,
                max_horizon=fc.short_lag_max_horizon,
                min_ic=fc.short_lag_min_ic,
                stage_name="Formation FDR",
            )
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Formation FDR with Economic Priors] SUMMARY:")
    print("=" * 70)
    print(f"  Features in (raw):             {X.shape[1]}")
    if use_economic_priors:
        print(f"  After economic prior filter:   {len(prior_approved_features)} "
              f"({100*len(prior_approved_features)/X.shape[1]:.1f}%)")
        print(f"    - Rejected (no prior):       {prior_stats.get('rejected_no_prior', 0)}")
        print(f"    - Rejected (wrong sign):     {prior_stats.get('rejected_wrong_sign', 0)}")
        print(f"    - Rejected (forbidden):      {prior_stats.get('rejected_forbidden', 0)}")
        print(f"    - Rejected (low IC):         {prior_stats.get('rejected_low_ic', 0)}")
    print(f"  After FDR control:             {len(approved_features)}")
    print(f"  IC stats - Mean: {diagnostics_df['ic_weighted'].mean():.4f}, "
          f"Max: {diagnostics_df['ic_weighted'].max():.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)
    
    del ic_daily
    gc.collect()
    
    return approved_features, diagnostics_df


def map_actual_features_to_priors(feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze your actual feature set against economic priors.
    
    Useful diagnostic: See which of your features have priors,
    which are missing, and which patterns you might want to add.
    """
    results = []
    for feature in feature_names:
        prior = get_prior_for_feature(feature)
        results.append({
            'feature': feature,
            'has_prior': prior is not None,
            'expected_sign': prior.expected_sign.value if prior else 'none',
            'min_ic': prior.min_abs_ic if prior else None,
            'confidence': prior.confidence if prior else None,
            'pattern_matched': prior.pattern if prior else None,
        })
    
    df = pd.DataFrame(results)
    
    # Summary
    n_with_prior = df['has_prior'].sum()
    n_without = len(df) - n_with_prior
    
    print(f"\nFeature-Prior Mapping Summary:")
    print(f"  Features with prior: {n_with_prior} ({100*n_with_prior/len(df):.1f}%)")
    print(f"  Features without prior: {n_without}")
    
    if n_without > 0:
        print(f"\n  Sample features without priors:")
        no_prior = df[~df['has_prior']]['feature'].tolist()
        for feat in no_prior[:10]:
            print(f"    - {feat}")
        if len(no_prior) > 10:
            print(f"    ... and {len(no_prior)-10} more")
    
    return df


# =============================================================================
# EXAMPLE: How to update your walk_forward_engine.py
# =============================================================================
"""
In walk_forward_engine.py, compute_formation_artifacts(), add this:

# BEFORE:
approved_features, fdr_diag = formation_fdr(
    X=formation_X_primitives,
    y=formation_y,
    dates=formation_dates,
    half_life=fc.formation_half_life,
    fdr_level=fc.fdr_level,
    config=config,
)

# AFTER:
from formation_priors_integration import formation_fdr_with_priors

approved_features, fdr_diag = formation_fdr_with_priors(
    X=formation_X_primitives,
    y=formation_y,
    dates=formation_dates,
    half_life=fc.formation_half_life,
    fdr_level=fc.fdr_level,
    config=config,
    use_economic_priors=config.features.use_economic_priors,  # Add to config
    require_sign_match=config.features.require_sign_match,    # Add to config
)
"""


# =============================================================================
# EXAMPLE: Update your config to include prior settings
# =============================================================================
"""
In your config.yaml:

features:
  # ... existing settings ...
  
  # Economic Prior Settings
  use_economic_priors: true         # Enable prior filtering
  require_sign_match: true          # IC sign must match theory
  allow_unprioried_features: false  # Strict: only theorized features
  min_prior_confidence: 0.3         # Minimum confidence to use prior
"""


if __name__ == "__main__":
    # Demo: Analyze features from a real panel
    import pyarrow.parquet as pq
    
    print("Loading panel to check feature coverage...")
    panel_path = r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet"
    
    # Just get column names (fast)
    parquet_file = pq.ParquetFile(panel_path)
    schema = parquet_file.schema_arrow
    all_columns = [field.name for field in schema]
    
    # Filter to feature columns (exclude metadata)
    exclude = {'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
               'Dividends', 'Stock Splits', 'y_cs_21d', 'FwdRet_21'}
    feature_cols = [c for c in all_columns if c not in exclude]
    
    print(f"\nFound {len(feature_cols)} feature columns in panel")
    
    # Map to priors
    df = map_actual_features_to_priors(feature_cols)
    
    # Show coverage by expected sign
    print("\nCoverage by Expected Sign:")
    sign_counts = df['expected_sign'].value_counts()
    for sign, count in sign_counts.items():
        print(f"  {sign}: {count}")
