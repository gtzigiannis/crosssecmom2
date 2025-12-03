"""
Single-Window Profiling Test for V3 Feature Selection Pipeline
===============================================================

This script runs a single walk-forward window to profile the computational
bottlenecks in the v3 feature selection pipeline.

Settings for profiling:
- max_rebalance_dates_for_debug = 1 (single window only)
- n_jobs = 1 (sequential execution for accurate timing)

Run from: D:\REPOSITORY\morias\Quant\strategies\crosssecmom2

Usage:
    python run_profiling_test.py

Expected output:
- Detailed timing for each stage in Formation (FDR, ElasticNet tuning)
- Detailed timing for each stage in Training window (v3 pipeline)
- Feature counts after each filter
- Per-bucket breakdown at each stage
"""

# MUST be set BEFORE importing numpy/pandas to avoid Windows Intel MKL threading crashes
import os
for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS", "LAPACK_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Configure logging to be less verbose
import logging
logging.getLogger('feature_selection').setLevel(logging.WARNING)
logging.getLogger('walk_forward_engine').setLevel(logging.WARNING)

# Import after setting environment
from config import get_default_config
from walk_forward_engine import run_walk_forward_backtest
from universe_metadata import validate_universe_metadata
from feature_engineering import get_feature_bucket, get_features_by_bucket


def compute_bucket_summary(features: List[str]) -> Dict[Tuple[str, str], int]:
    """Compute (family, horizon) bucket counts for a list of features."""
    buckets = get_features_by_bucket(features)
    return {k: len(v) for k, v in buckets.items()}


def print_bucket_breakdown(stage_name: str, features: List[str]) -> Dict[Tuple[str, str], int]:
    """Print and return per-bucket breakdown for a stage."""
    bucket_counts = compute_bucket_summary(features)
    
    # Aggregate by family
    family_totals = defaultdict(int)
    for (family, horizon), count in bucket_counts.items():
        family_totals[family] += count
    
    print(f"\n  [{stage_name}] {len(features)} features across {len(bucket_counts)} buckets")
    print(f"    By family: ", end="")
    for family, count in sorted(family_totals.items(), key=lambda x: -x[1])[:6]:
        print(f"{family}={count}", end="  ")
    if len(family_totals) > 6:
        print(f"... (+{len(family_totals)-6} more)")
    else:
        print()
    
    return bucket_counts


def print_family_breakdown_table(stages_data: Dict[str, List[str]]):
    """Print a comprehensive table showing family counts at each stage."""
    # Collect all families across all stages
    all_families = set()
    for stage_name, features in stages_data.items():
        for feat in features:
            family, _ = get_feature_bucket(feat)
            all_families.add(family)
    
    # Sort families: interaction first, then alphabetically
    families = sorted(all_families, key=lambda f: ('0' if f == 'interaction' else '1') + f)
    
    # Count by family for each stage
    stage_family_counts = {}
    for stage_name, features in stages_data.items():
        family_counts = defaultdict(int)
        for feat in features:
            family, _ = get_feature_bucket(feat)
            family_counts[family] += 1
        stage_family_counts[stage_name] = family_counts
    
    # Print table
    print("\n" + "=" * 120)
    print("FAMILY DISTRIBUTION ACROSS PIPELINE STAGES")
    print("=" * 120)
    
    # Header
    stage_names = list(stages_data.keys())
    header = f"{'Family':<15}"
    for name in stage_names:
        short_name = name[:10]
        header += f" {short_name:>10}"
    print(header)
    print("-" * 120)
    
    # Rows
    for family in families:
        row = f"{family:<15}"
        for stage_name in stage_names:
            count = stage_family_counts[stage_name].get(family, 0)
            if count > 0:
                row += f" {count:>10}"
            else:
                row += f" {'-':>10}"
        print(row)
    
    # Total row
    print("-" * 120)
    row = f"{'TOTAL':<15}"
    for stage_name in stage_names:
        total = len(stages_data[stage_name])
        row += f" {total:>10}"
    print(row)
    
    # Percentage rows
    print("-" * 120)
    row = f"{'% Interaction':<15}"
    for stage_name in stage_names:
        total = len(stages_data[stage_name])
        interaction_count = stage_family_counts[stage_name].get('interaction', 0)
        if total > 0:
            pct = 100.0 * interaction_count / total
            row += f" {pct:>9.1f}%"
        else:
            row += f" {'-':>10}"
    print(row)
    
    row = f"{'% Primitives':<15}"
    for stage_name in stage_names:
        total = len(stages_data[stage_name])
        interaction_count = stage_family_counts[stage_name].get('interaction', 0)
        primitive_count = total - interaction_count
        if total > 0:
            pct = 100.0 * primitive_count / total
            row += f" {pct:>9.1f}%"
        else:
            row += f" {'-':>10}"
    print(row)


def compute_correlation_stats(panel_df: pd.DataFrame, features: List[str], stage_name: str, 
                               t_start: pd.Timestamp, t_end: pd.Timestamp) -> Dict:
    """
    Compute correlation statistics for a set of features on training window data.
    
    Returns dict with:
    - mean_abs_corr: mean of absolute pairwise correlations (excluding diagonal)
    - max_abs_corr: maximum absolute pairwise correlation
    - median_abs_corr: median of absolute pairwise correlations
    - pct_high_corr: percentage of pairs with |corr| > 0.5
    - pct_very_high_corr: percentage of pairs with |corr| > 0.8
    """
    import numpy as np
    
    # Filter to features that exist in panel
    valid_features = [f for f in features if f in panel_df.columns]
    if len(valid_features) < 2:
        return None
    
    # Get data for training window
    mask = (panel_df.index.get_level_values('Date') >= t_start) & \
           (panel_df.index.get_level_values('Date') <= t_end)
    X = panel_df.loc[mask, valid_features].dropna()
    
    if len(X) < 100:
        return None
    
    # Compute correlation matrix
    corr_matrix = X.corr().values
    n = corr_matrix.shape[0]
    
    # Get upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_values = np.abs(corr_matrix[upper_tri_indices])
    
    # Handle any NaN values
    upper_tri_values = upper_tri_values[~np.isnan(upper_tri_values)]
    
    if len(upper_tri_values) == 0:
        return None
    
    n_pairs = len(upper_tri_values)
    
    # Find highly correlated pairs for debugging
    high_corr_pairs = []
    if np.max(upper_tri_values) > 0.9:
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix[i, j]) > 0.9:
                    high_corr_pairs.append((valid_features[i], valid_features[j], corr_matrix[i, j]))
    
    return {
        'stage': stage_name,
        'n_features': len(valid_features),
        'n_pairs': n_pairs,
        'mean_abs_corr': float(np.mean(upper_tri_values)),
        'median_abs_corr': float(np.median(upper_tri_values)),
        'max_abs_corr': float(np.max(upper_tri_values)),
        'std_abs_corr': float(np.std(upper_tri_values)),
        'pct_high_corr': float(100.0 * np.sum(upper_tri_values > 0.5) / n_pairs),
        'pct_very_high_corr': float(100.0 * np.sum(upper_tri_values > 0.8) / n_pairs),
        'high_corr_pairs': high_corr_pairs[:5],  # Top 5 high-corr pairs
    }


def print_correlation_analysis(panel_df: pd.DataFrame, stages_features: Dict[str, List[str]],
                                t_start: pd.Timestamp, t_end: pd.Timestamp):
    """Print correlation analysis for multiple stages."""
    print("\n" + "=" * 100)
    print("CORRELATION ANALYSIS (on training window data)")
    print("=" * 100)
    print(f"Training window: {t_start.date()} to {t_end.date()}")
    
    results = []
    for stage_name, features in stages_features.items():
        stats = compute_correlation_stats(panel_df, features, stage_name, t_start, t_end)
        if stats:
            results.append(stats)
    
    if not results:
        print("\n  [No correlation data available]")
        return
    
    # Print table
    print(f"\n  {'Stage':<20} {'Features':>8} {'Mean|ρ|':>10} {'Median|ρ|':>10} {'Max|ρ|':>10} {'%>0.5':>8} {'%>0.8':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    
    for r in results:
        print(f"  {r['stage']:<20} {r['n_features']:>8} {r['mean_abs_corr']:>10.3f} "
              f"{r['median_abs_corr']:>10.3f} {r['max_abs_corr']:>10.3f} "
              f"{r['pct_high_corr']:>7.1f}% {r['pct_very_high_corr']:>7.1f}%")
        
        # Show high-corr pairs if any
        if r.get('high_corr_pairs'):
            print(f"       ⚠ High-corr pairs (|ρ|>0.9):")
            for f1, f2, corr in r['high_corr_pairs']:
                # Truncate long feature names
                f1_short = f1[:30] + "..." if len(f1) > 33 else f1
                f2_short = f2[:30] + "..." if len(f2) > 33 else f2
                print(f"         {f1_short} ↔ {f2_short} = {corr:.3f}")
    
    # Key insights
    print("\n  [INTERPRETATION]")
    print("  - Mean|ρ|: Average absolute pairwise correlation (lower = less redundancy)")
    print("  - Max|ρ|: Maximum correlation pair (should decrease after redundancy filter)")
    print("  - %>0.5: Percentage of pairs with moderate+ correlation")
    print("  - %>0.8: Percentage of pairs with high correlation (collinearity concern)")


def run_profiling_test():
    """Run a single-window profiling test."""
    
    print("=" * 80)
    print("SINGLE-WINDOW PROFILING TEST - V3 FEATURE SELECTION PIPELINE")
    print("=" * 80)
    
    # Get default config
    config = get_default_config()
    
    # CRITICAL: Enable profiling mode
    config.compute.max_rebalance_dates_for_debug = 1  # Single window only
    config.compute.n_jobs = 1  # Sequential execution for accurate timing
    config.compute.verbose = True
    
    # Print profiling settings
    print("\n[PROFILING SETTINGS]")
    print(f"  max_rebalance_dates_for_debug: {config.compute.max_rebalance_dates_for_debug}")
    print(f"  n_jobs: {config.compute.n_jobs}")
    print(f"  formation_years: {config.features.formation_years}")
    print(f"  training_years: {config.features.training_years}")
    print(f"  formation_fdr_q_threshold: {config.features.formation_fdr_q_threshold}")
    print(f"  per_window_top_k: {config.features.per_window_top_k}")
    print(f"  corr_threshold: {config.features.corr_threshold}")
    
    # Load panel data
    print(f"\n[1] Loading panel data from {config.paths.panel_parquet}...")
    panel_start = time.time()
    
    if not Path(config.paths.panel_parquet).exists():
        print(f"[ERROR] Panel file not found: {config.paths.panel_parquet}")
        print("Please run feature engineering first: python main.py --step feature_eng")
        return None
    
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    panel_elapsed = time.time() - panel_start
    print(f"    Loaded panel: {panel_df.shape[0]:,} rows, {panel_df.shape[1]} columns ({panel_elapsed:.2f}s)")
    print(f"    Date range: {panel_df.index.get_level_values('Date').min().date()} to {panel_df.index.get_level_values('Date').max().date()}")
    
    # Load universe metadata
    print(f"\n[2] Loading universe metadata from {config.paths.universe_metadata_output}...")
    
    if not Path(config.paths.universe_metadata_output).exists():
        print(f"[WARNING] Metadata file not found, creating basic metadata")
        # Create basic metadata
        tickers = panel_df.index.get_level_values('Ticker').unique().tolist()
        universe_metadata = pd.DataFrame({
            'ticker': tickers,
            'family': 'UNKNOWN',
            'in_core_universe': True,
            'in_core_after_duplicates': True,
        })
    else:
        universe_metadata = pd.read_csv(config.paths.universe_metadata_output)
    
    # Validate metadata
    universe_metadata = validate_universe_metadata(universe_metadata, config)
    print(f"    Loaded metadata: {len(universe_metadata)} ETFs")
    
    # Run single-window backtest
    print("\n[3] Running single-window walk-forward backtest...")
    print("=" * 80)
    backtest_start = time.time()
    
    results_df = run_walk_forward_backtest(
        panel_df=panel_df,
        universe_metadata=universe_metadata,
        config=config,
        model_type='supervised_binned',  # Uses v3 pipeline
        portfolio_method='cvxpy',
        verbose=True
    )
    
    backtest_elapsed = time.time() - backtest_start
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING TEST COMPLETE")
    print("=" * 80)
    print(f"\n[TIMING SUMMARY]")
    print(f"  Panel loading:        {panel_elapsed:.2f}s")
    print(f"  Single-window total:  {backtest_elapsed:.2f}s ({backtest_elapsed/60:.2f} min)")
    
    # ==========================================================================
    # PER-BUCKET BREAKDOWN AT EACH STAGE
    # ==========================================================================
    if results_df is not None and hasattr(results_df, 'attrs') and 'diagnostics' in results_df.attrs:
        diagnostics = results_df.attrs['diagnostics']
        if len(diagnostics) > 0:
            # Get the first (and only) window's diagnostics
            first_diag = diagnostics[0]
            
            print("\n" + "=" * 80)
            print("PER-BUCKET FEATURE BREAKDOWN")
            print("=" * 80)
            
            # Formation stage feature lists (if available)
            if 'formation_artifacts' in first_diag:
                formation = first_diag['formation_artifacts']
                if 'feature_lists' in formation:
                    fl = formation['feature_lists']
                    
                    print("\n[FORMATION STAGES]")
                    
                    if 'primitive_base' in fl and fl['primitive_base']:
                        print_bucket_breakdown("1. Primitive Base Features", fl['primitive_base'])
                    
                    if 'all_interactions' in fl and fl['all_interactions']:
                        print_bucket_breakdown("2. All Interactions (mult + derived)", fl['all_interactions'])
                    
                    if 'approved_interactions' in fl and fl['approved_interactions']:
                        print_bucket_breakdown("3. Screened Interactions (top 150)", fl['approved_interactions'])
                    
                    if 'combined_pool' in fl and fl['combined_pool']:
                        print_bucket_breakdown("4. Combined Pool (primitive + approved)", fl['combined_pool'])
                    
                    if 'after_fdr' in fl and fl['after_fdr']:
                        print_bucket_breakdown("5. After Formation FDR", fl['after_fdr'])
                    
                    if 'after_bucket_redundancy' in fl and fl['after_bucket_redundancy']:
                        print_bucket_breakdown("6. After Bucket Redundancy", fl['after_bucket_redundancy'])
            
            # Training stage feature lists (if available)
            if 'feature_lists' in first_diag:
                fl = first_diag['feature_lists']
                
                print("\n[TRAINING STAGES]")
                
                if 'input_approved' in fl and fl['input_approved']:
                    print_bucket_breakdown("7. Input to Training (=Bucket Redundancy)", fl['input_approved'])
                
                if 'after_soft_ranking' in fl and fl['after_soft_ranking']:
                    print_bucket_breakdown("8. After Soft-Rank Top-K", fl['after_soft_ranking'])
                
                if 'after_redundancy' in fl and fl['after_redundancy']:
                    print_bucket_breakdown("9. After Training Redundancy Filter", fl['after_redundancy'])
                
                if 'after_lars' in fl and fl['after_lars']:
                    final_features = fl['after_lars']
                    bucket_counts = print_bucket_breakdown("10. FINAL (After LARS/Ridge)", final_features)
                    
                    # Detailed final breakdown
                    print("\n  [FINAL FEATURES - DETAILED BREAKDOWN]")
                    
                    # By family with horizon
                    family_horizon_counts = defaultdict(lambda: defaultdict(int))
                    for feat in final_features:
                        family, horizon = get_feature_bucket(feat)
                        family_horizon_counts[family][horizon] += 1
                    
                    for family in sorted(family_horizon_counts.keys(), key=lambda f: -sum(family_horizon_counts[f].values())):
                        total = sum(family_horizon_counts[family].values())
                        horizon_str = ", ".join(f"{hz}:{family_horizon_counts[family][hz]}" 
                                               for hz in ['H1', 'H2_10', 'H11_42', 'H43p'] 
                                               if family_horizon_counts[family][hz] > 0)
                        print(f"    {family:15s}: {total:3d}  ({horizon_str})")
                    
                    # Print sample feature names from final selection
                    print("\n  [SAMPLE FINAL FEATURES (first 15)]")
                    for i, feat in enumerate(final_features[:15]):
                        family, horizon = get_feature_bucket(feat)
                        print(f"    {i+1:2d}. ({family:12s}, {horizon:6s}) {feat[:60]}{'...' if len(feat) > 60 else ''}")
            
            # Print summary table
            print("\n" + "=" * 80)
            print("FEATURE FLOW SUMMARY TABLE")
            print("=" * 80)
            
            # Build summary table
            stage_data = []
            
            if 'formation_artifacts' in first_diag and 'feature_lists' in first_diag['formation_artifacts']:
                fl = first_diag['formation_artifacts']['feature_lists']
                if 'primitive_base' in fl:
                    stage_data.append(("Primitive Base", len(fl['primitive_base'])))
                if 'all_interactions' in fl:
                    stage_data.append(("All Interactions (mult+derived)", len(fl['all_interactions'])))
                if 'approved_interactions' in fl:
                    stage_data.append(("Screened Interactions (top 150)", len(fl['approved_interactions'])))
                if 'combined_pool' in fl:
                    stage_data.append(("Combined Pool (prim+approved)", len(fl['combined_pool'])))
                if 'after_fdr' in fl:
                    stage_data.append(("After FDR", len(fl['after_fdr'])))
                if 'after_bucket_redundancy' in fl:
                    stage_data.append(("After Bucket Redundancy", len(fl['after_bucket_redundancy'])))
            
            if 'feature_lists' in first_diag:
                fl = first_diag['feature_lists']
                if 'after_soft_ranking' in fl:
                    stage_data.append(("After Soft-Rank Top-K", len(fl['after_soft_ranking'])))
                if 'after_redundancy' in fl:
                    stage_data.append(("After Training Redundancy", len(fl['after_redundancy'])))
                if 'after_lars' in fl:
                    stage_data.append(("FINAL (LARS/Ridge)", len(fl['after_lars'])))
            
            print(f"\n  {'Stage':<35} {'Features':>10}")
            print(f"  {'-'*35} {'-'*10}")
            for stage_name, count in stage_data:
                print(f"  {stage_name:<35} {count:>10}")
            
            # ==========================================================================
            # COMPREHENSIVE FAMILY BREAKDOWN TABLE
            # ==========================================================================
            # Collect all feature lists for comprehensive table
            stages_for_table = {}
            
            if 'formation_artifacts' in first_diag and 'feature_lists' in first_diag['formation_artifacts']:
                fl = first_diag['formation_artifacts']['feature_lists']
                if 'primitive_base' in fl and fl['primitive_base']:
                    stages_for_table['Primitive'] = fl['primitive_base']
                if 'approved_interactions' in fl and fl['approved_interactions']:
                    stages_for_table['Screened'] = fl['approved_interactions']
                if 'combined_pool' in fl and fl['combined_pool']:
                    stages_for_table['Combined'] = fl['combined_pool']
                if 'after_fdr' in fl and fl['after_fdr']:
                    stages_for_table['AfterFDR'] = fl['after_fdr']
                if 'after_bucket_redundancy' in fl and fl['after_bucket_redundancy']:
                    stages_for_table['AfterRedund'] = fl['after_bucket_redundancy']
            
            if 'feature_lists' in first_diag:
                fl = first_diag['feature_lists']
                if 'after_soft_ranking' in fl and fl['after_soft_ranking']:
                    stages_for_table['SoftRank'] = fl['after_soft_ranking']
                if 'after_redundancy' in fl and fl['after_redundancy']:
                    stages_for_table['TrainRedund'] = fl['after_redundancy']
                if 'after_lars' in fl and fl['after_lars']:
                    stages_for_table['FINAL'] = fl['after_lars']
            
            if stages_for_table:
                print_family_breakdown_table(stages_for_table)
            
            # ==========================================================================
            # CORRELATION ANALYSIS
            # ==========================================================================
            # Reconstruct training window dates from config and rebalance date
            t0 = first_diag.get('date')
            if t0 is not None:
                training_days = int(config.features.training_years * 252)
                t_train_start = t0 - pd.Timedelta(days=training_days)
                t_train_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
                
                # Prepare stages for correlation analysis
                corr_stages = {}
                
                # After Bucket Redundancy (formation output)
                if 'formation_artifacts' in first_diag and 'feature_lists' in first_diag['formation_artifacts']:
                    fl = first_diag['formation_artifacts']['feature_lists']
                    if 'after_bucket_redundancy' in fl and fl['after_bucket_redundancy']:
                        corr_stages['After Bucket Redund'] = fl['after_bucket_redundancy']
                
                # Training stages
                if 'feature_lists' in first_diag:
                    fl = first_diag['feature_lists']
                    if 'after_soft_ranking' in fl and fl['after_soft_ranking']:
                        corr_stages['After Soft-Rank'] = fl['after_soft_ranking']
                    if 'after_redundancy' in fl and fl['after_redundancy']:
                        corr_stages['After Train Redund'] = fl['after_redundancy']
                    if 'after_lars' in fl and fl['after_lars']:
                        corr_stages['FINAL (LARS/Ridge)'] = fl['after_lars']
                
                if corr_stages:
                    print_correlation_analysis(panel_df, corr_stages, t_train_start, t_train_end)
    
    if results_df is not None and len(results_df) > 0:
        print(f"\n[RESULTS]")
        print(f"  Rebalance dates processed: {len(results_df)}")
        # Get the date column - could be 'date' or in index
        if 'date' in results_df.columns:
            print(f"  First rebalance date: {results_df['date'].iloc[0]}")
        elif 'rebalance_date' in results_df.columns:
            print(f"  First rebalance date: {results_df['rebalance_date'].iloc[0]}")
        else:
            print(f"  Columns: {list(results_df.columns)}")
        # Print key numeric columns only (avoid attrs with DataFrames causing print errors)
        numeric_cols = ['long_ret', 'short_ret', 'ls_return', 'turnover', 'transaction_cost', 'n_long', 'n_short', 'capital']
        display_cols = [c for c in numeric_cols if c in results_df.columns]
        if display_cols:
            print(results_df[display_cols].head().to_string())
        else:
            print(f"  (No numeric columns to display)")
    else:
        print("\n[WARNING] No results returned - check for errors above")
    
    return results_df


if __name__ == "__main__":
    results = run_profiling_test()
