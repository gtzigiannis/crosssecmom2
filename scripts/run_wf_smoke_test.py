#!/usr/bin/env python
"""
Walk-Forward Smoke Test for New Target Label (y_resid_z_21d)
============================================================

This script runs a short walk-forward backtest to validate the new 
cross-sectionally demeaned, risk-adjusted target label.

Usage:
    python -m scripts.run_wf_smoke_test [--n-windows N]

Example:
    python -m scripts.run_wf_smoke_test --n-windows 5

Diagnostics computed:
1. Label stats: mean, std, min, max of y_resid_z_21d
2. Per-date z-score properties (should be ~N(0,1))
3. Cross-sectional IC between model scores and targets
4. Simple performance metrics (Sharpe, long/short returns)
"""

import argparse
import copy
import os
import sys
from pathlib import Path

# Set threading env vars before numpy import
for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS", "LAPACK_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
from scipy import stats


def load_data_and_config(verbose: bool = True):
    """Load panel data, metadata, and config without modifying defaults."""
    from config import get_default_config
    
    # Get a fresh config (don't modify global defaults)
    config = get_default_config()
    
    # Load panel data
    if verbose:
        print(f"[load] Loading panel from {config.paths.panel_parquet}")
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    
    # Load metadata
    if verbose:
        print(f"[load] Loading metadata from {config.paths.universe_metadata_output}")
    universe_metadata = pd.read_csv(config.paths.universe_metadata_output)
    
    # Compute target labels if they don't exist
    target_cols = ['y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d']
    if not all(c in panel_df.columns for c in target_cols):
        if verbose:
            print(f"[load] Computing target labels (not found in panel)...")
        from label_engineering import compute_targets
        panel_df = compute_targets(panel_df, config, raw_return_col='FwdRet_21')
        if verbose:
            print(f"[load] Target labels computed: {[c for c in target_cols if c in panel_df.columns]}")
    
    return panel_df, universe_metadata, config


def compute_label_stats(panel_df: pd.DataFrame, target_col: str = 'y_resid_z_21d') -> dict:
    """
    Compute summary statistics for the target label.
    
    Returns dict with overall and per-date stats.
    """
    if target_col not in panel_df.columns:
        return {'error': f"Column {target_col} not found in panel"}
    
    target = panel_df[target_col]
    
    # Overall stats
    stats_dict = {
        'column': target_col,
        'n_total': len(target),
        'n_valid': target.notna().sum(),
        'pct_valid': target.notna().mean() * 100,
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'q01': target.quantile(0.01),
        'q99': target.quantile(0.99),
    }
    
    # Per-date stats (should be ~N(0,1) for z-scored target)
    if panel_df.index.nlevels == 2:
        dates = panel_df.index.get_level_values('Date')
    elif 'Date' in panel_df.columns:
        dates = panel_df['Date']
    else:
        dates = None
    
    if dates is not None:
        daily_means = target.groupby(dates).mean()
        daily_stds = target.groupby(dates).std()
        
        stats_dict['avg_daily_mean'] = daily_means.mean()
        stats_dict['std_daily_mean'] = daily_means.std()
        stats_dict['avg_daily_std'] = daily_stds.mean()
        stats_dict['std_daily_std'] = daily_stds.std()
    
    return stats_dict


def compute_cross_sectional_ic(
    scores: pd.Series,
    targets: pd.Series,
    dates: pd.DatetimeIndex
) -> dict:
    """
    Compute cross-sectional Spearman IC per date.
    
    Returns dict with mean IC, std IC, and fraction positive.
    """
    # Align data
    common_idx = scores.index.intersection(targets.index)
    if len(common_idx) == 0:
        return {'mean_ic': np.nan, 'std_ic': np.nan, 'frac_positive': np.nan, 'n_dates': 0}
    
    scores = scores.loc[common_idx]
    targets = targets.loc[common_idx]
    dates = dates.loc[common_idx] if hasattr(dates, 'loc') else dates
    
    # Group by date and compute IC
    ic_values = []
    unique_dates = np.unique(dates)
    
    for d in unique_dates:
        mask = dates == d
        s = scores[mask]
        t = targets[mask]
        
        # Remove NaN
        valid = s.notna() & t.notna()
        if valid.sum() < 5:
            continue
        
        ic, _ = stats.spearmanr(s[valid], t[valid])
        if not np.isnan(ic):
            ic_values.append(ic)
    
    if len(ic_values) == 0:
        return {'mean_ic': np.nan, 'std_ic': np.nan, 'frac_positive': np.nan, 'n_dates': 0}
    
    ic_arr = np.array(ic_values)
    return {
        'mean_ic': np.mean(ic_arr),
        'std_ic': np.std(ic_arr),
        'frac_positive': np.mean(ic_arr > 0),
        'n_dates': len(ic_arr),
        't_stat': np.mean(ic_arr) / (np.std(ic_arr) / np.sqrt(len(ic_arr))) if len(ic_arr) > 1 else np.nan,
    }


def run_smoke_test(
    n_windows: int = 5,
    skip_windows: int = 0,
    target_col: str = 'y_resid_z_21d',
    verbose: bool = True
) -> dict:
    """
    Run a short walk-forward smoke test with the new target label.
    
    Parameters
    ----------
    n_windows : int
        Number of WF windows to run (default 5)
    skip_windows : int
        Number of initial windows to skip (default 0, useful when early 
        formation windows have NaN in risk features)
    target_col : str
        Target column to use (default 'y_resid_z_21d')
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Comprehensive diagnostics dictionary
    """
    from config import get_default_config
    from universe_metadata import validate_universe_metadata
    from walk_forward_engine import run_walk_forward_backtest, analyze_performance
    
    # Load data
    panel_df, universe_metadata, base_config = load_data_and_config(verbose=verbose)
    
    # Create a copy of config for this run (don't modify defaults)
    config = copy.deepcopy(base_config)
    
    # Set the target column explicitly
    config.target.target_column = target_col
    
    # Skip early windows if specified (useful when early formation windows have NaN in risk features)
    if skip_windows > 0:
        config.compute.skip_rebalance_dates = skip_windows
    
    # Limit to n_windows for smoke test
    config.compute.max_rebalance_dates_for_debug = n_windows
    
    # Reduce verbosity for cleaner output
    config.debug.enable_accounting_debug = False
    
    # Validate metadata
    universe_metadata = validate_universe_metadata(universe_metadata, config)
    
    if verbose:
        print("\n" + "="*80)
        print("SMOKE TEST: TARGET LABEL VALIDATION")
        print("="*80)
        print(f"Target column: {target_col}")
        print(f"Windows to run: {n_windows}")
        if skip_windows > 0:
            print(f"Skipping first: {skip_windows} windows")
        print(f"Risk controls: {config.target.target_risk_control_columns}")
        print(f"Risk adjustment enabled: {config.target.target_use_risk_adjustment}")
    
    # =========================================================================
    # STEP 1: Label Statistics
    # =========================================================================
    if verbose:
        print("\n" + "-"*60)
        print("LABEL STATISTICS")
        print("-"*60)
    
    label_stats = compute_label_stats(panel_df, target_col)
    raw_stats = compute_label_stats(panel_df, 'y_raw_21d') if 'y_raw_21d' in panel_df.columns else {}
    
    if verbose:
        if 'error' in label_stats:
            print(f"\n[error] {label_stats['error']}")
            return {'error': label_stats['error']}
        
        print(f"\n{target_col}:")
        print(f"  Valid values: {label_stats['n_valid']:,} / {label_stats['n_total']:,} ({label_stats['pct_valid']:.1f}%)")
        print(f"  Mean: {label_stats['mean']:.6f}")
        print(f"  Std:  {label_stats['std']:.6f}")
        print(f"  Range: [{label_stats['min']:.4f}, {label_stats['max']:.4f}]")
        print(f"  Q1/Q99: [{label_stats['q01']:.4f}, {label_stats['q99']:.4f}]")
        
        if 'avg_daily_mean' in label_stats:
            print(f"\n  Per-date properties (should be ~N(0,1)):")
            print(f"    Avg daily mean: {label_stats['avg_daily_mean']:.6f} (should be ~0)")
            print(f"    Avg daily std:  {label_stats['avg_daily_std']:.6f} (should be ~1)")
    
    # =========================================================================
    # STEP 2: Run Walk-Forward
    # =========================================================================
    if verbose:
        print("\n" + "-"*60)
        print("RUNNING WALK-FORWARD BACKTEST")
        print("-"*60)
    
    results_df = run_walk_forward_backtest(
        panel_df=panel_df,
        universe_metadata=universe_metadata,
        config=config,
        model_type='supervised_binned',
        portfolio_method='cvxpy',
        verbose=verbose
    )
    
    if results_df is None or len(results_df) == 0:
        print("[error] No results from walk-forward backtest")
        return {'error': 'empty_results'}
    
    # =========================================================================
    # STEP 3: Performance Metrics
    # =========================================================================
    if verbose:
        print("\n" + "-"*60)
        print("PERFORMANCE SUMMARY")
        print("-"*60)
    
    perf_stats = analyze_performance(results_df, config)
    
    # =========================================================================
    # STEP 4: Cross-Sectional IC Analysis
    # =========================================================================
    # Note: We can't compute IC without storing per-date scores, which the WF engine
    # doesn't expose. Instead, report the IC from diagnostics if available.
    
    if verbose:
        print("\n" + "-"*60)
        print("IC DIAGNOSTICS (from WF diagnostics)")
        print("-"*60)
        
        # Check if diagnostics contain IC info
        if 'diagnostics' in results_df.attrs:
            diagnostics = results_df.attrs['diagnostics']
            ic_values = []
            for diag in diagnostics:
                if 'ic_values' in diag and diag['ic_values']:
                    # Average IC across features for this window
                    window_ics = [v for v in diag['ic_values'].values() if pd.notna(v)]
                    if window_ics:
                        ic_values.append(np.mean(window_ics))
            
            if ic_values:
                print(f"  Mean IC (across windows): {np.mean(ic_values):.4f}")
                print(f"  Std IC: {np.std(ic_values):.4f}")
                print(f"  Frac positive: {np.mean(np.array(ic_values) > 0):.2%}")
            else:
                print("  No IC values in diagnostics")
        else:
            print("  Diagnostics not available")
    
    # =========================================================================
    # STEP 5: Compile Results
    # =========================================================================
    diagnostics_out = {
        'target_column': target_col,
        'n_windows': n_windows,
        'label_stats': label_stats,
        'raw_label_stats': raw_stats,
        'performance': perf_stats,
        'results_df': results_df,
    }
    
    if verbose:
        print("\n" + "="*80)
        print("SMOKE TEST COMPLETE")
        print("="*80)
        
        # Quick summary
        print(f"\nSummary:")
        print(f"  Target: {target_col}")
        print(f"  Windows processed: {len(results_df)}")
        print(f"  Long-short return: {results_df['ls_return'].sum()*100:.2f}% cumulative")
        print(f"  Long leg return: {results_df['long_ret'].sum()*100:.2f}% cumulative")
        print(f"  Short leg return: {results_df['short_ret'].sum()*100:.2f}% cumulative")
        
        # Check z-score properties
        if 'avg_daily_mean' in label_stats:
            mean_ok = abs(label_stats['avg_daily_mean']) < 0.1
            std_ok = abs(label_stats['avg_daily_std'] - 1.0) < 0.2
            print(f"\n  Z-score validation:")
            print(f"    Daily mean ~0: {'PASS' if mean_ok else 'FAIL'} ({label_stats['avg_daily_mean']:.4f})")
            print(f"    Daily std ~1:  {'PASS' if std_ok else 'FAIL'} ({label_stats['avg_daily_std']:.4f})")
    
    return diagnostics_out


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward smoke test for new target label"
    )
    parser.add_argument(
        '--n-windows', '-n',
        type=int,
        default=5,
        help="Number of WF windows to run (default: 5)"
    )
    parser.add_argument(
        '--skip', '-s',
        type=int,
        default=0,
        help="Number of initial windows to skip (default: 0)"
    )
    parser.add_argument(
        '--target', '-t',
        type=str,
        default='y_resid_z_21d',
        help="Target column to use (default: y_resid_z_21d)"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Run smoke test
    results = run_smoke_test(
        n_windows=args.n_windows,
        skip_windows=args.skip,
        target_col=args.target,
        verbose=not args.quiet
    )
    
    if 'error' in results:
        print(f"\n[ERROR] {results['error']}")
        sys.exit(1)
    
    print("\n[done] Smoke test completed successfully!")
    return results


if __name__ == "__main__":
    main()
