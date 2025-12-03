#!/usr/bin/env python
"""
Side-by-Side Target Label Comparison
=====================================

This script runs walk-forward backtests with different target labels
and produces a comparison table showing the impact of the new
cross-sectionally demeaned, risk-adjusted target (y_resid_z_21d) vs
the raw forward return (y_raw_21d).

Usage:
    python -m scripts.compare_targets --max-windows 12

Output:
    - Side-by-side comparison table (printed and saved as CSV)
    - Metrics: IC, Sharpe, long/short returns, turnover
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


def load_data_and_config(verbose: bool = True):
    """Load panel data, metadata, and config without modifying defaults."""
    from config import get_default_config
    
    config = get_default_config()
    
    if verbose:
        print(f"[load] Loading panel from {config.paths.panel_parquet}")
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    
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


def run_single_target(
    panel_df: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    base_config,
    target_col: str,
    max_windows: int,
    verbose: bool = True
) -> dict:
    """
    Run WF backtest for a single target column.
    
    Returns dict with performance metrics.
    """
    from universe_metadata import validate_universe_metadata
    from walk_forward_engine import run_walk_forward_backtest, analyze_performance
    
    # Create a copy of config for this run
    config = copy.deepcopy(base_config)
    
    # Set target column
    config.target.target_column = target_col
    
    # Limit windows
    config.compute.max_rebalance_dates_for_debug = max_windows
    
    # Reduce verbosity for cleaner output
    config.debug.enable_accounting_debug = False
    
    # Validate metadata
    universe_metadata_copy = validate_universe_metadata(universe_metadata.copy(), config)
    
    if verbose:
        print(f"\n  Running WF with target={target_col}, {max_windows} windows...")
    
    # Run backtest
    results_df = run_walk_forward_backtest(
        panel_df=panel_df,
        universe_metadata=universe_metadata_copy,
        config=config,
        model_type='supervised_binned',
        portfolio_method='cvxpy',
        verbose=False  # Suppress per-window output
    )
    
    if results_df is None or len(results_df) == 0:
        print(f"  [warn] No results for {target_col}")
        return {'target': target_col, 'error': 'empty_results'}
    
    # Compute performance metrics
    perf = analyze_performance(results_df, config)
    
    # Extract key metrics
    metrics = {
        'target': target_col,
        'n_windows': len(results_df),
        # Returns
        'cumulative_ls': results_df['ls_return'].sum() * 100,
        'cumulative_long': results_df['long_ret'].sum() * 100,
        'cumulative_short': results_df['short_ret'].sum() * 100,
        # Averages
        'avg_ls_bps': results_df['ls_return'].mean() * 10000,
        'avg_long_bps': results_df['long_ret'].mean() * 10000,
        'avg_short_bps': results_df['short_ret'].mean() * 10000,
        # Risk-adjusted
        'sharpe_approx': (results_df['ls_return'].mean() / results_df['ls_return'].std() * np.sqrt(12)) 
                         if results_df['ls_return'].std() > 0 else np.nan,
        # Turnover (if available)
        'avg_turnover': results_df['turnover'].mean() if 'turnover' in results_df.columns else np.nan,
        # Win rate
        'win_rate': (results_df['ls_return'] > 0).mean() * 100,
    }
    
    # Store results_df for further analysis
    metrics['results_df'] = results_df
    
    return metrics


def compare_targets(
    max_windows: int = 12,
    targets: list = None,
    output_csv: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run WF for multiple targets and produce comparison table.
    
    Parameters
    ----------
    max_windows : int
        Number of WF windows per target (default 12)
    targets : list
        List of target columns to compare. Defaults to ['y_raw_21d', 'y_resid_z_21d']
    output_csv : str
        Path to save comparison CSV (optional)
    verbose : bool
        Print progress
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each target
    """
    if targets is None:
        targets = ['y_raw_21d', 'y_resid_z_21d']
    
    if verbose:
        print("\n" + "="*80)
        print("TARGET LABEL COMPARISON")
        print("="*80)
        print(f"Targets: {targets}")
        print(f"Windows per target: {max_windows}")
    
    # Load data once
    panel_df, universe_metadata, base_config = load_data_and_config(verbose=verbose)
    
    # Run each target
    all_metrics = []
    for target in targets:
        if target not in panel_df.columns:
            print(f"\n  [warn] Target {target} not in panel, skipping")
            continue
        
        metrics = run_single_target(
            panel_df=panel_df,
            universe_metadata=universe_metadata,
            base_config=base_config,
            target_col=target,
            max_windows=max_windows,
            verbose=verbose
        )
        all_metrics.append(metrics)
    
    if len(all_metrics) == 0:
        print("\n[error] No valid targets to compare")
        return pd.DataFrame()
    
    # Build comparison table
    comparison = []
    for m in all_metrics:
        if 'error' in m:
            continue
        comparison.append({
            'Target': m['target'],
            'Windows': m['n_windows'],
            'Cum L-S (%)': f"{m['cumulative_ls']:.2f}",
            'Cum Long (%)': f"{m['cumulative_long']:.2f}",
            'Cum Short (%)': f"{m['cumulative_short']:.2f}",
            'Avg L-S (bps)': f"{m['avg_ls_bps']:.1f}",
            'Sharpe (ann)': f"{m['sharpe_approx']:.2f}" if pd.notna(m['sharpe_approx']) else 'N/A',
            'Win Rate (%)': f"{m['win_rate']:.1f}",
            'Avg Turnover': f"{m['avg_turnover']:.1%}" if pd.notna(m['avg_turnover']) else 'N/A',
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Print comparison
    if verbose:
        print("\n" + "-"*80)
        print("COMPARISON TABLE")
        print("-"*80)
        print(comparison_df.to_string(index=False))
        
        # Improvement analysis
        if len(all_metrics) >= 2:
            m_raw = all_metrics[0]
            m_new = all_metrics[1]
            
            if 'error' not in m_raw and 'error' not in m_new:
                print("\n" + "-"*80)
                print("IMPROVEMENT ANALYSIS (new vs raw)")
                print("-"*80)
                
                ls_diff = m_new['cumulative_ls'] - m_raw['cumulative_ls']
                print(f"  Cumulative L-S change: {ls_diff:+.2f}% ({m_raw['cumulative_ls']:.2f}% -> {m_new['cumulative_ls']:.2f}%)")
                
                sharpe_raw = m_raw['sharpe_approx'] if pd.notna(m_raw['sharpe_approx']) else 0
                sharpe_new = m_new['sharpe_approx'] if pd.notna(m_new['sharpe_approx']) else 0
                print(f"  Sharpe change: {sharpe_new - sharpe_raw:+.2f} ({sharpe_raw:.2f} -> {sharpe_new:.2f})")
                
                wr_diff = m_new['win_rate'] - m_raw['win_rate']
                print(f"  Win rate change: {wr_diff:+.1f}% ({m_raw['win_rate']:.1f}% -> {m_new['win_rate']:.1f}%)")
    
    # Save to CSV if requested
    if output_csv:
        comparison_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n[saved] Comparison table saved to {output_csv}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return comparison_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare target labels side-by-side via walk-forward backtest"
    )
    parser.add_argument(
        '--max-windows', '-w',
        type=int,
        default=12,
        help="Number of WF windows per target (default: 12)"
    )
    parser.add_argument(
        '--targets', '-t',
        type=str,
        nargs='+',
        default=['y_raw_21d', 'y_resid_z_21d'],
        help="Target columns to compare (default: y_raw_21d y_resid_z_21d)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Path to save comparison CSV"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    comparison_df = compare_targets(
        max_windows=args.max_windows,
        targets=args.targets,
        output_csv=args.output,
        verbose=not args.quiet
    )
    
    if comparison_df.empty:
        print("\n[ERROR] Comparison failed")
        sys.exit(1)
    
    print("\n[done] Comparison completed successfully!")
    return comparison_df


if __name__ == "__main__":
    main()
