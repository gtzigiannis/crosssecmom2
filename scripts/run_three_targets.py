#!/usr/bin/env python
"""
Run Walk-Forward Backtest for Three Target Labels
=================================================

This script runs the backtest separately for each target label:
1. y_raw_21d - Raw 21-day forward return
2. y_cs_21d - Cross-sectionally demeaned return
3. y_resid_z_21d - Risk-adjusted z-score return

Each run performs FRESH feature selection from scratch because the 
feature selection pipeline (interaction screening, FDR, etc.) uses 
the target label for supervised filtering.
"""

import sys
import os
import copy
import numpy as np
import pandas as pd

# Set threading env vars before numpy import
for var in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 
            'NUMEXPR_NUM_THREADS', 'BLAS_NUM_THREADS', 'LAPACK_NUM_THREADS'):
    os.environ.setdefault(var, '1')

from config import get_default_config
from universe_metadata import validate_universe_metadata
from walk_forward_engine import run_walk_forward_backtest, analyze_performance


def run_single_backtest(panel_df, universe_metadata, target_col, n_windows=5, skip_windows=0, verbose=True):
    """
    Run a single backtest with a specific target column.
    
    This creates a fresh config and runs the full walk-forward backtest
    with fresh feature selection for each window.
    
    Parameters
    ----------
    skip_windows : int
        Number of initial windows to skip (useful to avoid NaN period in early data)
    """
    config = get_default_config()
    cfg = copy.deepcopy(config)
    
    # Set target column
    cfg.target.target_column = target_col
    # Run extra windows to account for skipping
    cfg.compute.max_rebalance_dates_for_debug = n_windows + skip_windows
    cfg.debug.enable_accounting_debug = False
    
    # Validate metadata (fresh copy)
    meta = validate_universe_metadata(universe_metadata.copy(), cfg)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"BACKTEST: {target_col}")
        print(f"{'='*80}")
        print(f"Target column: {cfg.target.target_column}")
        print(f"Windows: {n_windows}")
    
    # Run backtest - this does fresh feature selection each window
    results_df = run_walk_forward_backtest(
        panel_df=panel_df,
        universe_metadata=meta,
        config=cfg,
        model_type='supervised_binned',
        portfolio_method='cvxpy',
        verbose=verbose
    )
    
    if results_df is None or len(results_df) == 0:
        print(f"[error] No results for {target_col}")
        return None, None, None
    
    # Get performance metrics
    perf = analyze_performance(results_df, cfg)
    
    # Extract feature selection diagnostics
    feat_sel_info = []
    if 'diagnostics' in results_df.attrs:
        for i, diag in enumerate(results_df.attrs['diagnostics']):
            # Formation artifacts diagnostics
            form = diag.get('formation_artifacts', {})
            form_diag = form.get('formation_diagnostics', {})
            int_diag = form.get('interaction_diagnostics', {})
            
            # Per-window pipeline diagnostics (from train_window_model -> per_window_pipeline_v3)
            feature_flow = diag.get('feature_flow', {})
            
            info = {
                'window': i + 1,
                'date': diag.get('date', ''),
                # Formation stage
                'n_primitive_base': form_diag.get('n_primitive_base', 0),
                'n_interactions_screened': int_diag.get('n_candidates', 0) if int_diag else 0,
                'n_interactions_approved': int_diag.get('n_approved', 0) if int_diag else 0,
                'n_combined_pool': form_diag.get('n_combined_pool', 0),
                'n_fdr_approved': form_diag.get('n_approved', 0),
                'n_after_redundancy': form_diag.get('n_after_redundancy', 0),
                # Per-window pipeline stage (from feature_flow)
                'n_soft_ranking': feature_flow.get('after_soft_ranking', 0),
                'n_window_redundancy': feature_flow.get('after_redundancy', 0),
                'n_lasso': feature_flow.get('after_lars', 0),
                'n_short_lag': feature_flow.get('short_lag_added', 0),
                'n_final': feature_flow.get('after_short_lag', diag.get('n_features', 0)),
                # Selected features
                'selected_features': diag.get('selected_features', []),
            }
            feat_sel_info.append(info)
    
    return results_df, perf, feat_sel_info


def main():
    # Load data once
    config = get_default_config()
    
    print(f"Loading panel from {config.paths.panel_parquet}", flush=True)
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    
    print(f"Loading metadata from {config.paths.universe_metadata_output}", flush=True)
    universe_metadata = pd.read_csv(config.paths.universe_metadata_output)
    
    # Check for target columns, compute if needed
    target_cols = ['y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d']
    if not all(c in panel_df.columns for c in target_cols):
        print(f"Computing target labels...", flush=True)
        from label_engineering import compute_targets
        panel_df = compute_targets(panel_df, config, raw_return_col='FwdRet_21')
        print(f"Target labels computed", flush=True)
    
    # Store results
    all_results = {}
    
    # ========================================================================
    # BACKTEST 1: y_raw_21d
    # ========================================================================
    results1, perf1, feat1 = run_single_backtest(
        panel_df, universe_metadata, 'y_raw_21d', n_windows=10, verbose=True
    )
    if results1 is not None:
        all_results['y_raw_21d'] = {
            'ls_return': results1['ls_return'].sum() * 100,
            'long_ret': results1['long_ret'].sum() * 100,
            'short_ret': results1['short_ret'].sum() * 100,
            'sharpe': perf1.get('sharpe', np.nan),
            'max_dd': perf1.get('max_dd', np.nan),
            'feature_selection': feat1,
        }
    
    # ========================================================================
    # BACKTEST 2: y_cs_21d
    # ========================================================================
    results2, perf2, feat2 = run_single_backtest(
        panel_df, universe_metadata, 'y_cs_21d', n_windows=10, verbose=True
    )
    if results2 is not None:
        all_results['y_cs_21d'] = {
            'ls_return': results2['ls_return'].sum() * 100,
            'long_ret': results2['long_ret'].sum() * 100,
            'short_ret': results2['short_ret'].sum() * 100,
            'sharpe': perf2.get('sharpe', np.nan),
            'max_dd': perf2.get('max_dd', np.nan),
            'feature_selection': feat2,
        }
    
    # ========================================================================
    # BACKTEST 3: y_resid_z_21d
    # ========================================================================
    results3, perf3, feat3 = run_single_backtest(
        panel_df, universe_metadata, 'y_resid_z_21d', n_windows=10, verbose=True
    )
    if results3 is not None:
        all_results['y_resid_z_21d'] = {
            'ls_return': results3['ls_return'].sum() * 100,
            'long_ret': results3['long_ret'].sum() * 100,
            'short_ret': results3['short_ret'].sum() * 100,
            'sharpe': perf3.get('sharpe', np.nan),
            'max_dd': perf3.get('max_dd', np.nan),
            'feature_selection': feat3,
        }
    
    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: THREE TARGET LABELS COMPARISON")
    print("="*80)
    
    print("\n┌─────────────────┬──────────────┬────────────┬────────────┬─────────┐")
    print("│ Target          │ L/S Return   │ Long Leg   │ Short Leg  │ Sharpe  │")
    print("├─────────────────┼──────────────┼────────────┼────────────┼─────────┤")
    
    for target, data in all_results.items():
        print(f"│ {target:<15} │ {data['ls_return']:>10.2f}% │ {data['long_ret']:>8.2f}% │ {data['short_ret']:>8.2f}% │ {data['sharpe']:>7.2f} │")
    
    print("└─────────────────┴──────────────┴────────────┴────────────┴─────────┘")
    
    # ========================================================================
    # FEATURE SELECTION DIAGNOSTICS
    # ========================================================================
    print("\n" + "-"*80)
    print("FEATURE SELECTION PIPELINE BY TARGET")
    print("-"*80)
    
    for target, data in all_results.items():
        print(f"\n{target}:")
        feat_info = data.get('feature_selection', [])
        if feat_info:
            print("  Window   Date         IntApproved  FDR     Bucket  LassoCV  ShortLag  Final")
            print("  ------   ----------   -----------  ------  ------  -------  --------  -----")
            for f in feat_info:
                date_str = str(f.get('date', ''))[:10]
                print(f"  {f['window']:^6}   {date_str:10}   {f['n_interactions_approved']:^11}  {f['n_fdr_approved']:^6}  {f['n_after_redundancy']:^6}  {f['n_lasso']:^7}  {f['n_short_lag']:^8}  {f['n_final']:^5}")
        else:
            print("  No diagnostics available")
    
    print("\n[done] All backtests completed!")
    
    return all_results


if __name__ == "__main__":
    results = main()
