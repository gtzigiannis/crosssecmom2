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

# Configure logging to be less verbose
import logging
logging.getLogger('feature_selection').setLevel(logging.WARNING)
logging.getLogger('walk_forward_engine').setLevel(logging.WARNING)

# Import after setting environment
from config import get_default_config
from walk_forward_engine import run_walk_forward_backtest
from universe_metadata import validate_universe_metadata


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
