#!/usr/bin/env python
"""
Run Forensic Study on Latest Backtest Results
==============================================

This script loads the most recent backtest results and runs the full
forensic analysis suite.

Usage:
    python run_forensic_analysis.py
    python run_forensic_analysis.py --quick  # Fast mode
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from config import get_default_config
from forensic_study import ForensicStudy, ForensicConfig


def find_latest_results(data_dir: Path) -> Path:
    """Find the most recent backtest results file."""
    # Look for backtest_results_*.parquet files
    result_files = list(data_dir.glob("backtest_results_*.parquet"))
    
    if not result_files:
        raise FileNotFoundError(f"No backtest result files found in {data_dir}")
    
    # Sort by modification time, most recent first
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return result_files[0]


def load_data(config):
    """Load all required data for forensic analysis."""
    print("="*80)
    print("LOADING DATA FOR FORENSIC ANALYSIS")
    print("="*80)
    
    data_dir = Path(config.paths.data_dir)
    
    # 1. Load latest backtest results
    results_path = find_latest_results(data_dir)
    print(f"\n[1] Loading results: {results_path.name}")
    results_df = pd.read_parquet(results_path)
    print(f"    Periods: {len(results_df)}")
    print(f"    Date range: {results_df.index.min()} to {results_df.index.max()}")
    
    # Check if diagnostics are attached
    if 'diagnostics' in results_df.attrs:
        print(f"    Diagnostics: {len(results_df.attrs['diagnostics'])} windows")
    else:
        print("    ⚠️  No diagnostics attached to results!")
    
    # 2. Load panel data
    print(f"\n[2] Loading panel: {config.paths.panel_parquet}")
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    print(f"    Shape: {panel_df.shape}")
    print(f"    Date range: {panel_df.index.get_level_values('Date').min()} to {panel_df.index.get_level_values('Date').max()}")
    
    # Check for forward returns column
    fwd_ret_col = f"FwdRet_{config.time.HOLDING_PERIOD_DAYS}"
    if fwd_ret_col in panel_df.columns:
        print(f"    ✓ Forward return column present: {fwd_ret_col}")
    else:
        print(f"    ⚠️  Forward return column missing: {fwd_ret_col}")
    
    # 3. Load universe metadata
    print(f"\n[3] Loading metadata: {config.paths.universe_metadata_output}")
    if Path(config.paths.universe_metadata_output).exists():
        metadata = pd.read_csv(config.paths.universe_metadata_output)
        print(f"    ETFs: {len(metadata)}")
    else:
        print("    ⚠️  Metadata file not found, creating empty DataFrame")
        metadata = pd.DataFrame()
    
    return results_df, panel_df, metadata


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run forensic study on backtest results")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode with fewer trials")
    parser.add_argument("--phase", "-p", type=str, help="Run specific phase only (0, 0b, 2, 4, 5)")
    parser.add_argument("--n-trials", "-n", type=int, default=1000, help="Number of Monte Carlo trials")
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print("FORENSIC STUDY - Cross-Sectional Momentum Strategy")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load config
    config = get_default_config()
    
    # Load data
    results_df, panel_df, metadata = load_data(config)
    
    # Configure forensic study
    n_trials = 100 if args.quick else args.n_trials
    output_dir = Path(config.paths.data_dir) / "forensic_outputs"
    
    forensic_config = ForensicConfig(
        output_dir=output_dir,
        n_random_trials=n_trials,
        n_jobs=-1,  # Use all cores
        data_dir=Path(config.paths.data_dir),
        holding_period_days=config.time.HOLDING_PERIOD_DAYS,
        bootstrap_samples=500 if args.quick else 1000,
    )
    
    print(f"\n[Config] Output directory: {output_dir}")
    print(f"[Config] Monte Carlo trials: {n_trials}")
    print(f"[Config] Holding period: {config.time.HOLDING_PERIOD_DAYS} days")
    
    # Create forensic study
    study = ForensicStudy(
        results_df=results_df,
        panel_df=panel_df,
        universe_metadata=metadata,
        config=config,
        forensic_config=forensic_config
    )
    
    # Run analysis
    if args.phase:
        phase = args.phase.lower()
        if phase == "0":
            study.run_phase_0_integrity_audit()
        elif phase == "0b":
            study.run_phase_0b_null_hypothesis_tests()
        elif phase == "2":
            study.run_phase_2_ranking_analysis()
        elif phase == "4":
            study.run_statistical_power_analysis()
        elif phase == "5":
            study.run_phase_5_counterfactual_analysis()
        else:
            print(f"Unknown phase: {phase}")
            print("Valid phases: 0, 0b, 2, 4, 5")
            return 1
    else:
        # Run all phases
        study.run_all(skip_slow=args.quick)
    
    # Print timing
    elapsed = datetime.now() - start_time
    print(f"\n{'='*80}")
    print(f"FORENSIC STUDY COMPLETE")
    print(f"Total time: {elapsed}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
