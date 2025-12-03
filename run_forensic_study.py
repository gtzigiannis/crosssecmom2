"""
Forensic Study Runner Script
=============================

Quick launcher for forensic analysis on walk-forward backtest results.

Usage:
    python run_forensic_study.py --results path/to/results.parquet
    python run_forensic_study.py --quick  # Fast mode (fewer Monte Carlo trials)
    python run_forensic_study.py --phase 0  # Run only Phase 0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.crosssecmom2.forensic_study import ForensicStudy, ForensicConfig


def find_latest_results(data_dir: Path) -> Path:
    """Find the most recent walk_forward_results file."""
    patterns = [
        "walk_forward_results*.parquet",
        "wf_results*.parquet",
        "results*.parquet",
    ]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            # Return most recently modified
            return max(files, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(f"No results file found in {data_dir}")


def load_data(results_path: Path, data_dir: Path):
    """Load all required data for forensic analysis."""
    print(f"\nLoading data...")
    
    # Load results
    print(f"  Results: {results_path}")
    results_df = pd.read_parquet(results_path)
    
    # Load panel data
    panel_path = data_dir / "panel_features.parquet"
    if panel_path.exists():
        print(f"  Panel: {panel_path}")
        panel_df = pd.read_parquet(panel_path)
    else:
        # Try alternate locations
        alt_paths = [
            data_dir / "ohlcv" / "panel_features.parquet",
            Path("D:/REPOSITORY/Data/crosssecmom2/panel_features.parquet"),
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"  Panel: {alt_path}")
                panel_df = pd.read_parquet(alt_path)
                break
        else:
            raise FileNotFoundError(f"Panel data not found. Tried: {panel_path}, {alt_paths}")
    
    # Load metadata
    metadata_path = data_dir / "universe_metadata.csv"
    if metadata_path.exists():
        print(f"  Metadata: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
    else:
        print(f"  Metadata: Not found, using empty DataFrame")
        metadata = pd.DataFrame()
    
    print(f"\n  Results shape: {results_df.shape}")
    print(f"  Panel shape: {panel_df.shape}")
    print(f"  Metadata shape: {metadata.shape}")
    
    return results_df, panel_df, metadata


def main():
    parser = argparse.ArgumentParser(description="Run forensic study on backtest results")
    parser.add_argument("--results", "-r", type=str, help="Path to walk_forward_results.parquet")
    parser.add_argument("--data-dir", "-d", type=str, 
                       default="D:/REPOSITORY/Data/crosssecmom2",
                       help="Path to data directory")
    parser.add_argument("--output-dir", "-o", type=str, 
                       default="forensic_outputs",
                       help="Output directory for results")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode: fewer Monte Carlo trials")
    parser.add_argument("--phase", "-p", type=str, 
                       help="Run specific phase only (0, 0b, 2, 4, 5)")
    parser.add_argument("--n-trials", "-n", type=int, default=1000,
                       help="Number of Monte Carlo trials")
    parser.add_argument("--n-jobs", "-j", type=int, default=-1,
                       help="Number of parallel jobs (-1 = all cores)")
    
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Find results file
    if args.results:
        results_path = Path(args.results)
    else:
        try:
            results_path = find_latest_results(data_dir)
            print(f"Using latest results: {results_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify --results path/to/results.parquet")
            return 1
    
    # Load data
    try:
        results_df, panel_df, metadata = load_data(results_path, data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Configure
    n_trials = 100 if args.quick else args.n_trials
    
    config = ForensicConfig(
        output_dir=output_dir,
        n_random_trials=n_trials,
        n_jobs=args.n_jobs,
        data_dir=data_dir,
    )
    
    # Create study
    study = ForensicStudy(
        results_df=results_df,
        panel_df=panel_df,
        universe_metadata=metadata,
        forensic_config=config
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
        # Run all
        study.run_all(skip_slow=args.quick)
    
    print(f"\n✓ Forensic study complete!")
    print(f"  Output: {output_dir.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
