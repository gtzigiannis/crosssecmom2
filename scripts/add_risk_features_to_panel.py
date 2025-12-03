#!/usr/bin/env python
"""
Add multi-timeframe risk features (beta_VT, idio_vol) to existing panel.

This script loads the existing panel, downloads macro reference returns
(loading historical data from BEFORE the panel start to ensure no NaN),
computes risk factor features at 21/63/126-day windows, and saves an updated panel.

Features added per window (21, 63, 126):
- beta_VT_{window}: Rolling beta to VT (global equity)
- downside_beta_VT_{window}: Beta only when VT is negative
- idio_vol_{window}: Idiosyncratic volatility (residual from VT regression)
- r_squared_VT_{window}: R² of the VT regression
- drawdown_corr_VT_{window}: Correlation of drawdowns with VT
- beta_BNDW_{window}: Rolling beta to BNDW (global bonds)
- corr_VIX_{window}: Correlation with VIX
- corr_MOVE_{window}: Correlation with MOVE

Usage:
    python -m scripts.add_risk_features_to_panel [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager import CrossSecMomDataManager
from feature_engineering import add_risk_factor_features_to_panel


def main():
    parser = argparse.ArgumentParser(
        description="Add multi-timeframe risk features (beta_VT, idio_vol) to existing panel"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet",
        help="Path to input panel parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output parquet file (default: overwrites input)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="21,63,126",
        help="Comma-separated rolling windows (default: 21,63,126)",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=365,
        help="Extra historical days to load before panel start for warmup (default: 365)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all cores)",
    )
    args = parser.parse_args()
    
    # Parse windows
    windows = [int(w.strip()) for w in args.windows.split(',')]
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    
    print(f"Loading panel from: {input_path}")
    panel = pd.read_parquet(input_path)
    print(f"  Shape: {panel.shape}")
    print(f"  Index: {panel.index.names}")
    
    # Check if any risk features already exist (check for any window pattern)
    existing_risk_cols = [c for c in panel.columns if 
                         any(p in c for p in ['beta_VT_', 'idio_vol_', 'downside_beta_', 
                                               'r_squared_VT_', 'drawdown_corr_VT_',
                                               'beta_BNDW_', 'corr_VIX_', 'corr_MOVE_'])]
    if existing_risk_cols:
        print(f"\n[WARN] Risk columns already exist: {len(existing_risk_cols)} columns")
        print(f"       Examples: {existing_risk_cols[:5]}")
        response = input("Recompute and overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborting.")
            return
        # Drop existing risk columns
        panel = panel.drop(columns=existing_risk_cols)
        print(f"  Dropped {len(existing_risk_cols)} existing risk columns")
    
    # Initialize data manager
    data_dir = input_path.parent
    dm = CrossSecMomDataManager(data_dir=data_dir)
    
    # Get date range from panel
    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values('Date')
    else:
        dates = panel['Date']
    
    panel_start = dates.min()
    panel_end = dates.max()
    
    # Load macro returns from BEFORE panel start to eliminate NaN warmup
    # Need at least max(windows) + buffer days before panel start
    max_window = max(windows)
    warmup_days = max(args.history_days, max_window * 2)
    download_start = (panel_start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    download_end = panel_end.strftime("%Y-%m-%d")
    
    print(f"\nLoading macro reference returns with extended history...")
    print(f"  Panel date range: {panel_start.date()} to {panel_end.date()}")
    print(f"  Download from: {download_start} (extra {warmup_days} days for warmup)")
    print(f"  Windows: {windows}")
    
    macro_ref_returns = dm.load_macro_reference_returns(
        start_date=download_start,
        end_date=download_end,
    )
    
    if not macro_ref_returns:
        print("[ERROR] No macro reference returns loaded!")
        return
    
    print(f"\nLoaded returns for: {list(macro_ref_returns.keys())}")
    for name, series in macro_ref_returns.items():
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        print(f"  {name}: {len(series)} obs, {first_valid.date()} to {last_valid.date()}")
    
    # Deduplicate macro returns (keep last for each date)
    print("\nDeduplicating macro reference returns...")
    for name, series in macro_ref_returns.items():
        if series.index.duplicated().any():
            n_dups = series.index.duplicated().sum()
            print(f"  {name}: removing {n_dups} duplicate dates")
            macro_ref_returns[name] = series[~series.index.duplicated(keep='last')]
    
    # Add risk factor features
    # The function expects Ticker as a column, not as part of the index
    has_multiindex = isinstance(panel.index, pd.MultiIndex)
    if has_multiindex:
        print("\nResetting MultiIndex for feature computation...")
        panel = panel.reset_index()
    
    print(f"\nComputing risk factor features (windows={windows})...")
    panel = add_risk_factor_features_to_panel(
        panel,
        macro_ref_returns,
        windows=windows,
        n_jobs=args.n_jobs,
    )
    
    # Restore MultiIndex
    if has_multiindex and 'Date' in panel.columns and 'Ticker' in panel.columns:
        print("\nRestoring MultiIndex...")
        panel = panel.set_index(['Date', 'Ticker'])
    
    # Check what was added and NaN statistics
    risk_cols = [c for c in panel.columns if 
                 any(p in c for p in ['beta_VT_', 'idio_vol_', 'downside_beta_', 
                                       'r_squared_VT_', 'drawdown_corr_VT_',
                                       'beta_BNDW_', 'corr_VIX_', 'corr_MOVE_'])]
    print(f"\nNew risk columns: {len(risk_cols)}")
    
    # Show NaN statistics for key columns
    print("\nNaN statistics (first valid date per feature):")
    for window in windows:
        key_cols = [f'beta_VT_{window}', f'idio_vol_{window}']
        for col in key_cols:
            if col in panel.columns:
                series = panel[col]
                pct_nan = series.isna().mean() * 100
                # Get first valid index
                if isinstance(panel.index, pd.MultiIndex):
                    first_valid_date = panel.reset_index().groupby('Date')[col].apply(
                        lambda x: x.notna().any()
                    )
                    first_valid = first_valid_date[first_valid_date].index.min()
                else:
                    first_valid = series.first_valid_index()
                print(f"  {col}: {pct_nan:.2f}% NaN, first valid: {first_valid}")
    
    # Show statistics for key columns
    print("\nFeature statistics:")
    for col in ['beta_VT_63', 'idio_vol_63']:
        if col in panel.columns:
            stats = panel[col].describe()
            print(f"\n{col}:")
            print(f"  Count: {stats['count']:.0f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Save atomically (write to temp file first, then rename)
    # This prevents corruption if write is interrupted
    import tempfile
    import shutil
    
    print(f"\nSaving panel to: {output_path}")
    temp_path = output_path.with_suffix('.parquet.tmp')
    panel.to_parquet(temp_path, index=True, engine='pyarrow', compression='snappy')
    
    # Verify the temp file is readable
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(temp_path)
        print(f"  Verification: {pf.metadata.num_columns} columns, {pf.metadata.num_rows} rows")
    except Exception as e:
        print(f"  [ERROR] Written file failed verification: {e}")
        temp_path.unlink()
        raise RuntimeError(f"Failed to write valid parquet file: {e}")
    
    # Atomic rename
    if output_path.exists():
        output_path.unlink()
    temp_path.rename(output_path)
    
    print(f"  Final shape: {panel.shape}")
    print("\nDone!")


if __name__ == "__main__":
    main()
