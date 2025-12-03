#!/usr/bin/env python
"""
Manual verification of forward return calculations.
This script verifies FwdRet_21 is calculated correctly as:
    FwdRet_21[t] = Close[t+21] / Close[t] - 1
"""

import pandas as pd
import numpy as np

# Paths
DATA_DIR = "D:/REPOSITORY/Data/crosssecmom2"
OHLCV_DIR = f"{DATA_DIR}/ohlcv"
PANEL_PATH = f"{DATA_DIR}/cs_momentum_features.parquet"

def verify_forward_returns():
    """Verify forward return calculation from raw OHLCV data."""
    print("=" * 60)
    print("MANUAL VERIFICATION OF FORWARD RETURNS")
    print("=" * 60)
    
    # Load raw OHLCV for VT
    print("\n[1] Loading raw OHLCV data for VT...")
    vt = pd.read_parquet(f"{OHLCV_DIR}/VT.parquet")
    vt = vt.sort_index()
    print(f"    VT data: {len(vt)} rows, from {vt.index.min().date()} to {vt.index.max().date()}")
    
    # Test several dates
    test_dates = [
        pd.Timestamp('2017-11-07'),  # Early in sample
        pd.Timestamp('2019-06-15'),  # Mid sample
        pd.Timestamp('2022-01-10'),  # Recent
    ]
    
    print("\n[2] Computing manual forward returns for test dates...")
    horizon = 21
    
    for test_date in test_dates:
        # Find closest trading day if test_date is not in index
        if test_date not in vt.index:
            # Find nearest date
            nearest_idx = vt.index.get_indexer([test_date], method='nearest')[0]
            test_date = vt.index[nearest_idx]
        
        close_t = vt.loc[test_date, 'Close']
        current_idx = vt.index.get_loc(test_date)
        
        if current_idx + horizon < len(vt):
            future_date = vt.index[current_idx + horizon]
            close_t21 = vt.loc[future_date, 'Close']
            manual_fwd_ret = (close_t21 / close_t) - 1
            
            print(f"\n    Date: {test_date.date()}")
            print(f"    Close[t]:      ${close_t:.4f}")
            print(f"    T+{horizon} date:    {future_date.date()}")
            print(f"    Close[t+{horizon}]:   ${close_t21:.4f}")
            print(f"    FwdRet_{horizon}:     {manual_fwd_ret:.6f} ({manual_fwd_ret*100:.2f}%)")
        else:
            print(f"\n    Date: {test_date.date()} - insufficient future data")
    
    # Now load the panel and compare
    print("\n" + "=" * 60)
    print("[3] Loading panel to compare FwdRet_21 values...")
    try:
        panel = pd.read_parquet(PANEL_PATH, columns=['Close', 'FwdRet_21'])
        print(f"    Panel shape: {panel.shape}")
        print(f"    Panel index: {panel.index.names}")
    except Exception as e:
        print(f"    ERROR loading panel: {e}")
        return
    
    # Check VT entries
    print("\n[4] Comparing panel FwdRet_21 with manual calculation for VT...")
    vt_panel = panel.xs('VT', level='Ticker')
    
    matches = 0
    mismatches = 0
    
    for test_date in test_dates:
        if test_date not in vt_panel.index:
            nearest_idx = vt_panel.index.get_indexer([test_date], method='nearest')[0]
            test_date = vt_panel.index[nearest_idx]
        
        try:
            panel_fwd = vt_panel.loc[test_date, 'FwdRet_21']
            
            # Compute manual
            close_t = vt.loc[test_date, 'Close']
            current_idx = vt.index.get_loc(test_date)
            if current_idx + horizon < len(vt):
                future_date = vt.index[current_idx + horizon]
                close_t21 = vt.loc[future_date, 'Close']
                manual_fwd_ret = (close_t21 / close_t) - 1
                
                diff = abs(panel_fwd - manual_fwd_ret)
                is_match = diff < 0.0001
                
                print(f"\n    Date: {test_date.date()}")
                print(f"    Panel FwdRet_21:  {panel_fwd:.6f}")
                print(f"    Manual FwdRet_21: {manual_fwd_ret:.6f}")
                print(f"    Difference:       {diff:.8f}")
                print(f"    Match:            {'✓ YES' if is_match else '✗ NO'}")
                
                if is_match:
                    matches += 1
                else:
                    mismatches += 1
        except KeyError:
            print(f"\n    Date: {test_date.date()} - not found in panel")
    
    # Broader validation - check a random sample
    print("\n" + "=" * 60)
    print("[5] Broader validation: random sample of 100 VT dates...")
    
    common_dates = vt_panel.index.intersection(vt.index)
    sample_size = min(100, len(common_dates) - horizon)
    sample_dates = np.random.choice(common_dates[:-(horizon+5)], size=sample_size, replace=False)
    
    matches = 0
    total = 0
    
    for date in sample_dates:
        try:
            panel_fwd = vt_panel.loc[date, 'FwdRet_21']
            if pd.isna(panel_fwd):
                continue
            
            close_t = vt.loc[date, 'Close']
            current_idx = vt.index.get_loc(date)
            if current_idx + horizon < len(vt):
                future_date = vt.index[current_idx + horizon]
                close_t21 = vt.loc[future_date, 'Close']
                manual_fwd_ret = (close_t21 / close_t) - 1
                
                diff = abs(panel_fwd - manual_fwd_ret)
                if diff < 0.0001:
                    matches += 1
                total += 1
        except:
            pass
    
    print(f"    Tested {total} dates, {matches} matched ({100*matches/total:.1f}%)")
    if matches == total:
        print("    ✓ ALL FwdRet_21 values match manual calculation!")
    else:
        print(f"    ✗ {total - matches} mismatches found!")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_forward_returns()
