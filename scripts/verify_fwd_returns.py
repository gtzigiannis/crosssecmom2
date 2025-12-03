#!/usr/bin/env python
"""Verify forward returns are correctly calculated (forward-looking, not backward)."""

import pandas as pd
import numpy as np
import sys

def main():
    print('Loading panel...')
    panel = pd.read_parquet('D:/REPOSITORY/Data/crosssecmom2/cs_momentum_features.parquet')
    print(f'Panel shape: {panel.shape}')
    print(f'Index: {panel.index.names}')
    print(f'Columns (first 10): {list(panel.columns[:10])}')
    
    # Reset index if needed
    if 'Ticker' in panel.index.names or 'Date' in panel.index.names:
        panel = panel.reset_index()
    
    # Check FwdRet_21 calculation
    print('\n' + '='*60)
    print('FORWARD RETURN VERIFICATION')
    print('='*60)
    
    # Get sample data for one ticker
    ticker = 'SPY'
    spy = panel[panel['Ticker'] == ticker].copy()
    spy = spy.sort_values('Date').reset_index(drop=True)
    print(f'\nSPY rows: {len(spy)}')
    
    # Pick a date in the middle
    mid_idx = len(spy) // 2
    sample_date = spy.loc[mid_idx, 'Date']
    sample_close = spy.loc[mid_idx, 'Close']
    future_close = spy.loc[mid_idx + 21, 'Close'] if mid_idx + 21 < len(spy) else None
    fwd_ret_in_panel = spy.loc[mid_idx, 'FwdRet_21']
    
    print(f'\nSample date: {sample_date}')
    print(f'Close on sample date: {sample_close:.4f}')
    
    if future_close is not None:
        print(f'Close 21 days later: {future_close:.4f}')
        expected_fwd_ret = (future_close / sample_close) - 1
        print(f'Expected FwdRet_21 (forward-looking): {expected_fwd_ret:.6f}')
    
    print(f'FwdRet_21 in panel: {fwd_ret_in_panel:.6f}')
    
    # Also compute backward return for comparison
    if mid_idx >= 21:
        past_close = spy.loc[mid_idx - 21, 'Close']
        backward_ret = (sample_close / past_close) - 1
        print(f'Backward return (past 21d): {backward_ret:.6f}')
    
    if future_close is not None:
        if abs(expected_fwd_ret - fwd_ret_in_panel) < 0.0001:
            print('\n✅ PASS: FwdRet_21 matches expected forward-looking return!')
        else:
            print('\n❌ FAIL: FwdRet_21 does NOT match expected value!')
            print(f'   Difference: {abs(expected_fwd_ret - fwd_ret_in_panel):.6f}')
            
            if mid_idx >= 21:
                if abs(backward_ret - fwd_ret_in_panel) < 0.0001:
                    print('   ⚠️ BUG: FwdRet_21 appears to be BACKWARD-looking!')
                    sys.exit(1)
    
    # Check for NaN pattern
    print(f'\n' + '='*60)
    print('NaN PATTERN CHECK (FwdRet_21)')
    print('='*60)
    last_21_nan = spy['FwdRet_21'].tail(21).isna().all()
    first_21_nan = spy['FwdRet_21'].head(21).isna().all()
    total_nan = spy['FwdRet_21'].isna().sum()
    
    print(f'Total NaN count: {total_nan}')
    print(f'Last 21 rows all NaN: {last_21_nan}')
    print(f'First 21 rows all NaN: {first_21_nan}')
    
    if last_21_nan and not first_21_nan:
        print('\n✅ PASS: NaN pattern is correct (forward-looking)')
    elif first_21_nan and not last_21_nan:
        print('\n❌ FAIL: NaN pattern is wrong (backward-looking)')
        sys.exit(1)
    
    # Check multiple tickers
    print(f'\n' + '='*60)
    print('CROSS-TICKER VERIFICATION')
    print('='*60)
    
    tickers = ['QQQ', 'VT', 'IWM', 'EEM', 'TLT']
    all_pass = True
    
    for ticker in tickers:
        df = panel[panel['Ticker'] == ticker].copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) < 50:
            print(f'{ticker}: Not enough data')
            continue
        
        # Test at multiple points
        test_points = [25, len(df)//2, len(df) - 30]
        
        for idx in test_points:
            if idx + 21 >= len(df):
                continue
                
            close_t = df.loc[idx, 'Close']
            close_t21 = df.loc[idx + 21, 'Close']
            expected = (close_t21 / close_t) - 1
            actual = df.loc[idx, 'FwdRet_21']
            
            if pd.isna(actual):
                print(f'{ticker} idx={idx}: FwdRet_21 is NaN (unexpected)')
                all_pass = False
            elif abs(expected - actual) > 0.0001:
                print(f'{ticker} idx={idx}: MISMATCH expected={expected:.6f}, actual={actual:.6f}')
                all_pass = False
    
    if all_pass:
        print('✅ All cross-ticker checks passed!')
    else:
        print('\n❌ Some checks failed!')
        sys.exit(1)
    
    print('\n' + '='*60)
    print('VERIFICATION COMPLETE - ALL CHECKS PASSED')
    print('='*60)

if __name__ == '__main__':
    main()
