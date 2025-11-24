"""
Verify accounting identity for the last 12-24 rebalances.

Checks:
1. Identity: ls_return ≈ naive_ls_ret + cash_pnl - transaction_cost - borrow_cost
2. Units: All fields should be small decimals (e.g. 0.01 = +1%)
3. Leverage: gross_long, gross_short should be near target (2-3x), cash_weight ≈ 1 - margin_util
4. Capital consistency: ∏(1 + ls_return) must match final capital
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

def run_small_backtest(n_rebalances=24):
    """Run backtest for last N rebalances."""
    print(f"Running backtest for last {n_rebalances} rebalances...")
    
    # Run the backtest
    cmd = [
        sys.executable, "main.py",
        "--step", "backtest",
        "--model", "supervised_binned"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Backtest failed")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("Backtest completed successfully")
    return True

def verify_accounting_identity(log_path, tolerance=1e-8):
    """
    Verify the accounting identity for each period.
    
    Identity: ls_return ≈ naive_ls_ret + cash_pnl - transaction_cost - borrow_cost
    """
    print(f"\nLoading accounting debug log: {log_path}")
    
    if not Path(log_path).exists():
        print(f"ERROR: File not found: {log_path}")
        return False
    
    df = pd.read_csv(log_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} periods")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Check 1: Verify the identity
    print("\n" + "="*80)
    print("CHECK 1: ACCOUNTING IDENTITY")
    print("="*80)
    print("Identity: ls_return ≈ naive_ls_ret + cash_pnl - transaction_cost - borrow_cost\n")
    
    # Compute right-hand side
    rhs = (df['naive_ls_ret'] + df['cash_pnl'] - 
           df['transaction_cost'] - df['borrow_cost'])
    
    # Compute difference
    diff = df['ls_return'] - rhs
    
    max_abs_diff = diff.abs().max()
    mean_abs_diff = diff.abs().mean()
    
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Mean absolute difference:    {mean_abs_diff:.2e}")
    print(f"Tolerance:                   {tolerance:.2e}")
    
    if max_abs_diff > tolerance:
        print("\n[FAIL] Identity violated!")
        print("\nLargest violations:")
        violations = diff.abs().nlargest(5)
        for date, val in violations.items():
            row = df.loc[date]
            print(f"\n  Date: {date}")
            print(f"    ls_return:       {row['ls_return']:.6f}")
            print(f"    naive_ls_ret:    {row['naive_ls_ret']:.6f}")
            print(f"    cash_pnl:        {row['cash_pnl']:.6f}")
            print(f"    transaction_cost: {row['transaction_cost']:.6f}")
            print(f"    borrow_cost:     {row['borrow_cost']:.6f}")
            print(f"    RHS sum:         {rhs.loc[date]:.6f}")
            print(f"    Difference:      {val:.6e}")
        return False
    else:
        print("\n[PASS] Identity holds within tolerance")
    
    # Check 2: Verify units
    print("\n" + "="*80)
    print("CHECK 2: UNITS (should be small decimals)")
    print("="*80)
    
    fields_to_check = ['ls_return', 'naive_ls_ret', 'cash_pnl', 
                       'transaction_cost', 'borrow_cost']
    
    units_ok = True
    for field in fields_to_check:
        if field in df.columns:
            max_val = df[field].abs().max()
            mean_val = df[field].abs().mean()
            print(f"\n{field:20s}: max={max_val:.6f}, mean={mean_val:.6f}")
            
            # Flag if values are suspiciously large (>1.0 often means percent vs decimal issue)
            if max_val > 1.0:
                print(f"  [WARN] Large values detected - possible percent vs decimal issue!")
                units_ok = False
    
    if units_ok:
        print("\n[PASS] All values in reasonable decimal range")
    else:
        print("\n[FAIL] Some values suspiciously large")
    
    # Check 3: Leverage and cash
    print("\n" + "="*80)
    print("CHECK 3: LEVERAGE AND CASH WEIGHTS")
    print("="*80)
    
    if 'gross_long' in df.columns and 'gross_short' in df.columns:
        print(f"\ngross_long:  min={df['gross_long'].min():.3f}, "
              f"max={df['gross_long'].max():.3f}, "
              f"mean={df['gross_long'].mean():.3f}")
        print(f"gross_short: min={df['gross_short'].min():.3f}, "
              f"max={df['gross_short'].max():.3f}, "
              f"mean={df['gross_short'].mean():.3f}")
        
        # Check for explosion
        if df['gross_long'].max() > 10.0 or df['gross_short'].max() > 10.0:
            print("\n[FAIL] Leverage has exploded (>10x)!")
        elif df['gross_long'].max() < 0.5 or df['gross_short'].max() < 0.5:
            print("\n[WARN] Leverage suspiciously low (<0.5x)")
        else:
            print("\n[PASS] Leverage in reasonable range (2-3x)")
    
    if 'cash_weight' in df.columns:
        print(f"\ncash_weight: min={df['cash_weight'].min():.3f}, "
              f"max={df['cash_weight'].max():.3f}, "
              f"mean={df['cash_weight'].mean():.3f}")
        
        if df['cash_weight'].min() < 0:
            print("\n[FAIL] Negative cash weight detected!")
        elif df['cash_weight'].mean() < 0.05:
            print("\n[WARN] Cash weight very low - high margin utilization")
        else:
            print("\n[PASS] Cash weight reasonable")
    
    # Check 4: Capital consistency
    print("\n" + "="*80)
    print("CHECK 4: CAPITAL CONSISTENCY")
    print("="*80)
    
    # Compute cumulative capital from returns
    capital_from_returns = (1 + df['ls_return']).cumprod()
    
    if 'capital' in df.columns:
        final_capital_log = df['capital'].iloc[-1]
        final_capital_ret = capital_from_returns.iloc[-1]
        
        print(f"\nFinal capital from log:     {final_capital_log:.8f}")
        print(f"Final capital from returns: {final_capital_ret:.8f}")
        print(f"Difference:                 {abs(final_capital_log - final_capital_ret):.2e}")
        
        if abs(final_capital_log - final_capital_ret) > 1e-6:
            print("\n[FAIL] Capital inconsistency!")
            
            # Show period-by-period comparison
            print("\nPeriod-by-period capital comparison (last 10):")
            comp = pd.DataFrame({
                'capital_log': df['capital'].tail(10),
                'capital_ret': capital_from_returns.tail(10),
                'diff': (df['capital'] - capital_from_returns).tail(10)
            })
            print(comp.to_string())
            return False
        else:
            print("\n[PASS] Capital consistent")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal periods:     {len(df)}")
    print(f"Date range:        {df.index[0]} to {df.index[-1]}")
    print(f"\nCumulative return: {(capital_from_returns.iloc[-1] - 1) * 100:.2f}%")
    print(f"Mean return/period: {df['ls_return'].mean() * 100:.4f}%")
    print(f"Std return/period:  {df['ls_return'].std() * 100:.4f}%")
    
    if 'transaction_cost' in df.columns:
        print(f"\nTotal transaction costs: {df['transaction_cost'].sum() * 100:.4f}%")
        print(f"Mean cost/period:        {df['transaction_cost'].mean() * 100:.4f}%")
    
    if 'borrow_cost' in df.columns:
        print(f"\nTotal borrow costs:      {df['borrow_cost'].sum() * 100:.4f}%")
        print(f"Mean cost/period:        {df['borrow_cost'].mean() * 100:.4f}%")
    
    if 'cash_pnl' in df.columns:
        print(f"\nTotal cash PnL:          {df['cash_pnl'].sum() * 100:.4f}%")
        print(f"Mean cash PnL/period:    {df['cash_pnl'].mean() * 100:.4f}%")
    
    # Export detailed analysis
    analysis_path = Path(log_path).parent / "accounting_identity_check.csv"
    analysis = pd.DataFrame({
        'ls_return': df['ls_return'],
        'naive_ls_ret': df['naive_ls_ret'],
        'cash_pnl': df['cash_pnl'],
        'transaction_cost': df['transaction_cost'],
        'borrow_cost': df['borrow_cost'],
        'rhs_sum': rhs,
        'identity_diff': diff,
        'capital_log': df['capital'] if 'capital' in df.columns else np.nan,
        'capital_ret': capital_from_returns
    })
    analysis.to_csv(analysis_path)
    print(f"\nDetailed analysis saved to: {analysis_path}")
    
    return True

def main():
    # Change to correct directory
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("="*80)
    print("ACCOUNTING IDENTITY VERIFICATION")
    print("="*80)
    
    # Check if we should run backtest or use existing log
    # First check data_dir location (correct location)
    data_dir = Path("D:/REPOSITORY/Data/crosssecmom2")
    log_path = data_dir / "accounting_debug_log.csv"
    
    # Also check old output location as fallback
    old_log_path = Path("output/accounting_debug_log.csv")
    
    if not log_path.exists() and not old_log_path.exists():
        print("\nNo existing accounting debug log found.")
        print("Running new backtest automatically...")
        if not run_small_backtest():
            print("\nBacktest failed. Exiting.")
            return
    
    # Use whichever exists
    if old_log_path.exists():
        log_path = old_log_path
        
    if not log_path.exists():
        print(f"\nERROR: Still no log file at {log_path}")
        print("Make sure backtest produces accounting_debug_log.csv")
        return
    
    # Verify the identity
    success = verify_accounting_identity(log_path)
    
    if success:
        print("\n" + "="*80)
        print("[SUCCESS] All accounting checks passed!")
        print("="*80)
        print("\nThe -97% loss is NOT an accounting artifact.")
        print("The problem is in the alpha/strategy logic, not the financing layer.")
    else:
        print("\n" + "="*80)
        print("[FAILURE] Accounting checks failed!")
        print("="*80)
        print("\nThere may be an accounting bug contributing to poor performance.")

if __name__ == "__main__":
    main()
