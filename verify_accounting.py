"""Verify accounting is correct in backtest results"""
import pandas as pd

df = pd.read_csv(r'D:\REPOSITORY\Data\crosssecmom2\cs_momentum_results.csv')

print("="*60)
print("ACCOUNTING VERIFICATION")
print("="*60)

# Check first period
row = df.iloc[0]
print(f"\nFirst Period ({row['date']}):")
print(f"  long_ret:          {row['long_ret']:>10.6f}")
print(f"  short_ret:         {row['short_ret']:>10.6f}")
print(f"  cash_ret:          {row['cash_ret']:>10.6f}")
print(f"  transaction_cost: -{row['transaction_cost']:>10.6f}")
print(f"  borrow_cost:      -{row['borrow_cost']:>10.6f}")
print(f"  " + "-"*30)

calc = row['long_ret'] + row['short_ret'] + row['cash_ret'] - row['transaction_cost'] - row['borrow_cost']
print(f"  Calculated:        {calc:>10.6f}")
print(f"  Actual ls_return:  {row['ls_return']:>10.6f}")
print(f"  Error:             {abs(calc - row['ls_return']):>10.2e}")
print(f"  ✓ Match: {abs(calc - row['ls_return']) < 1e-10}")

# Check a few more periods
print("\nRandom sample verification:")
for i in [10, 50, 100]:
    row = df.iloc[i]
    calc = row['long_ret'] + row['short_ret'] + row['cash_ret'] - row['transaction_cost'] - row['borrow_cost']
    error = abs(calc - row['ls_return'])
    status = "✓" if error < 1e-10 else "✗"
    print(f"  Period {i}: error={error:.2e} {status}")

# Overall statistics
print(f"\n{'='*60}")
print("OVERALL STATISTICS")
print(f"{'='*60}")
print(f"Total periods:     {len(df)}")
print(f"Mean long_ret:     {df['long_ret'].mean():>10.6f} ({df['long_ret'].mean()*100:>6.2f}%)")
print(f"Mean short_ret:    {df['short_ret'].mean():>10.6f} ({df['short_ret'].mean()*100:>6.2f}%)")
print(f"Mean cash_ret:     {df['cash_ret'].mean():>10.6f} ({df['cash_ret'].mean()*100:>6.2f}%)")
print(f"Mean txn_cost:     {df['transaction_cost'].mean():>10.6f} ({df['transaction_cost'].mean()*100:>6.2f}%)")
print(f"Mean borrow_cost:  {df['borrow_cost'].mean():>10.6f} ({df['borrow_cost'].mean()*100:>6.2f}%)")
print(f"Mean ls_return:    {df['ls_return'].mean():>10.6f} ({df['ls_return'].mean()*100:>6.2f}%)")
print(f"\nCumulative return: {(df['capital'].iloc[-1] - 1)*100:>10.2f}%")
print(f"Win rate:          {(df['ls_return'] > 0).sum() / len(df) * 100:>10.2f}%")

# Key issue diagnosis
print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")
gross_return = df['long_ret'].mean() + df['short_ret'].mean()
total_costs = df['transaction_cost'].mean() + df['borrow_cost'].mean()
print(f"Gross return (long+short): {gross_return:.6f}")
print(f"Total costs:               {total_costs:.6f}")
print(f"Net return:                {gross_return - total_costs:.6f}")
print(f"\nCosts as % of gross return: {total_costs / abs(gross_return) * 100 if gross_return != 0 else float('inf'):.1f}%")

if total_costs > abs(gross_return):
    print("\n⚠ WARNING: Costs exceed gross returns!")
    print("  This suggests high turnover or leverage relative to alpha.")
