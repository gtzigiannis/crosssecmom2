import pandas as pd
import numpy as np

df = pd.read_csv('D:/REPOSITORY/Data/crosssecmom2/cs_momentum_results.csv')

print("="*70)
print("DETAILED PERFORMANCE ANALYSIS")
print("="*70)

print("\n--- COST BREAKDOWN ---")
print(f"Transaction costs avg: {df['transaction_cost'].mean():.4f}%")
print(f"Borrowing costs avg: {df['borrow_cost'].mean():.4f}%")
print(f"Total costs avg: {(df['transaction_cost'] + df['borrow_cost']).mean():.4f}%")
print(f"Total costs cumulative: {(df['transaction_cost'] + df['borrow_cost']).sum():.2f}%")

print("\n--- POSITIONS ---")
print(f"Long positions avg: {df['n_long'].mean():.1f}")
print(f"Short positions avg: {df['n_short'].mean():.1f}")
print(f"Long positions range: {df['n_long'].min():.0f} - {df['n_long'].max():.0f}")
print(f"Short positions range: {df['n_short'].min():.0f} - {df['n_short'].max():.0f}")

print("\n--- COMPONENT RETURNS (per period) ---")
print(f"Long returns avg: {df['long_ret'].mean():.4f}%")
print(f"Short returns avg: {df['short_ret'].mean():.4f}%")
print(f"L/S returns avg: {df['ls_return'].mean():.4f}%")

print("\n--- GROSS RETURNS (before costs) ---")
gross_return = df['ls_return'] + df['transaction_cost'] + df['borrow_cost']
print(f"Gross return avg: {gross_return.mean():.4f}%")
print(f"Net return avg: {df['ls_return'].mean():.4f}%")
print(f"Cost drag: {(df['transaction_cost'] + df['borrow_cost']).mean():.4f}%")

print("\n--- WIN/LOSS ANALYSIS ---")
winners = df[df['ls_return'] > 0]
losers = df[df['ls_return'] <= 0]
print(f"Winning periods: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
print(f"Losing periods: {len(losers)} ({len(losers)/len(df)*100:.1f}%)")
print(f"Avg win: {winners['ls_return'].mean():.4f}%")
print(f"Avg loss: {losers['ls_return'].mean():.4f}%")
print(f"Win/Loss ratio: {abs(winners['ls_return'].mean() / losers['ls_return'].mean()):.2f}")

print("\n--- LONG vs SHORT PERFORMANCE ---")
long_wins = len(df[df['long_ret'] > 0])
short_wins = len(df[df['short_ret'] > 0])
print(f"Long win rate: {long_wins/len(df)*100:.1f}%")
print(f"Short win rate: {short_wins/len(df)*100:.1f}%")
print(f"Long total return: {((1 + df['long_ret']/100).prod() - 1)*100:.2f}%")
print(f"Short total return: {((1 + df['short_ret']/100).prod() - 1)*100:.2f}%")

print("\n--- TIME PERIODS ---")
df['date'] = pd.to_datetime(df['date'])
print(f"First date: {df['date'].min().strftime('%Y-%m-%d')}")
print(f"Last date: {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Duration: {(df['date'].max() - df['date'].min()).days} days")
print(f"Number of periods: {len(df)}")
