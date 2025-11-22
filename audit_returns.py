"""
Comprehensive Returns Audit
============================
Deep dive into where losses are coming from in the cross-sectional momentum strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("COMPREHENSIVE RETURNS AUDIT")
print("="*80)

# Load results
results_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_results.csv")
df = pd.read_csv(results_path)
df['date'] = pd.to_datetime(df['date'])

print(f"\n[LOADED] {len(df)} periods from {df['date'].min().date()} to {df['date'].max().date()}")

# =============================================================================
# 1. PERIOD-BY-PERIOD ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("1. PERIOD-BY-PERIOD ANALYSIS")
print("="*80)

# Add derived columns
df['gross_return'] = df['ls_return'] + df['transaction_cost'] + df['borrow_cost']
df['total_cost'] = df['transaction_cost'] + df['borrow_cost']
df['long_contribution'] = df['long_ret'] * df['gross_long'] / (df['gross_long'] + df['gross_short'])
df['short_contribution'] = df['short_ret'] * df['gross_short'] / (df['gross_long'] + df['gross_short'])

print("\nWorst 10 Periods:")
worst = df.nsmallest(10, 'ls_return')[['date', 'ls_return', 'long_ret', 'short_ret', 'gross_long', 'gross_short', 'n_long', 'n_short']]
for idx, row in worst.iterrows():
    print(f"{row['date'].date()}: LS={row['ls_return']:7.2f}% | L={row['long_ret']:7.2f}% | S={row['short_ret']:7.2f}% | "
          f"GL={row['gross_long']:.2f} | GS={row['gross_short']:.2f} | #L={row['n_long']:.0f} #S={row['n_short']:.0f}")

print("\nBest 10 Periods:")
best = df.nlargest(10, 'ls_return')[['date', 'ls_return', 'long_ret', 'short_ret', 'gross_long', 'gross_short', 'n_long', 'n_short']]
for idx, row in best.iterrows():
    print(f"{row['date'].date()}: LS={row['ls_return']:7.2f}% | L={row['long_ret']:7.2f}% | S={row['short_ret']:7.2f}% | "
          f"GL={row['gross_long']:.2f} | GS={row['gross_short']:.2f} | #L={row['n_long']:.0f} #S={row['n_short']:.0f}")

# =============================================================================
# 2. LONG vs SHORT BREAKDOWN
# =============================================================================
print("\n" + "="*80)
print("2. LONG vs SHORT COMPONENT ANALYSIS")
print("="*80)

print("\nLONG SIDE:")
print(f"  Avg return per period: {df['long_ret'].mean():.4f}%")
print(f"  Win rate: {(df['long_ret'] > 0).mean()*100:.1f}%")
print(f"  Avg win: {df[df['long_ret'] > 0]['long_ret'].mean():.4f}%")
print(f"  Avg loss: {df[df['long_ret'] <= 0]['long_ret'].mean():.4f}%")
print(f"  Cumulative return: {((1 + df['long_ret']/100).prod() - 1)*100:.2f}%")
print(f"  Volatility: {df['long_ret'].std():.4f}%")
print(f"  Sharpe: {df['long_ret'].mean() / df['long_ret'].std():.3f}")

print("\nSHORT SIDE:")
print(f"  Avg return per period: {df['short_ret'].mean():.4f}%")
print(f"  Win rate: {(df['short_ret'] > 0).mean()*100:.1f}%")
print(f"  Avg win: {df[df['short_ret'] > 0]['short_ret'].mean():.4f}%")
print(f"  Avg loss: {df[df['short_ret'] <= 0]['short_ret'].mean():.4f}%")
print(f"  Cumulative return: {((1 + df['short_ret']/100).prod() - 1)*100:.2f}%")
print(f"  Volatility: {df['short_ret'].std():.4f}%")
print(f"  Sharpe: {df['short_ret'].mean() / df['short_ret'].std():.3f}")

print("\nCORRELATION:")
print(f"  Long vs Short correlation: {df['long_ret'].corr(df['short_ret']):.3f}")

# =============================================================================
# 3. LEVERAGE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3. LEVERAGE VERIFICATION")
print("="*80)

print(f"\nGross Long (avg): {df['gross_long'].mean():.4f}")
print(f"Gross Short (avg): {df['gross_short'].mean():.4f}")
print(f"Gross Total (avg): {(df['gross_long'] + df['gross_short']).mean():.4f}")
print(f"Net Exposure (avg): {(df['gross_long'] - df['gross_short']).mean():.4f}")

print(f"\nLeverage consistency check:")
print(f"  Min gross long: {df['gross_long'].min():.4f}")
print(f"  Max gross long: {df['gross_long'].max():.4f}")
print(f"  Min gross short: {df['gross_short'].min():.4f}")
print(f"  Max gross short: {df['gross_short'].max():.4f}")

# Expected leverage for reg_t_maintenance: 1.82 per side
expected_leverage = 1.0 / (0.25 + 0.30)
print(f"\nExpected leverage per side: {expected_leverage:.4f}")
print(f"Actual long leverage: {df['gross_long'].mean():.4f} (diff: {df['gross_long'].mean() - expected_leverage:.4f})")
print(f"Actual short leverage: {df['gross_short'].mean():.4f} (diff: {df['gross_short'].mean() - expected_leverage:.4f})")

# =============================================================================
# 4. CAPITAL EVOLUTION
# =============================================================================
print("\n" + "="*80)
print("4. CAPITAL EVOLUTION")
print("="*80)

df['cumulative_return'] = (1 + df['ls_return']/100).cumprod()
df['cumulative_cost'] = df['total_cost'].cumsum()

print(f"\nInitial capital: 1.0000")
print(f"Final capital: {df['capital'].iloc[-1]:.4f}")
print(f"Total return: {(df['capital'].iloc[-1] - 1.0)*100:.2f}%")
print(f"Cumulative costs: {df['cumulative_cost'].iloc[-1]:.4f}%")

print("\nCapital drawdown periods:")
df['peak'] = df['capital'].cummax()
df['drawdown'] = (df['capital'] - df['peak']) / df['peak'] * 100

worst_dd = df.nsmallest(5, 'drawdown')[['date', 'capital', 'drawdown']]
for idx, row in worst_dd.iterrows():
    print(f"  {row['date'].date()}: Capital={row['capital']:.4f}, DD={row['drawdown']:.2f}%")

# =============================================================================
# 5. TICKER-LEVEL ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("5. TICKER-LEVEL ANALYSIS")
print("="*80)

# Parse ticker lists
long_tickers_all = []
short_tickers_all = []

for idx, row in df.iterrows():
    try:
        long_list = eval(row['long_tickers']) if isinstance(row['long_tickers'], str) else []
        short_list = eval(row['short_tickers']) if isinstance(row['short_tickers'], str) else []
        long_tickers_all.extend(long_list)
        short_tickers_all.extend(short_list)
    except:
        pass

from collections import Counter

long_counts = Counter(long_tickers_all)
short_counts = Counter(short_tickers_all)

print("\nMost frequently held LONG positions:")
for ticker, count in long_counts.most_common(10):
    print(f"  {ticker}: {count} periods ({count/len(df)*100:.1f}%)")

print("\nMost frequently held SHORT positions:")
for ticker, count in short_counts.most_common(10):
    print(f"  {ticker}: {count} periods ({count/len(df)*100:.1f}%)")

# =============================================================================
# 6. COST ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("6. COST BREAKDOWN")
print("="*80)

print(f"\nTransaction costs:")
print(f"  Per period avg: {df['transaction_cost'].mean():.6f}%")
print(f"  Total: {df['transaction_cost'].sum():.4f}%")
print(f"  Min: {df['transaction_cost'].min():.6f}%")
print(f"  Max: {df['transaction_cost'].max():.6f}%")

print(f"\nBorrowing costs:")
print(f"  Per period avg: {df['borrow_cost'].mean():.6f}%")
print(f"  Total: {df['borrow_cost'].sum():.4f}%")
print(f"  Min: {df['borrow_cost'].min():.6f}%")
print(f"  Max: {df['borrow_cost'].max():.6f}%")

print(f"\nTurnover:")
print(f"  Long turnover avg: {df['turnover_long'].mean():.4f}")
print(f"  Short turnover avg: {df['turnover_short'].mean():.4f}")
print(f"  Total turnover avg: {df['turnover'].mean():.4f}")

# =============================================================================
# 7. TIME SERIES PATTERNS
# =============================================================================
print("\n" + "="*80)
print("7. TIME SERIES PATTERNS")
print("="*80)

# Group by year
df['year'] = df['date'].dt.year

print("\nAnnual performance:")
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]
    year_return = ((1 + year_data['ls_return']/100).prod() - 1) * 100
    print(f"  {year}: {len(year_data)} periods, Return: {year_return:7.2f}%, "
          f"Win rate: {(year_data['ls_return'] > 0).mean()*100:.1f}%")

# =============================================================================
# 8. ATTRIBUTION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("8. RETURN ATTRIBUTION")
print("="*80)

# Calculate contributions
total_gross = df['gross_long'].mean() + df['gross_short'].mean()
long_weight = df['gross_long'].mean() / total_gross
short_weight = df['gross_short'].mean() / total_gross

long_contrib = df['long_ret'].mean() * long_weight
short_contrib = df['short_ret'].mean() * short_weight
cost_drag = -df['total_cost'].mean()

print(f"\nReturn attribution (avg per period):")
print(f"  Long contribution: {long_contrib:.4f}% (weight: {long_weight:.2f})")
print(f"  Short contribution: {short_contrib:.4f}% (weight: {short_weight:.2f})")
print(f"  Cost drag: {cost_drag:.4f}%")
print(f"  Total: {long_contrib + short_contrib + cost_drag:.4f}%")
print(f"  Actual L/S return: {df['ls_return'].mean():.4f}%")

# =============================================================================
# 9. DIAGNOSTIC FLAGS
# =============================================================================
print("\n" + "="*80)
print("9. DIAGNOSTIC FLAGS")
print("="*80)

# Check for anomalies
issues = []

# Check leverage consistency
if abs(df['gross_long'].mean() - expected_leverage) > 0.01:
    issues.append(f"⚠️  Long leverage deviates from expected ({df['gross_long'].mean():.4f} vs {expected_leverage:.4f})")

if abs(df['gross_short'].mean() - expected_leverage) > 0.01:
    issues.append(f"⚠️  Short leverage deviates from expected ({df['gross_short'].mean():.4f} vs {expected_leverage:.4f})")

# Check net exposure
if abs(df['gross_long'].mean() - df['gross_short'].mean()) > 0.01:
    issues.append(f"⚠️  Not dollar-neutral (net: {(df['gross_long'] - df['gross_short']).mean():.4f})")

# Check short performance
if df['short_ret'].mean() < 0:
    issues.append(f"❌  Shorts are LOSING money on average ({df['short_ret'].mean():.4f}%)")
    issues.append(f"     This suggests model is ranking LOW performers that go UP")

# Check correlation
if df['long_ret'].corr(df['short_ret']) > 0.3:
    issues.append(f"⚠️  High correlation between long/short ({df['long_ret'].corr(df['short_ret']):.3f})")

if issues:
    print("\nISSUES DETECTED:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ No mechanical issues detected")

# =============================================================================
# 10. RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("10. INVESTIGATION RECOMMENDATIONS")
print("="*80)

print("\n1. Check if scores are inverted:")
print("   - Are high scores assigned to stocks that subsequently fall?")
print("   - Are low scores assigned to stocks that subsequently rise?")

print("\n2. Examine feature selection:")
print("   - Which features are being selected?")
print("   - Are their ICs actually predictive in the direction assumed?")

print("\n3. Review supervised binning:")
print("   - Are bins creating proper monotonic relationships?")
print("   - Is the forward return target aligned correctly?")

print("\n4. Check scoring mechanism:")
print("   - Is the aggregate score calculation correct?")
print("   - Are weights applied in the right direction?")

print("\n5. Validate universe filtering:")
print("   - Are we selecting liquid, tradable ETFs?")
print("   - Is the core universe appropriate?")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
