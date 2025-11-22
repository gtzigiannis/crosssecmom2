"""
Market Regime Analysis: Is Momentum Working?
=============================================
Check if cross-sectional momentum is actually working in this time period.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("MOMENTUM REGIME ANALYSIS")
print("="*80)

# Load panel
panel_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet")
panel = pd.read_parquet(panel_path)

# Load results to get backtest period
results_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_results.csv")
results = pd.read_csv(results_path)
results['date'] = pd.to_datetime(results['date'])

backtest_start = results['date'].min()
backtest_end = results['date'].max()

print(f"\nBacktest period: {backtest_start.date()} to {backtest_end.date()}")

# Filter panel to backtest period
panel_bt = panel[
    (panel.index.get_level_values('Date') >= backtest_start) &
    (panel.index.get_level_values('Date') <= backtest_end)
].copy()

# =============================================================================
# 1. SIMPLE MOMENTUM TEST
# =============================================================================
print("\n" + "="*80)
print("1. SIMPLE MOMENTUM BACKTEST (Top/Bottom Decile)")
print("="*80)

# Sample rebalance dates (monthly)
dates = panel_bt.index.get_level_values('Date').unique()
rebalance_dates = sorted(dates)[::21][:20]  # Every 21 days, first 20 periods

simple_results = []

for date in rebalance_dates:
    if date not in panel.index.get_level_values('Date'):
        continue
    
    cross_section = panel.loc[date].copy()
    
    # Need both momentum and forward return
    if 'Close%-21' not in cross_section.columns or 'FwdRet_21' not in cross_section.columns:
        continue
    
    valid = cross_section[['Close%-21', 'FwdRet_21']].dropna()
    
    if len(valid) < 20:
        continue
    
    # Sort by momentum
    sorted_by_mom = valid.sort_values('Close%-21', ascending=False)
    
    # Top and bottom deciles
    n_decile = max(1, len(sorted_by_mom) // 10)
    
    top_decile = sorted_by_mom.head(n_decile)
    bottom_decile = sorted_by_mom.tail(n_decile)
    
    # Average forward returns
    top_fwd = top_decile['FwdRet_21'].mean()
    bottom_fwd = bottom_decile['FwdRet_21'].mean()
    spread = top_fwd - bottom_fwd
    
    simple_results.append({
        'date': date,
        'top_momentum_return': top_fwd,
        'bottom_momentum_return': bottom_fwd,
        'spread': spread
    })

simple_df = pd.DataFrame(simple_results)

print(f"\nSimple momentum backtest ({len(simple_df)} periods):")
print(f"  Top decile avg return: {simple_df['top_momentum_return'].mean():.4f}")
print(f"  Bottom decile avg return: {simple_df['bottom_momentum_return'].mean():.4f}")
print(f"  Spread (top - bottom): {simple_df['spread'].mean():.4f}")
print(f"  Win rate (spread > 0): {(simple_df['spread'] > 0).mean()*100:.1f}%")

if simple_df['spread'].mean() < 0:
    print("\n  ❌ MOMENTUM IS INVERTED - Bottom performers outperform top performers!")
elif simple_df['spread'].mean() < 0.001:
    print("\n  ⚠️  MOMENTUM IS WEAK - Spread close to zero")
else:
    print("\n  ✓ MOMENTUM IS WORKING")

# =============================================================================
# 2. TIME SERIES OF MOMENTUM EFFECTIVENESS
# =============================================================================
print("\n" + "="*80)
print("2. MOMENTUM EFFECTIVENESS OVER TIME")
print("="*80)

# Group by year
simple_df['year'] = pd.to_datetime(simple_df['date']).dt.year

print(f"\n{'Year':<10} {'Periods':>10} {'Avg Spread':>15} {'Win Rate':>12}")
print("-" * 50)

for year in sorted(simple_df['year'].unique()):
    year_data = simple_df[simple_df['year'] == year]
    avg_spread = year_data['spread'].mean()
    win_rate = (year_data['spread'] > 0).mean() * 100
    print(f"{year:<10} {len(year_data):>10} {avg_spread:>15.4f} {win_rate:>11.1f}%")

# =============================================================================
# 3. FEATURE CORRELATIONS BY YEAR
# =============================================================================
print("\n" + "="*80)
print("3. FEATURE CORRELATIONS WITH FORWARD RETURNS BY YEAR")
print("="*80)

panel_bt['year'] = panel_bt.index.get_level_values('Date').year

momentum_features = ['Close%-21', 'Close%-63', 'Close_RSI14', 'Close_RSI21']

print(f"\n{'Year':<10}", end="")
for feat in momentum_features:
    print(f"{feat:>15}", end="")
print()
print("-" * (10 + 15 * len(momentum_features)))

for year in sorted(panel_bt['year'].unique()):
    year_data = panel_bt[panel_bt['year'] == year]
    print(f"{year:<10}", end="")
    
    for feat in momentum_features:
        if feat in year_data.columns and 'FwdRet_21' in year_data.columns:
            valid = year_data[[feat, 'FwdRet_21']].dropna()
            if len(valid) > 100:
                corr = valid[feat].corr(valid['FwdRet_21'])
                print(f"{corr:>15.4f}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        else:
            print(f"{'N/A':>15}", end="")
    print()

# =============================================================================
# 4. REVERSAL ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4. REVERSAL vs CONTINUATION")
print("="*80)

# Check if there's mean reversion
reversal_periods = simple_df[simple_df['spread'] < 0]
continuation_periods = simple_df[simple_df['spread'] > 0]

print(f"\nReversal periods (bottom > top): {len(reversal_periods)} ({len(reversal_periods)/len(simple_df)*100:.1f}%)")
print(f"Continuation periods (top > bottom): {len(continuation_periods)} ({len(continuation_periods)/len(simple_df)*100:.1f}%)")

if len(reversal_periods) > len(continuation_periods):
    print("\n❌ CRITICAL: Market is exhibiting MEAN REVERSION, not momentum")
    print("   Cross-sectional momentum strategy will LOSE MONEY in this regime")

# =============================================================================
# 5. RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("5. RECOMMENDATIONS")
print("="*80)

avg_spread = simple_df['spread'].mean()

if avg_spread < -0.0005:
    print("\n❌ MOMENTUM STRATEGY NOT VIABLE IN THIS PERIOD")
    print("\nOptions:")
    print("  1. INVERT THE STRATEGY - Short winners, long losers (mean reversion)")
    print("  2. Use different features that work in mean-reversion regimes")
    print("  3. Test on different time period (momentum may work pre-2021)")
    print("  4. Implement regime detection to switch between momentum/reversal")
    
elif avg_spread < 0.0005:
    print("\n⚠️  MOMENTUM IS TOO WEAK - Strategy has no edge")
    print("\nOptions:")
    print("  1. Increase concentration (use more extreme quantiles)")
    print("  2. Use longer lookback periods")
    print("  3. Add market timing/regime filters")
    
else:
    print("\n✓ MOMENTUM IS WORKING")
    print("  Problem likely in implementation details:")
    print("  - Check binning logic")
    print("  - Check score aggregation")
    print("  - Check portfolio construction")

print("\n" + "="*80)
