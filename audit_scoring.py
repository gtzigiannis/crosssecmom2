"""
Deep Dive: Scoring and Forward Returns Relationship
====================================================
Investigate if the scoring mechanism is inverted - do high scores predict down moves?
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("SCORING MECHANISM INVESTIGATION")
print("="*80)

# Load feature panel to analyze forward returns vs features
panel_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet")
panel = pd.read_parquet(panel_path)

print(f"\n[LOADED] Panel with {len(panel)} observations")
print(f"Date range: {panel.index.get_level_values('Date').min()} to {panel.index.get_level_values('Date').max()}")
print(f"Tickers: {panel.index.get_level_values('Ticker').nunique()}")

# Focus on the backtest period
results_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_results.csv")
results = pd.read_csv(results_path)
results['date'] = pd.to_datetime(results['date'])

backtest_start = results['date'].min()
backtest_end = results['date'].max()

print(f"\nBacktest period: {backtest_start.date()} to {backtest_end.date()}")

# =============================================================================
# 1. EXAMINE KEY MOMENTUM FEATURES
# =============================================================================
print("\n" + "="*80)
print("1. KEY MOMENTUM FEATURES vs FORWARD RETURNS")
print("="*80)

# Filter to backtest period
panel_bt = panel[
    (panel.index.get_level_values('Date') >= backtest_start) &
    (panel.index.get_level_values('Date') <= backtest_end)
].copy()

# Key momentum features
momentum_features = ['Close%-21', 'Close%-63', 'Close_Mom21', 'Close_RSI14', 'Close_RSI21']
forward_return = 'FwdRet_21'

print(f"\nAnalyzing {len(panel_bt)} observations in backtest period")

# Calculate correlations
print(f"\n{'Feature':<20} {'Correlation with FwdRet_21':>25} {'Direction':>15}")
print("-" * 65)

for feat in momentum_features:
    if feat in panel_bt.columns and forward_return in panel_bt.columns:
        # Drop NaN for correlation
        valid_data = panel_bt[[feat, forward_return]].dropna()
        if len(valid_data) > 100:
            corr = valid_data[feat].corr(valid_data[forward_return])
            direction = "✓ CORRECT" if corr > 0 else "❌ INVERTED"
            print(f"{feat:<20} {corr:>25.4f} {direction:>15}")

# =============================================================================
# 2. ANALYZE ACTUAL SELECTIONS
# =============================================================================
print("\n" + "="*80)
print("2. ANALYZING ACTUAL PORTFOLIO SELECTIONS")
print("="*80)

# Parse a few periods to see what was selected
sample_periods = results.head(5)

for idx, row in sample_periods.iterrows():
    print(f"\n--- {row['date'].date()} ---")
    
    try:
        long_tickers = eval(row['long_tickers']) if isinstance(row['long_tickers'], str) else []
        short_tickers = eval(row['short_tickers']) if isinstance(row['short_tickers'], str) else []
        
        print(f"Long tickers: {long_tickers}")
        print(f"Short tickers: {short_tickers}")
        
        # Get their forward returns from panel
        date = pd.Timestamp(row['date'])
        
        if date in panel.index.get_level_values('Date'):
            long_fwd_rets = []
            short_fwd_rets = []
            
            for ticker in long_tickers:
                if (date, ticker) in panel.index:
                    fwd_ret = panel.loc[(date, ticker), 'FwdRet_21']
                    if not pd.isna(fwd_ret):
                        long_fwd_rets.append(fwd_ret)
            
            for ticker in short_tickers:
                if (date, ticker) in panel.index:
                    fwd_ret = panel.loc[(date, ticker), 'FwdRet_21']
                    if not pd.isna(fwd_ret):
                        short_fwd_rets.append(fwd_ret)
            
            if long_fwd_rets:
                print(f"Long portfolio forward returns: {np.mean(long_fwd_rets):.4f} (avg)")
                print(f"  Individual: {[f'{x:.4f}' for x in long_fwd_rets[:5]]}")
            
            if short_fwd_rets:
                print(f"Short portfolio forward returns: {np.mean(short_fwd_rets):.4f} (avg)")
                print(f"  Individual: {[f'{x:.4f}' for x in short_fwd_rets[:5]]}")
            
            # CRITICAL CHECK: Are shorts going UP?
            if short_fwd_rets:
                shorts_up = sum(1 for x in short_fwd_rets if x > 0)
                print(f"  Shorts that went UP: {shorts_up}/{len(short_fwd_rets)} ({shorts_up/len(short_fwd_rets)*100:.1f}%)")
    except Exception as e:
        print(f"  Error parsing: {e}")

# =============================================================================
# 3. RANK CORRELATION TEST
# =============================================================================
print("\n" + "="*80)
print("3. RANK CORRELATION ACROSS ALL BACKTEST PERIODS")
print("="*80)

# For each date in backtest, check if ranking is correct
rank_correlations = []

for date in results['date'].head(10):  # Sample 10 dates
    date = pd.Timestamp(date)
    
    if date not in panel.index.get_level_values('Date'):
        continue
    
    # Get cross-section at this date
    cross_section = panel.loc[date].copy()
    
    # Calculate simple momentum score (Close%-21)
    if 'Close%-21' in cross_section.columns and 'FwdRet_21' in cross_section.columns:
        valid = cross_section[['Close%-21', 'FwdRet_21']].dropna()
        
        if len(valid) > 10:
            # Rank by momentum
            valid['momentum_rank'] = valid['Close%-21'].rank(ascending=True)
            valid['fwd_ret_rank'] = valid['FwdRet_21'].rank(ascending=True)
            
            # Spearman correlation (should be positive if correct)
            rank_corr = valid['momentum_rank'].corr(valid['fwd_ret_rank'])
            rank_correlations.append(rank_corr)
            
            # Show top/bottom 3
            top3_momentum = valid.nlargest(3, 'Close%-21')
            bottom3_momentum = valid.nsmallest(3, 'Close%-21')
            
            print(f"\n{date.date()}:")
            print(f"  Rank correlation: {rank_corr:.4f}")
            print(f"  Top 3 momentum → Avg fwd ret: {top3_momentum['FwdRet_21'].mean():.4f}")
            print(f"  Bottom 3 momentum → Avg fwd ret: {bottom3_momentum['FwdRet_21'].mean():.4f}")

if rank_correlations:
    print(f"\nOverall rank correlation (avg): {np.mean(rank_correlations):.4f}")
    print(f"Positive periods: {sum(1 for x in rank_correlations if x > 0)}/{len(rank_correlations)}")

# =============================================================================
# 4. CHECK SUPERVISED BINNING DIRECTION
# =============================================================================
print("\n" + "="*80)
print("4. CHECK FOR BINNED FEATURES")
print("="*80)

binned_cols = [c for c in panel.columns if c.endswith('_Bin')]
print(f"\nFound {len(binned_cols)} binned features in panel:")
if binned_cols:
    print(f"  {binned_cols[:10]}")
    print("\n⚠️  WARNING: Binned features in panel suggest global binning (look-ahead bias!)")
    print("   Binning should happen INSIDE walk-forward loop, not globally")
else:
    print("  ✓ No binned features found (correct - binning should be per-period)")

# =============================================================================
# 5. FINAL DIAGNOSIS
# =============================================================================
print("\n" + "="*80)
print("5. DIAGNOSIS SUMMARY")
print("="*80)

issues = []

# Check feature correlations
momentum_corrs = []
for feat in momentum_features:
    if feat in panel_bt.columns and forward_return in panel_bt.columns:
        valid_data = panel_bt[[feat, forward_return]].dropna()
        if len(valid_data) > 100:
            corr = valid_data[feat].corr(valid_data[forward_return])
            momentum_corrs.append(corr)

if momentum_corrs:
    avg_corr = np.mean(momentum_corrs)
    if avg_corr < 0:
        issues.append("❌ Momentum features have NEGATIVE correlation with forward returns")
        issues.append("   → Model is trained on OPPOSITE relationship")
    elif avg_corr < 0.05:
        issues.append("⚠️  Momentum features have WEAK correlation with forward returns")
        issues.append("   → Predictive power is very low")

# Check rank correlations
if rank_correlations and np.mean(rank_correlations) < 0:
    issues.append("❌ Ranking is INVERTED - high momentum stocks have negative forward returns")

if binned_cols:
    issues.append("❌ Global binning detected - creates look-ahead bias")

if issues:
    print("\nCRITICAL ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ No obvious scoring inversions detected")
    print("  Problem may be in:")
    print("  - Feature selection choosing wrong features")
    print("  - Supervised binning creating incorrect monotonic relationships")
    print("  - Model aggregation logic")

print("\n" + "="*80)
