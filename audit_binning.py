"""
Deep Dive: Supervised Binning Analysis
=======================================
Check if supervised binning is creating incorrect monotonic relationships.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import get_default_config
from alpha_models import train_alpha_model

print("="*80)
print("SUPERVISED BINNING ANALYSIS")
print("="*80)

# Load config and panel
config = get_default_config()
panel_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet")
panel = pd.read_parquet(panel_path)

metadata_path = Path(r"D:\REPOSITORY\Data\crosssecmom2\universe_metadata.csv")
metadata = pd.read_csv(metadata_path, index_col=0)

# Pick ONE training window to analyze in detail
train_end = pd.Timestamp('2021-12-06')  # 21 days before first backtest date
train_start = train_end - pd.Timedelta(days=1260)

print(f"\nAnalyzing training window:")
print(f"  Start: {train_start.date()}")
print(f"  End: {train_end.date()}")

# Train model
print("\n[TRAINING] Fitting supervised binned model...")
model, selected_features, cv_ic_series = train_alpha_model(
    panel=panel,
    universe_metadata=metadata,
    t_train_start=train_start,
    t_train_end=train_end,
    config=config,
    model_type='supervised_binned'
)

print(f"\n[COMPLETE] Model trained")
print(f"  Selected features: {len(selected_features)}")
print(f"  Binned features: {len([f for f in model.selected_features if f.endswith('_Bin')])}")

# =============================================================================
# 1. EXAMINE FEATURE SELECTION
# =============================================================================
print("\n" + "="*80)
print("1. SELECTED FEATURES AND THEIR IC VALUES")
print("="*80)

print(f"\n{'Feature':<30} {'CV-IC':>10} {'Sign':>8}")
print("-" * 50)

for feat in selected_features[:15]:  # Top 15
    ic = cv_ic_series[feat]
    sign = "POS ✓" if ic > 0 else "NEG ⚠️"
    print(f"{feat:<30} {ic:>10.4f} {sign:>8}")

negative_ics = [ic for ic in cv_ic_series[selected_features].values if ic < 0]
if negative_ics:
    print(f"\n⚠️  {len(negative_ics)} features have NEGATIVE IC")
    print("   These will be weighted negatively (inverted contribution)")

# =============================================================================
# 2. EXAMINE BINNING DICT
# =============================================================================
print("\n" + "="*80)
print("2. BINNING BOUNDARIES")
print("="*80)

binning_dict = model.binning_dict

print(f"\nTotal binned features: {len(binning_dict)}")

# Show first 3 binned features
for i, (feat, boundaries) in enumerate(list(binning_dict.items())[:3]):
    print(f"\n{feat}:")
    print(f"  Boundaries: {boundaries}")
    print(f"  Number of bins: {len(boundaries) - 1}")

# =============================================================================
# 3. TEST SCORING ON TRAINING DATA
# =============================================================================
print("\n" + "="*80)
print("3. SCORING TEST ON TRAINING WINDOW SAMPLE")
print("="*80)

# Pick a date in the training window
test_date = pd.Timestamp('2021-11-01')

if test_date in panel.index.get_level_values('Date'):
    print(f"\nScoring at {test_date.date()}...")
    
    scores = model.score_at_date(panel, test_date, metadata, config)
    
    # Get forward returns for this date
    cross_section = panel.loc[test_date].copy()
    
    if 'FwdRet_21' in cross_section.columns:
        # Combine scores and forward returns
        comparison = pd.DataFrame({
            'score': scores,
            'fwd_ret': cross_section['FwdRet_21']
        }).dropna()
        
        print(f"  Observations with both score and fwd_ret: {len(comparison)}")
        
        # Rank correlation
        rank_corr = comparison['score'].corr(comparison['fwd_ret'], method='spearman')
        print(f"  Rank correlation (score vs fwd_ret): {rank_corr:.4f}")
        
        # Top/bottom deciles
        n_decile = max(1, len(comparison) // 10)
        top_scores = comparison.nlargest(n_decile, 'score')
        bottom_scores = comparison.nsmallest(n_decile, 'score')
        
        print(f"\n  Top scored ETFs (n={n_decile}):")
        print(f"    Avg score: {top_scores['score'].mean():.4f}")
        print(f"    Avg fwd_ret: {top_scores['fwd_ret'].mean():.4f}")
        
        print(f"\n  Bottom scored ETFs (n={n_decile}):")
        print(f"    Avg score: {bottom_scores['score'].mean():.4f}")
        print(f"    Avg fwd_ret: {bottom_scores['fwd_ret'].mean():.4f}")
        
        spread = top_scores['fwd_ret'].mean() - bottom_scores['fwd_ret'].mean()
        print(f"\n  Spread (top - bottom): {spread:.4f}")
        
        if spread < 0:
            print("    ❌ INVERTED - Top scores have LOWER forward returns!")
        else:
            print("    ✓ CORRECT - Top scores have higher forward returns")

# =============================================================================
# 4. TEST ON OUT-OF-SAMPLE DATE
# =============================================================================
print("\n" + "="*80)
print("4. SCORING TEST ON OUT-OF-SAMPLE DATE")
print("="*80)

# Use first backtest date
oos_date = pd.Timestamp('2021-12-27')

if oos_date in panel.index.get_level_values('Date'):
    print(f"\nScoring at {oos_date.date()} (out-of-sample)...")
    
    scores = model.score_at_date(panel, oos_date, metadata, config)
    
    # Get forward returns
    cross_section = panel.loc[oos_date].copy()
    
    if 'FwdRet_21' in cross_section.columns:
        comparison = pd.DataFrame({
            'score': scores,
            'fwd_ret': cross_section['FwdRet_21']
        }).dropna()
        
        print(f"  Observations: {len(comparison)}")
        
        rank_corr = comparison['score'].corr(comparison['fwd_ret'], method='spearman')
        print(f"  Rank correlation: {rank_corr:.4f}")
        
        n_decile = max(1, len(comparison) // 10)
        top_scores = comparison.nlargest(n_decile, 'score')
        bottom_scores = comparison.nsmallest(n_decile, 'score')
        
        print(f"\n  Top scored ETFs:")
        print(f"    Avg fwd_ret: {top_scores['fwd_ret'].mean():.4f}")
        print(f"    Tickers: {list(top_scores.index[:5])}")
        
        print(f"\n  Bottom scored ETFs:")
        print(f"    Avg fwd_ret: {bottom_scores['fwd_ret'].mean():.4f}")
        print(f"    Tickers: {list(bottom_scores.index[:5])}")
        
        spread = top_scores['fwd_ret'].mean() - bottom_scores['fwd_ret'].mean()
        print(f"\n  Spread: {spread:.4f}")
        
        if spread < 0:
            print("    ❌ INVERTED - Model is selecting wrong stocks!")
        else:
            print("    ✓ Model is working correctly")

# =============================================================================
# 5. DIAGNOSIS
# =============================================================================
print("\n" + "="*80)
print("5. DIAGNOSIS")
print("="*80)

issues = []

if negative_ics:
    issues.append(f"⚠️  {len(negative_ics)}/{len(selected_features)} features have negative IC")
    issues.append("   This is NORMAL if features are mean-reverting")

# Check if scoring is inverted
if 'comparison' in locals():
    if rank_corr < 0:
        issues.append("❌ SCORING IS INVERTED - scores anticorrelated with forward returns")
    elif abs(rank_corr) < 0.05:
        issues.append("⚠️  SCORING HAS NO PREDICTIVE POWER - correlation near zero")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ No obvious issues detected")

print("\n" + "="*80)
