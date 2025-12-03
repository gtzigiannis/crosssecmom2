"""
Manual validation experiment for feature_selection.py core functions.

This script creates synthetic panel data with known signal structure and
verifies that formation_fdr and per_window_ic_filter behave as expected.

Expected results:
- feature_0 and feature_1 should be detected (strong positive signal)
- feature_2 through feature_9 should mostly be rejected (noise)
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'd:\\REPOSITORY\\morias\\Quant\\strategies\\crosssecmom2')

from feature_selection import formation_fdr, per_window_ic_filter

print("=" * 80)
print("MANUAL VALIDATION EXPERIMENT: Feature Selection Core Functions")
print("=" * 80)

# =======================
# CREATE SYNTHETIC PANEL
# =======================
print("\n1. Creating synthetic panel data...")
print("   - 50 dates × 200 assets × 10 features")
print("   - feature_0 and feature_1: STRONG signal (coef = 0.3)")
print("   - feature_2-9: NOISE")

np.random.seed(42)
n_dates = 50
n_assets = 200
n_features = 10

dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
X_list, y_list, date_list = [], [], []

for date in dates:
    X_date = np.random.randn(n_assets, n_features)
    
    # Make feature_0 and feature_1 strongly predictive
    y_date = (0.3 * X_date[:, 0] + 
              0.3 * X_date[:, 1] + 
              np.random.randn(n_assets) * 0.5)
    
    X_list.append(X_date)
    y_list.append(y_date)
    date_list.extend([date] * n_assets)

X = np.vstack(X_list)
y = np.concatenate(y_list)
feature_names = [f'feature_{i}' for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)
X_df['date'] = date_list
X_df = X_df.set_index('date')
y_series = pd.Series(y, index=X_df.index, name='forward_return')

print(f"   ✓ Panel shape: {X_df.shape}")
print(f"   ✓ Date range: {dates[0]} to {dates[-1]}")

# =========================
# EXPERIMENT 1: FORMATION FDR
# =========================
print("\n2. Running formation_fdr on full panel...")
print("   Parameters: half_life=126, fdr_level=0.1")

approved, diagnostics = formation_fdr(
    X_df, y_series, dates,
    half_life=126,
    fdr_level=0.1,
    n_jobs=1
)

print(f"\n   Results:")
print(f"   - Approved features: {len(approved)} of {n_features}")
print(f"   - Approved list: {sorted(approved)}")

# Check if strong features are approved
strong_features = ['feature_0', 'feature_1']
strong_approved = [f for f in strong_features if f in approved]
print(f"\n   Signal Detection:")
print(f"   - Strong features approved: {len(strong_approved)}/2")
print(f"   - List: {strong_approved}")

if len(strong_approved) >= 1:
    print("   ✓ PASS: At least one strong feature detected")
else:
    print("   ✗ FAIL: No strong features detected")

# Show diagnostics for top features
print("\n   Top 5 Features by |IC|:")
diag_sorted = diagnostics.sort_values('ic_weighted', key=abs, ascending=False)
print(diag_sorted[['feature', 'ic_weighted', 't_nw', 'p_value', 'fdr_reject']].head())

# ==================================
# EXPERIMENT 2: PER-WINDOW IC FILTER
# ==================================
print("\n3. Running per_window_ic_filter on training window...")
print("   Parameters: theta_ic=0.05, t_min=1.5")

window_dates = dates[:25]  # First 25 days
X_window = X_df.loc[window_dates]
y_window = y_series.loc[window_dates]
weights = np.ones(len(window_dates))

selected, diag = per_window_ic_filter(
    X_window, y_window, window_dates, weights,
    theta_ic=0.05,
    t_min=1.5,
    n_jobs=1
)

print(f"\n   Results:")
print(f"   - Selected features: {len(selected)} of {n_features}")
print(f"   - Selected list: {sorted(selected)}")

# Check if strong features are selected
strong_selected = [f for f in strong_features if f in selected]
print(f"\n   Signal Detection:")
print(f"   - Strong features selected: {len(strong_selected)}/2")
print(f"   - List: {strong_selected}")

if len(strong_selected) >= 1:
    print("   ✓ PASS: At least one strong feature detected in window")
else:
    print("   ✗ FAIL: No strong features detected in window")

print(f"\n   Diagnostics:")
for key, val in diag.items():
    print(f"   - {key}: {val}")

# =================
# FINAL VERDICT
# =================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

formation_pass = len(strong_approved) >= 1
window_pass = len(strong_selected) >= 1

if formation_pass and window_pass:
    print("✓ BOTH TESTS PASSED")
    print("  → Core functions behave correctly on synthetic data")
    print("  → Signal detection working as expected")
    print("  → Ready to proceed with next pipeline stage")
else:
    print("✗ TESTS FAILED")
    if not formation_pass:
        print("  → formation_fdr failed to detect strong signal")
    if not window_pass:
        print("  → per_window_ic_filter failed to detect strong signal")
    print("  → Review implementation before proceeding")

print("=" * 80)
