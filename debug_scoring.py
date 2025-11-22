"""
Debug: Print actual scores and features for a single date
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import get_default_config
from alpha_models import train_alpha_model

# Load data
config = get_default_config()
panel = pd.read_parquet(r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet")
metadata = pd.read_csv(r"D:\REPOSITORY\Data\crosssecmom2\universe_metadata.csv", index_col=0)

# Train model
train_end = pd.Timestamp('2021-12-06')
train_start = train_end - pd.Timedelta(days=1260)

model, selected_features, cv_ic_series = train_alpha_model(
    panel=panel,
    universe_metadata=metadata,
    t_train_start=train_start,
    t_train_end=train_end,
    config=config,
    model_type='supervised_binned'
)

# Score at a date
test_date = pd.Timestamp('2021-12-27')
scores = model.score_at_date(panel, test_date, metadata, config)

# Get cross-section with forward returns
cross_section = panel.loc[test_date].copy()
cross_section['score'] = scores
cross_section['fwd_ret'] = cross_section['FwdRet_21']

# Apply binning to see transformed features
cross_section_binned = model._apply_binning(cross_section)

# Show top 5 and bottom 5 by score
print("="*80)
print("TOP 5 SCORED ETFS")
print("="*80)

top5 = cross_section.nlargest(5, 'score')

for ticker in top5.index:
    print(f"\n{ticker}: score={top5.loc[ticker, 'score']:.4f}, fwd_ret={top5.loc[ticker, 'fwd_ret']:.4f}")
    
    # Show key features
    print(f"  Close%-21: {cross_section.loc[ticker, 'Close%-21']:.4f}")
    print(f"  Close_RSI21: {cross_section.loc[ticker, 'Close_RSI21']:.4f}")
    if 'Close%-21_Bin' in cross_section_binned.columns:
        print(f"  Close%-21_Bin: {cross_section_binned.loc[ticker, 'Close%-21_Bin']:.4f}")

print("\n" + "="*80)
print("BOTTOM 5 SCORED ETFS")
print("="*80)

bottom5 = cross_section.nsmallest(5, 'score')

for ticker in bottom5.index:
    print(f"\n{ticker}: score={bottom5.loc[ticker, 'score']:.4f}, fwd_ret={bottom5.loc[ticker, 'fwd_ret']:.4f}")
    
    print(f"  Close%-21: {cross_section.loc[ticker, 'Close%-21']:.4f}")
    print(f"  Close_RSI21: {cross_section.loc[ticker, 'Close_RSI21']:.4f}")
    if 'Close%-21_Bin' in cross_section_binned.columns:
        print(f"  Close%-21_Bin: {cross_section_binned.loc[ticker, 'Close%-21_Bin']:.4f}")

print("\n" + "="*80)
print("FEATURE WEIGHTS")
print("="*80)

print("\nTop 5 features by |weight|:")
for feat in selected_features[:5]:
    weight = model.feature_weights[feat]
    print(f"  {feat}: {weight:.4f}")

print("\n" + "="*80)
