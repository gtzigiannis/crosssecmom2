"""
Debug script to trace universe filtering at each stage.
Shows how many tickers are removed by each filter.
"""
import sys
import pandas as pd
sys.path.insert(0, '.')

from config import get_default_config

# Load config and data
config = get_default_config()
panel_df = pd.read_parquet(config.paths.panel_parquet)
universe_metadata = pd.read_csv(config.paths.universe_metadata_output)

print("=" * 80)
print("UNIVERSE FILTERING ANALYSIS")
print("=" * 80)

# Get a representative date (middle of dataset)
dates = sorted(panel_df['Date'].unique())
mid_date = dates[len(dates) // 2]
cross_section = panel_df[panel_df['Date'] == mid_date].set_index('Ticker')

print(f"\nAnalyzing date: {mid_date}")
print(f"Starting with {len(cross_section)} tickers in raw cross-section")

# Setup metadata
if 'ticker' in universe_metadata.columns:
    metadata_idx = universe_metadata.set_index('ticker')
else:
    metadata_idx = universe_metadata

# Stage 1: Core universe filter
stage1_tickers = cross_section.index
if 'in_core_after_duplicates' in metadata_idx.columns:
    core_tickers = metadata_idx[metadata_idx['in_core_after_duplicates'] == True].index
    stage1_tickers = stage1_tickers.intersection(core_tickers)

print(f"\n1. CORE UNIVERSE FILTER")
print(f"   After removing leveraged/duplicates: {len(stage1_tickers)} tickers")
print(f"   Removed: {len(cross_section) - len(stage1_tickers)}")

# Stage 2: Equity-only filter
equity_keywords = config.universe.equity_family_keywords

def is_equity_family(family_name):
    if pd.isna(family_name):
        return False
    family_str = str(family_name)
    return any(keyword.lower() in family_str.lower() for keyword in equity_keywords)

equity_mask = metadata_idx['family'].apply(is_equity_family)
equity_tickers = metadata_idx[equity_mask].index
stage2_tickers = stage1_tickers.intersection(equity_tickers)

print(f"\n2. EQUITY-ONLY FILTER")
print(f"   After equity filter: {len(stage2_tickers)} tickers")
print(f"   Removed: {len(stage1_tickers) - len(stage2_tickers)}")

# Stage 3: ADV filter
if 'ADV_63_Rank' in cross_section.columns:
    adv_threshold = config.universe.min_adv_percentile
    adv_data = cross_section.loc[stage2_tickers, 'ADV_63_Rank']
    adv_pass = adv_data[adv_data >= adv_threshold].index
    stage3_tickers = stage2_tickers.intersection(adv_pass)
    
    print(f"\n3. ADV LIQUIDITY FILTER (>= {adv_threshold:.0%} percentile)")
    print(f"   After ADV filter: {len(stage3_tickers)} tickers")
    print(f"   Removed: {len(stage2_tickers) - len(stage3_tickers)}")
    
    # Show which tickers failed ADV
    failed_adv = stage2_tickers.difference(stage3_tickers)
    if len(failed_adv) > 0:
        print(f"   Failed ADV filter: {sorted(failed_adv)[:20]}")  # Show first 20
else:
    stage3_tickers = stage2_tickers
    print(f"\n3. ADV LIQUIDITY FILTER - SKIPPED (no ADV_63_Rank column)")

# Stage 4: Data quality filter
feature_cols = [c for c in cross_section.columns 
               if c not in ['Close', 'Ticker', 'ADV_63', 'ADV_63_Rank'] 
               and not c.startswith('FwdRet')]

if len(feature_cols) > 0:
    data_quality = cross_section.loc[stage3_tickers, feature_cols].notna().mean(axis=1)
    quality_threshold = config.universe.min_data_quality
    quality_pass = data_quality[data_quality >= quality_threshold].index
    stage4_tickers = stage3_tickers.intersection(quality_pass)
    
    print(f"\n4. DATA QUALITY FILTER (>= {quality_threshold:.0%} non-null features)")
    print(f"   After quality filter: {len(stage4_tickers)} tickers")
    print(f"   Removed: {len(stage3_tickers) - len(stage4_tickers)}")
    
    # Show which tickers failed quality
    failed_quality = stage3_tickers.difference(stage4_tickers)
    if len(failed_quality) > 0:
        print(f"   Failed quality filter: {sorted(failed_quality)[:20]}")
        # Show their quality scores
        print(f"   Quality scores of failed tickers:")
        for ticker in sorted(failed_quality)[:10]:
            score = data_quality.get(ticker, 0)
            print(f"      {ticker}: {score:.2%}")
else:
    stage4_tickers = stage3_tickers
    print(f"\n4. DATA QUALITY FILTER - SKIPPED (no features)")

print(f"\n" + "=" * 80)
print(f"FINAL UNIVERSE: {len(stage4_tickers)} tickers")
print(f"Total removed: {len(cross_section) - len(stage4_tickers)}")
print(f"Tickers: {sorted(stage4_tickers)}")
print("=" * 80)

# Check across multiple dates
print("\n\nCHECKING UNIVERSE STABILITY ACROSS DATES")
print("=" * 80)

sample_dates = [dates[i] for i in [0, len(dates)//4, len(dates)//2, 3*len(dates)//4, -1]]

for date in sample_dates:
    cs = panel_df[panel_df['Date'] == date].set_index('Ticker')
    
    # Apply all filters
    tickers = cs.index
    if 'in_core_after_duplicates' in metadata_idx.columns:
        core = metadata_idx[metadata_idx['in_core_after_duplicates'] == True].index
        tickers = tickers.intersection(core)
    
    tickers = tickers.intersection(equity_tickers)
    
    if 'ADV_63_Rank' in cs.columns:
        adv_pass = cs.loc[tickers, 'ADV_63_Rank'] >= config.universe.min_adv_percentile
        tickers = tickers[adv_pass[tickers]]
    
    if len(feature_cols) > 0:
        quality = cs.loc[tickers, feature_cols].notna().mean(axis=1)
        quality_pass = quality >= config.universe.min_data_quality
        tickers = tickers[quality_pass]
    
    print(f"{date.strftime('%Y-%m-%d')}: {len(tickers)} tickers")

print("\n✓ Analysis complete")
