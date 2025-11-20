"""
Panel Data Utilities for Cross-Sectional Research
==================================================
Demonstrates powerful operations enabled by the panel structure that were
difficult/impossible with wide format data.

Operations include:
- Universe evolution tracking
- Cross-sectional correlation analysis
- Feature importance via IC (Information Coefficient)
- Regime-dependent signal analysis
- Time-series of cross-sectional statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# ============================================================================
# UNIVERSE EVOLUTION ANALYSIS
# ============================================================================

def analyze_universe_evolution(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Track how the universe evolves over time.
    
    Returns
    -------
    pd.DataFrame
        Time series of universe statistics
    """
    print("[universe] Analyzing universe evolution...")
    
    # Group by date
    daily_stats = []
    
    for date in panel_df.index.get_level_values('Date').unique():
        cross_section = panel_df.loc[date]
        
        # Count tickers with sufficient data
        feature_cols = [c for c in cross_section.columns 
                       if c not in ['Close', 'Ticker'] and not c.startswith('FwdRet')]
        data_quality = cross_section[feature_cols].notna().mean(axis=1)
        
        high_quality_count = (data_quality >= 0.8).sum()
        medium_quality_count = ((data_quality >= 0.5) & (data_quality < 0.8)).sum()
        low_quality_count = (data_quality < 0.5).sum()
        
        daily_stats.append({
            'Date': date,
            'Total_Tickers': len(cross_section),
            'High_Quality': high_quality_count,
            'Medium_Quality': medium_quality_count,
            'Low_Quality': low_quality_count,
            'Avg_Data_Quality': data_quality.mean()
        })
    
    stats_df = pd.DataFrame(daily_stats).set_index('Date')
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Stacked area of universe composition
    axes[0].fill_between(stats_df.index, 0, stats_df['High_Quality'], 
                         alpha=0.7, label='High Quality (>80%)', color='green')
    axes[0].fill_between(stats_df.index, stats_df['High_Quality'], 
                         stats_df['High_Quality'] + stats_df['Medium_Quality'],
                         alpha=0.7, label='Medium Quality (50-80%)', color='orange')
    axes[0].fill_between(stats_df.index, 
                         stats_df['High_Quality'] + stats_df['Medium_Quality'],
                         stats_df['Total_Tickers'],
                         alpha=0.7, label='Low Quality (<50%)', color='red')
    axes[0].set_ylabel('Number of Tickers')
    axes[0].set_title('Universe Evolution: Data Quality Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Average data quality
    axes[1].plot(stats_df.index, stats_df['Avg_Data_Quality'], linewidth=2)
    axes[1].set_ylabel('Average Data Quality')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Average Data Quality Score (Fraction of Non-NaN Features)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'C:\REPOSITORY\Models\universe_evolution.png', dpi=150)
    print("[universe] Plot saved to: C:\\REPOSITORY\\Models\\universe_evolution.png")
    plt.close()
    
    return stats_df

# ============================================================================
# INFORMATION COEFFICIENT ANALYSIS
# ============================================================================

def compute_ic_time_series(panel_df: pd.DataFrame,
                          features: list = None,
                          target: str = 'FwdRet_21',
                          method: str = 'spearman') -> pd.DataFrame:
    """
    Compute Information Coefficient (IC) over time.
    
    IC measures the rank correlation between feature values and forward returns
    at each point in time. High IC indicates predictive power.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    features : list, optional
        Features to analyze (if None, use momentum features)
    target : str
        Forward return column
    method : str
        'spearman' or 'pearson'
        
    Returns
    -------
    pd.DataFrame
        Time series of IC values for each feature
    """
    print(f"[ic] Computing {method} IC time series...")
    
    if features is None:
        # Default to key momentum features
        features = [c for c in panel_df.columns if any(
            x in c for x in ['Close%-', 'Mom', 'RSI', 'MACD']
        ) and '_Rank' not in c and '_ZScore' not in c]
        features = features[:20]  # Limit to avoid overload
    
    ic_data = []
    
    dates = panel_df.index.get_level_values('Date').unique()
    for i, date in enumerate(dates):
        if i % 100 == 0:
            print(f"[ic] Progress: {i}/{len(dates)}")
        
        cross_section = panel_df.loc[date]
        
        # Skip if insufficient data
        if len(cross_section) < 20 or cross_section[target].isna().all():
            continue
        
        ic_row = {'Date': date}
        
        for feat in features:
            if feat not in cross_section.columns:
                continue
            
            # Get valid pairs
            valid_mask = cross_section[[feat, target]].notna().all(axis=1)
            if valid_mask.sum() < 10:
                ic_row[feat] = np.nan
                continue
            
            feat_values = cross_section.loc[valid_mask, feat]
            target_values = cross_section.loc[valid_mask, target]
            
            # Compute correlation
            if method == 'spearman':
                corr, _ = spearmanr(feat_values, target_values)
            else:
                corr, _ = pearsonr(feat_values, target_values)
            
            ic_row[feat] = corr
        
        ic_data.append(ic_row)
    
    ic_df = pd.DataFrame(ic_data).set_index('Date')
    
    # Summary statistics
    print("\n[ic] IC Summary Statistics:")
    print("="*60)
    ic_summary = pd.DataFrame({
        'Mean_IC': ic_df.mean(),
        'Std_IC': ic_df.std(),
        'IC_IR': ic_df.mean() / ic_df.std(),  # IC Information Ratio
        'Hit_Rate': (ic_df > 0).mean(),
        'Abs_Mean_IC': ic_df.abs().mean()
    }).sort_values('IC_IR', ascending=False)
    
    print(ic_summary.head(10).to_string())
    
    # Save
    ic_df.to_csv(r'C:\REPOSITORY\Models\ic_time_series.csv')
    print("\n[ic] IC time series saved to: C:\\REPOSITORY\\Models\\ic_time_series.csv")
    
    return ic_df

def plot_ic_analysis(ic_df: pd.DataFrame, top_n: int = 5):
    """Plot IC analysis for top features."""
    print(f"[ic] Plotting IC for top {top_n} features...")
    
    # Select top features by IC IR
    ic_ir = ic_df.mean() / ic_df.std()
    top_features = ic_ir.nlargest(top_n).index.tolist()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # IC time series
    for feat in top_features:
        axes[0].plot(ic_df.index, ic_df[feat], alpha=0.7, label=feat[:20])
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('IC')
    axes[0].set_title('Information Coefficient Time Series (Top Features)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # IC distribution
    for feat in top_features:
        axes[1].hist(ic_df[feat].dropna(), bins=30, alpha=0.5, label=feat[:20])
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('IC')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('IC Distribution')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Rolling IC mean (12-period)
    for feat in top_features:
        rolling_ic = ic_df[feat].rolling(12).mean()
        axes[2].plot(rolling_ic.index, rolling_ic, alpha=0.7, label=feat[:20])
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Rolling Mean IC')
    axes[2].set_title('Rolling 12-Period IC')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'C:\REPOSITORY\Models\ic_analysis.png', dpi=150)
    print("[ic] Plot saved to: C:\\REPOSITORY\\Models\\ic_analysis.png")
    plt.close()

# ============================================================================
# CROSS-SECTIONAL CORRELATION ANALYSIS
# ============================================================================

def analyze_cross_sectional_correlation(panel_df: pd.DataFrame,
                                       date: str = None,
                                       feature_categories: dict = None) -> pd.DataFrame:
    """
    Analyze correlation structure across the cross-section at a point in time.
    
    This reveals which features move together across assets, helping identify
    redundant features or factor structures.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    date : str, optional
        Specific date to analyze (if None, uses most recent)
    feature_categories : dict, optional
        {category: [features]} for grouping
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    print("[cs_corr] Analyzing cross-sectional feature correlation...")
    
    # Select date
    if date is None:
        date = panel_df.index.get_level_values('Date').max()
    
    cross_section = panel_df.loc[date]
    
    # Select features (exclude CS transforms to avoid multicollinearity)
    feature_cols = [c for c in cross_section.columns 
                   if c not in ['Close', 'Ticker'] 
                   and not c.startswith('FwdRet')
                   and '_Rank' not in c 
                   and '_ZScore' not in c
                   and '_Quantile' not in c
                   and '_Bin' not in c]
    
    # Limit to key features
    momentum_feats = [c for c in feature_cols if 'Close%-' in c or 'Mom' in c][:10]
    trend_feats = [c for c in feature_cols if '_MA' in c or '_EMA' in c][:5]
    vol_feats = [c for c in feature_cols if '_std' in c or '_ATR' in c][:5]
    osc_feats = [c for c in feature_cols if '_RSI' in c or '_MACD' in c][:5]
    
    selected_features = momentum_feats + trend_feats + vol_feats + osc_feats
    
    # Compute correlation
    corr_matrix = cross_section[selected_features].corr(method='spearman')
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True)
    plt.title(f'Cross-Sectional Feature Correlation\nDate: {date.date()}')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(r'C:\REPOSITORY\Models\cs_correlation_heatmap.png', dpi=150)
    print("[cs_corr] Heatmap saved to: C:\\REPOSITORY\\Models\\cs_correlation_heatmap.png")
    plt.close()
    
    # Find highly correlated pairs (potential redundancy)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        print("\n[cs_corr] Highly Correlated Feature Pairs (|corr| > 0.8):")
        print("="*80)
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                                  key=abs, 
                                                                  ascending=False)
        print(high_corr_df.head(20).to_string())
    
    return corr_matrix

# ============================================================================
# REGIME-DEPENDENT ANALYSIS
# ============================================================================

def analyze_regime_performance(panel_df: pd.DataFrame,
                              results_df: pd.DataFrame,
                              regime_feature: str = 'Close%-252') -> pd.DataFrame:
    """
    Analyze strategy performance across different market regimes.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    results_df : pd.DataFrame
        Backtest results from walk_forward_backtest()
    regime_feature : str
        Feature to define regimes (e.g., trailing 1-year market return)
        
    Returns
    -------
    pd.DataFrame
        Performance by regime
    """
    print("[regime] Analyzing regime-dependent performance...")
    
    # Define market regime based on SPY or broad market proxy
    # Assuming 'SPY' is in the universe
    spy_data = panel_df.xs('SPY', level='Ticker', drop_level=True)
    
    if regime_feature not in spy_data.columns:
        print(f"[regime] Feature {regime_feature} not found, using Close%-63 instead")
        regime_feature = 'Close%-63'
    
    # Map dates to regimes
    regime_map = {}
    for date in results_df.index:
        try:
            market_ret = spy_data.loc[date, regime_feature]
            if market_ret > 10:
                regime = 'Bull'
            elif market_ret < -10:
                regime = 'Bear'
            else:
                regime = 'Neutral'
            regime_map[date] = regime
        except:
            regime_map[date] = 'Unknown'
    
    results_df['Regime'] = results_df.index.map(regime_map)
    
    # Compute statistics by regime
    regime_stats = results_df.groupby('Regime')['ls_return'].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Sharpe', lambda x: x.mean() / x.std() if x.std() > 0 else 0),
        ('Win_Rate', lambda x: (x > 0).mean()),
        ('Max', 'max'),
        ('Min', 'min')
    ]).round(2)
    
    print("\n[regime] Performance by Market Regime:")
    print("="*60)
    print(regime_stats.to_string())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    results_df.boxplot(column='ls_return', by='Regime', ax=axes[0])
    axes[0].set_title('Return Distribution by Regime')
    axes[0].set_xlabel('Market Regime')
    axes[0].set_ylabel('L/S Return (%)')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # Cumulative returns by regime
    for regime in results_df['Regime'].unique():
        if regime == 'Unknown':
            continue
        regime_rets = results_df[results_df['Regime'] == regime]['ls_return']
        cum_ret = (1 + regime_rets / 100).cumprod()
        axes[1].plot(cum_ret.index, cum_ret.values, label=regime, linewidth=2)
    
    axes[1].set_title('Cumulative Returns by Regime')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Return')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'C:\REPOSITORY\Models\regime_analysis.png', dpi=150)
    print("[regime] Plot saved to: C:\\REPOSITORY\\Models\\regime_analysis.png")
    plt.close()
    
    return regime_stats

# ============================================================================
# TIME-SERIES OF CROSS-SECTIONAL STATISTICS
# ============================================================================

def track_cross_sectional_dispersion(panel_df: pd.DataFrame,
                                    features: list = None) -> pd.DataFrame:
    """
    Track the dispersion (cross-sectional volatility) of features over time.
    
    High dispersion indicates more differentiation between assets, which
    may create better opportunities for cross-sectional strategies.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    features : list, optional
        Features to track
        
    Returns
    -------
    pd.DataFrame
        Time series of cross-sectional std dev for each feature
    """
    print("[dispersion] Tracking cross-sectional dispersion...")
    
    if features is None:
        features = [c for c in panel_df.columns if 'Close%-63' in c or 'Mom63' in c][:5]
    
    dispersion_data = []
    
    dates = panel_df.index.get_level_values('Date').unique()
    for date in dates:
        cross_section = panel_df.loc[date]
        
        disp_row = {'Date': date}
        for feat in features:
            if feat in cross_section.columns:
                disp_row[feat] = cross_section[feat].std()
        
        dispersion_data.append(disp_row)
    
    disp_df = pd.DataFrame(dispersion_data).set_index('Date')
    
    # Plot
    plt.figure(figsize=(12, 6))
    for feat in features:
        if feat in disp_df.columns:
            plt.plot(disp_df.index, disp_df[feat], alpha=0.7, label=feat[:30], linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Cross-Sectional Std Dev')
    plt.title('Cross-Sectional Dispersion Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'C:\REPOSITORY\Models\cs_dispersion.png', dpi=150)
    print("[dispersion] Plot saved to: C:\\REPOSITORY\\Models\\cs_dispersion.png")
    plt.close()
    
    return disp_df

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_all_analyses(panel_path: str, results_path: str = None):
    """
    Run all utility analyses.
    
    Parameters
    ----------
    panel_path : str
        Path to panel feature data
    results_path : str, optional
        Path to backtest results (for regime analysis)
    """
    print("="*80)
    print("PANEL DATA UTILITY ANALYSES")
    print("="*80)
    
    # Load panel data
    print("\n[load] Loading panel data...")
    panel_df = pd.read_parquet(panel_path)
    print(f"[load] Loaded: {panel_df.shape}")
    
    # 1. Universe evolution
    print("\n" + "="*80)
    universe_stats = analyze_universe_evolution(panel_df)
    
    # 2. IC analysis
    print("\n" + "="*80)
    ic_df = compute_ic_time_series(panel_df, target='FwdRet_21')
    plot_ic_analysis(ic_df)
    
    # 3. Cross-sectional correlation
    print("\n" + "="*80)
    corr_matrix = analyze_cross_sectional_correlation(panel_df)
    
    # 4. Dispersion tracking
    print("\n" + "="*80)
    disp_df = track_cross_sectional_dispersion(panel_df)
    
    # 5. Regime analysis (if results available)
    if results_path and pd.io.common.file_exists(results_path):
        print("\n" + "="*80)
        results_df = pd.read_csv(results_path, index_col=0, parse_dates=True)
        regime_stats = analyze_regime_performance(panel_df, results_df)
    
    print("\n[done] All analyses complete!")

if __name__ == "__main__":
    # Run with your data
    PANEL_PATH = r'C:\REPOSITORY\Models\cs_momentum_features.parquet'
    RESULTS_PATH = r'C:\REPOSITORY\Models\cs_momentum_results.csv'
    
    run_all_analyses(PANEL_PATH, RESULTS_PATH)
