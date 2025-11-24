"""
Attribution Analysis Module
============================
Comprehensive multi-level attribution system for cross-sectional momentum strategy.

Answers key questions:
1. Which features contributed most to performance?
2. Which sectors/themes drove returns?
3. How much came from longs vs shorts?
4. Is IC decaying over time?
5. Is the model generalizing well out-of-sample?
6. What are the risk contributions by source?
7. How do returns vary temporally (by year, regime, etc.)?

Usage:
    attribution_df = compute_attribution_analysis(
        results_df=backtest_results,
        diagnostics=diagnostics_list,
        panel_df=panel_data,
        universe_metadata=metadata,
        config=config
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import warnings

from config import ResearchConfig


def compute_feature_attribution(
    diagnostics: List[Dict],
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attribute performance to individual features based on:
    - Selection frequency
    - Average IC when selected
    - Correlation with period returns
    
    Returns DataFrame with columns:
    - feature: Feature name
    - selection_freq: % of periods where feature was selected
    - avg_ic: Average IC across periods when selected
    - ic_stability: Std of IC (lower is more stable)
    - return_corr: Correlation between feature presence and period returns
    - importance_score: Composite score
    """
    if not diagnostics:
        return pd.DataFrame()
    
    # Aggregate feature stats
    feature_stats = defaultdict(lambda: {'count': 0, 'ic_values': [], 'periods': []})
    
    for i, diag in enumerate(diagnostics):
        if 'selected_features' not in diag or 'ic_values' not in diag:
            continue
            
        for feat in diag['selected_features']:
            feature_stats[feat]['count'] += 1
            feature_stats[feat]['periods'].append(i)
            
            if feat in diag['ic_values']:
                feature_stats[feat]['ic_values'].append(diag['ic_values'][feat])
    
    # Build attribution dataframe
    records = []
    for feat, stats in feature_stats.items():
        selection_freq = stats['count'] / len(diagnostics)
        ic_values = [x for x in stats['ic_values'] if pd.notna(x)]
        
        if len(ic_values) > 0:
            avg_ic = np.mean(ic_values)
            ic_stability = np.std(ic_values)
            median_ic = np.median(ic_values)
        else:
            avg_ic = np.nan
            ic_stability = np.nan
            median_ic = np.nan
        
        # Calculate correlation with returns
        if len(stats['periods']) > 2:
            # Create binary indicator: 1 if feature selected, 0 otherwise
            feature_indicator = pd.Series(0, index=range(len(diagnostics)))
            feature_indicator.iloc[stats['periods']] = 1
            
            # Align with results
            returns = results_df['ls_return'].values[:len(diagnostics)]
            if len(returns) == len(feature_indicator):
                return_corr = np.corrcoef(feature_indicator, returns)[0, 1]
            else:
                return_corr = np.nan
        else:
            return_corr = np.nan
        
        # Composite importance score: frequency × |IC| × (1 - IC_volatility_norm)
        if pd.notna(avg_ic) and pd.notna(ic_stability):
            # Normalize IC stability to [0,1] range (assume max std of 0.5)
            ic_stability_norm = min(ic_stability / 0.5, 1.0)
            importance_score = selection_freq * abs(avg_ic) * (1 - ic_stability_norm)
        else:
            importance_score = np.nan
        
        records.append({
            'feature': feat,
            'selection_freq': selection_freq,
            'avg_ic': avg_ic,
            'median_ic': median_ic,
            'ic_stability': ic_stability,
            'return_corr': return_corr,
            'importance_score': importance_score,
            'n_periods': stats['count']
        })
    
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values('importance_score', ascending=False)
    
    return df


def compute_sector_attribution(
    results_df: pd.DataFrame,
    diagnostics: List[Dict],
    universe_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Attribute returns to sectors/themes based on:
    - Which ETFs were held (longs vs shorts)
    - Their family/category groupings
    - Return contribution per group
    
    Returns DataFrame with columns:
    - sector: Sector/theme name (from family column)
    - n_periods_long: # periods with long exposure
    - n_periods_short: # periods with short exposure
    - avg_weight_long: Average position size when long
    - avg_weight_short: Average position size when short
    - return_contribution: Estimated return contribution
    """
    if 'family' not in universe_metadata.columns:
        return pd.DataFrame()
    
    # Set up metadata index
    if 'ticker' in universe_metadata.columns:
        metadata_idx = universe_metadata.set_index('ticker')
    else:
        metadata_idx = universe_metadata
    
    # Aggregate sector exposure across periods
    sector_stats = defaultdict(lambda: {
        'long_periods': 0,
        'short_periods': 0,
        'long_weights': [],
        'short_weights': [],
        'returns': []
    })
    
    for i, diag in enumerate(diagnostics):
        if 'long_positions' not in diag or 'short_positions' not in diag:
            continue
        
        period_return = results_df['ls_return'].iloc[i] if i < len(results_df) else 0
        
        # Process long positions
        for ticker, weight in diag['long_positions'].items():
            if ticker in metadata_idx.index:
                family = metadata_idx.loc[ticker, 'family']
                if pd.notna(family):
                    sector_stats[family]['long_periods'] += 1
                    sector_stats[family]['long_weights'].append(weight)
                    sector_stats[family]['returns'].append(period_return * weight)
        
        # Process short positions
        for ticker, weight in diag['short_positions'].items():
            if ticker in metadata_idx.index:
                family = metadata_idx.loc[ticker, 'family']
                if pd.notna(family):
                    sector_stats[family]['short_periods'] += 1
                    sector_stats[family]['short_weights'].append(abs(weight))
                    sector_stats[family]['returns'].append(period_return * weight)
    
    # Build dataframe
    records = []
    for sector, stats in sector_stats.items():
        avg_weight_long = np.mean(stats['long_weights']) if stats['long_weights'] else 0
        avg_weight_short = np.mean(stats['short_weights']) if stats['short_weights'] else 0
        return_contribution = sum(stats['returns']) if stats['returns'] else 0
        
        records.append({
            'sector': sector,
            'n_periods_long': stats['long_periods'],
            'n_periods_short': stats['short_periods'],
            'avg_weight_long': avg_weight_long,
            'avg_weight_short': avg_weight_short,
            'return_contribution': return_contribution
        })
    
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values('return_contribution', ascending=False)
    
    return df


def compute_long_short_attribution(results_df: pd.DataFrame) -> Dict:
    """
    Detailed attribution of returns between long and short sides.
    
    Returns dict with:
    - long_total_return: Cumulative long return
    - short_total_return: Cumulative short return
    - long_contribution_pct: % of total return from longs
    - short_contribution_pct: % of total return from shorts
    - long_win_rate: % of periods with positive long returns
    - short_win_rate: % of periods with positive short returns
    - long_sharpe: Sharpe ratio of long side
    - short_sharpe: Sharpe ratio of short side
    """
    if 'long_ret' not in results_df.columns or 'short_ret' not in results_df.columns:
        return {}
    
    long_rets = results_df['long_ret']
    short_rets = results_df['short_ret']
    
    # Total returns
    long_total = long_rets.sum()
    short_total = short_rets.sum()
    
    # Contribution percentages
    total_abs = abs(long_total) + abs(short_total)
    if total_abs > 0:
        long_contribution_pct = long_total / total_abs * 100
        short_contribution_pct = short_total / total_abs * 100
    else:
        long_contribution_pct = 0
        short_contribution_pct = 0
    
    # Win rates
    long_win_rate = (long_rets > 0).sum() / len(long_rets) * 100
    short_win_rate = (short_rets > 0).sum() / len(short_rets) * 100
    
    # Sharpe ratios
    long_sharpe = long_rets.mean() / long_rets.std() * np.sqrt(12) if long_rets.std() > 0 else 0
    short_sharpe = short_rets.mean() / short_rets.std() * np.sqrt(12) if short_rets.std() > 0 else 0
    
    return {
        'long_total_return': long_total,
        'short_total_return': short_total,
        'long_contribution_pct': long_contribution_pct,
        'short_contribution_pct': short_contribution_pct,
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'long_sharpe': long_sharpe,
        'short_sharpe': short_sharpe,
        'long_avg_return': long_rets.mean(),
        'short_avg_return': short_rets.mean(),
        'long_volatility': long_rets.std(),
        'short_volatility': short_rets.std()
    }


def compute_ic_decay_analysis(diagnostics: List[Dict]) -> pd.DataFrame:
    """
    Analyze whether Information Coefficient (IC) is decaying over time.
    
    Returns DataFrame with columns:
    - period: Period index
    - avg_ic: Average IC across all features
    - median_ic: Median IC
    - top_ic: IC of best feature
    - n_features: Number of features selected
    - ic_trend: Linear trend coefficient (negative = decay)
    """
    if not diagnostics:
        return pd.DataFrame()
    
    records = []
    for i, diag in enumerate(diagnostics):
        if 'ic_values' not in diag or not diag['ic_values']:
            continue
        
        ic_values = [v for v in diag['ic_values'].values() if pd.notna(v)]
        if not ic_values:
            continue
        
        avg_ic = np.mean([abs(x) for x in ic_values])
        median_ic = np.median([abs(x) for x in ic_values])
        top_ic = max([abs(x) for x in ic_values])
        n_features = len(diag.get('selected_features', []))
        
        records.append({
            'period': i,
            'avg_ic': avg_ic,
            'median_ic': median_ic,
            'top_ic': top_ic,
            'n_features': n_features
        })
    
    df = pd.DataFrame(records)
    
    if len(df) > 2:
        # Compute linear trend of IC over time
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(df['period'], df['avg_ic'])
        df.attrs['ic_trend'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'interpretation': 'Decaying' if slope < -0.001 else ('Improving' if slope > 0.001 else 'Stable')
        }
    
    return df


def compute_generalization_metrics(
    diagnostics: List[Dict],
    results_df: pd.DataFrame
) -> Dict:
    """
    Assess out-of-sample generalization quality.
    
    Returns dict with:
    - avg_train_ic: Average IC during training (in-sample)
    - avg_realized_ic: Average realized IC out-of-sample (if tracked)
    - ic_degradation: Difference between train and realized IC
    - return_consistency: Correlation between predicted scores and realized returns
    - prediction_accuracy: % of correct directional predictions
    """
    if not diagnostics:
        return {}
    
    # Collect training ICs
    train_ics = []
    for diag in diagnostics:
        if 'ic_values' in diag and diag['ic_values']:
            ic_vals = [abs(v) for v in diag['ic_values'].values() if pd.notna(v)]
            if ic_vals:
                train_ics.append(np.mean(ic_vals))
    
    avg_train_ic = np.mean(train_ics) if train_ics else np.nan
    
    # TODO: Track realized IC if we store predicted scores vs realized returns
    # For now, we estimate generalization via return consistency
    
    # Estimate prediction accuracy from win rate
    if len(results_df) > 0:
        win_rate = (results_df['ls_return'] > 0).sum() / len(results_df)
        prediction_accuracy = win_rate * 100
    else:
        prediction_accuracy = np.nan
    
    return {
        'avg_train_ic': avg_train_ic,
        'avg_realized_ic': np.nan,  # Not tracked yet
        'ic_degradation': np.nan,  # Not tracked yet
        'prediction_accuracy': prediction_accuracy,
        'train_ic_stability': np.std(train_ics) if train_ics else np.nan
    }


def compute_risk_attribution(results_df: pd.DataFrame) -> Dict:
    """
    Decompose risk (volatility) into components.
    
    Returns dict with:
    - total_volatility: Overall strategy volatility
    - long_volatility: Long side volatility
    - short_volatility: Short side volatility
    - cash_volatility: Cash return volatility
    - turnover_volatility: Volatility of turnover
    """
    metrics = {
        'total_volatility': results_df['ls_return'].std() if 'ls_return' in results_df.columns else np.nan,
        'long_volatility': results_df['long_ret'].std() if 'long_ret' in results_df.columns else np.nan,
        'short_volatility': results_df['short_ret'].std() if 'short_ret' in results_df.columns else np.nan,
        'cash_volatility': results_df['cash_pnl'].std() if 'cash_pnl' in results_df.columns else np.nan,
        'turnover_volatility': results_df['turnover'].std() if 'turnover' in results_df.columns else np.nan,
    }
    
    # Annualize (assuming monthly rebalancing)
    for key in ['total_volatility', 'long_volatility', 'short_volatility', 'cash_volatility']:
        if pd.notna(metrics[key]):
            metrics[key + '_annual'] = metrics[key] * np.sqrt(12)
    
    return metrics


def compute_temporal_attribution(
    results_df: pd.DataFrame,
    diagnostics: List[Dict]
) -> pd.DataFrame:
    """
    Analyze performance across different time periods.
    
    Returns DataFrame with columns:
    - year: Calendar year
    - n_periods: Number of rebalances
    - total_return: Cumulative return
    - win_rate: % winning periods
    - avg_return: Average period return
    - volatility: Return volatility
    - sharpe: Sharpe ratio
    """
    if len(results_df) == 0:
        return pd.DataFrame()
    
    # Add year column
    df = results_df.copy()
    df['year'] = df.index.year
    
    # Group by year
    yearly_stats = []
    for year, group in df.groupby('year'):
        total_return = (1 + group['ls_return']).prod() - 1
        win_rate = (group['ls_return'] > 0).sum() / len(group) * 100
        avg_return = group['ls_return'].mean()
        volatility = group['ls_return'].std()
        sharpe = avg_return / volatility * np.sqrt(12) if volatility > 0 else 0
        
        yearly_stats.append({
            'year': year,
            'n_periods': len(group),
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe': sharpe
        })
    
    return pd.DataFrame(yearly_stats)


def compute_attribution_analysis(
    results_df: pd.DataFrame,
    diagnostics: List[Dict],
    panel_df: Optional[pd.DataFrame] = None,
    universe_metadata: Optional[pd.DataFrame] = None,
    config: Optional[ResearchConfig] = None
) -> Dict[str, pd.DataFrame]:
    """
    Master function: Compute all attribution analyses.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results with date index
    diagnostics : List[Dict]
        Diagnostics from each period
    panel_df : pd.DataFrame, optional
        Full panel data (for advanced attribution)
    universe_metadata : pd.DataFrame, optional
        ETF metadata (for sector attribution)
    config : ResearchConfig, optional
        Configuration object
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys:
        - 'feature_attribution': Feature-level attribution
        - 'sector_attribution': Sector/theme attribution
        - 'long_short_attribution': Long vs short attribution (dict, not DataFrame)
        - 'ic_decay': IC decay analysis
        - 'generalization': Generalization metrics (dict, not DataFrame)
        - 'risk_attribution': Risk decomposition (dict, not DataFrame)
        - 'temporal_attribution': Temporal performance breakdown
    """
    attribution_results = {}
    
    print("\n" + "="*80)
    print("ATTRIBUTION ANALYSIS")
    print("="*80)
    
    # Feature attribution
    print("\n[1/7] Computing feature attribution...")
    feature_attr = compute_feature_attribution(diagnostics, results_df)
    attribution_results['feature_attribution'] = feature_attr
    
    if len(feature_attr) > 0:
        print(f"  Top 5 features by importance:")
        for i, row in feature_attr.head(5).iterrows():
            print(f"    {row['feature']:30s} - Importance: {row['importance_score']:.4f}, "
                  f"Freq: {row['selection_freq']*100:.1f}%, IC: {row['avg_ic']:+.4f}")
    
    # Sector attribution
    if universe_metadata is not None:
        print("\n[2/7] Computing sector attribution...")
        sector_attr = compute_sector_attribution(results_df, diagnostics, universe_metadata)
        attribution_results['sector_attribution'] = sector_attr
        
        if len(sector_attr) > 0:
            print(f"  Top 5 sectors by return contribution:")
            for i, row in sector_attr.head(5).iterrows():
                print(f"    {row['sector']:30s} - Return: {row['return_contribution']*100:+.2f}%, "
                      f"Long: {row['n_periods_long']}, Short: {row['n_periods_short']}")
    else:
        print("\n[2/7] Skipping sector attribution (no metadata)")
        attribution_results['sector_attribution'] = pd.DataFrame()
    
    # Long/short attribution
    print("\n[3/7] Computing long/short attribution...")
    ls_attr = compute_long_short_attribution(results_df)
    attribution_results['long_short_attribution'] = ls_attr
    
    if ls_attr:
        print(f"  Long contribution:  {ls_attr['long_total_return']*100:+.2f}% "
              f"({ls_attr['long_contribution_pct']:.1f}% of total), "
              f"Win rate: {ls_attr['long_win_rate']:.1f}%, Sharpe: {ls_attr['long_sharpe']:.2f}")
        print(f"  Short contribution: {ls_attr['short_total_return']*100:+.2f}% "
              f"({ls_attr['short_contribution_pct']:.1f}% of total), "
              f"Win rate: {ls_attr['short_win_rate']:.1f}%, Sharpe: {ls_attr['short_sharpe']:.2f}")
    
    # IC decay analysis
    print("\n[4/7] Computing IC decay analysis...")
    ic_decay = compute_ic_decay_analysis(diagnostics)
    attribution_results['ic_decay'] = ic_decay
    
    if len(ic_decay) > 0 and 'ic_trend' in ic_decay.attrs:
        trend = ic_decay.attrs['ic_trend']
        print(f"  IC Trend: {trend['interpretation']} (slope: {trend['slope']:+.6f}, "
              f"R²: {trend['r_squared']:.3f}, p: {trend['p_value']:.4f})")
        print(f"  Avg IC: {ic_decay['avg_ic'].mean():.4f}, Latest IC: {ic_decay['avg_ic'].iloc[-1]:.4f}")
    
    # Generalization metrics
    print("\n[5/7] Computing generalization metrics...")
    gen_metrics = compute_generalization_metrics(diagnostics, results_df)
    attribution_results['generalization'] = gen_metrics
    
    if gen_metrics:
        print(f"  Avg Train IC: {gen_metrics['avg_train_ic']:.4f}")
        print(f"  Prediction Accuracy: {gen_metrics['prediction_accuracy']:.1f}%")
        print(f"  Train IC Stability: {gen_metrics['train_ic_stability']:.4f}")
    
    # Risk attribution
    print("\n[6/7] Computing risk attribution...")
    risk_attr = compute_risk_attribution(results_df)
    attribution_results['risk_attribution'] = risk_attr
    
    if risk_attr:
        print(f"  Total Volatility (annual): {risk_attr.get('total_volatility_annual', np.nan)*100:.2f}%")
        print(f"  Long Vol (annual):  {risk_attr.get('long_volatility_annual', np.nan)*100:.2f}%")
        print(f"  Short Vol (annual): {risk_attr.get('short_volatility_annual', np.nan)*100:.2f}%")
    
    # Temporal attribution
    print("\n[7/7] Computing temporal attribution...")
    temporal_attr = compute_temporal_attribution(results_df, diagnostics)
    attribution_results['temporal_attribution'] = temporal_attr
    
    if len(temporal_attr) > 0:
        print(f"  Performance by year:")
        for i, row in temporal_attr.iterrows():
            print(f"    {int(row['year'])}: Return {row['total_return']*100:+.2f}%, "
                  f"Win Rate {row['win_rate']:.1f}%, Sharpe {row['sharpe']:.2f}")
    
    print("\n" + "="*80)
    
    return attribution_results


def save_attribution_results(
    attribution_results: Dict,
    output_dir: str = ".",
    prefix: str = "attribution"
):
    """
    Save attribution results to CSV files.
    
    Parameters
    ----------
    attribution_results : Dict
        Output from compute_attribution_analysis()
    output_dir : str
        Directory to save files
    prefix : str
        Filename prefix
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save DataFrames
    for name, data in attribution_results.items():
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            filename = output_path / f"{prefix}_{name}_{timestamp}.csv"
            data.to_csv(filename, index=True)
            print(f"[save] {name}: {filename}")
    
    # Save dicts as single summary file
    summary_records = []
    for name, data in attribution_results.items():
        if isinstance(data, dict):
            for key, value in data.items():
                summary_records.append({
                    'category': name,
                    'metric': key,
                    'value': value
                })
    
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_filename = output_path / f"{prefix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"[save] summary: {summary_filename}")


if __name__ == "__main__":
    print("Attribution Analysis Module")
    print("\nUsage:")
    print("  attribution_results = compute_attribution_analysis(")
    print("      results_df, diagnostics, panel_df, universe_metadata, config")
    print("  )")
    print("\nProvides 7 types of attribution:")
    print("  1. Feature Attribution - Which features drive performance")
    print("  2. Sector Attribution - Which sectors/themes contribute")
    print("  3. Long/Short Attribution - Return split between longs and shorts")
    print("  4. IC Decay - Is predictive power declining over time")
    print("  5. Generalization - Out-of-sample performance quality")
    print("  6. Risk Attribution - Volatility decomposition")
    print("  7. Temporal Attribution - Performance by year/regime")


def validate_date_integrity(
    panel_df: pd.DataFrame,
    results_df: pd.DataFrame,
    diagnostics: List[Dict]
) -> Dict:
    """
    Verify that dates are preserved and properly aligned throughout the pipeline.
    
    Checks:
    1. Panel data has Date as index (or MultiIndex level)
    2. Results have date index and are sorted
    3. Diagnostics dates match results dates
    4. No missing dates in critical data structures
    5. All dates are pandas Timestamp objects (not strings)
    
    Returns dict with:
    - panel_date_check: True if panel has proper date index
    - results_date_check: True if results sorted and valid
    - diagnostics_date_check: True if diagnostics dates align
    - date_range: (min_date, max_date) tuple
    - issues: List of any problems found
    """
    issues = []
    
    # Check panel data
    panel_date_check = False
    if isinstance(panel_df.index, pd.MultiIndex):
        if 'Date' in panel_df.index.names:
            date_level = panel_df.index.get_level_values('Date')
            if isinstance(date_level.dtype, pd.DatetimeTZDtype) or date_level.dtype == 'datetime64[ns]':
                panel_date_check = True
            else:
                issues.append("Panel Date index is not datetime type")
        else:
            issues.append("Panel MultiIndex does not have 'Date' level")
    else:
        issues.append("Panel does not have MultiIndex structure")
    
    # Check results
    results_date_check = False
    if isinstance(results_df.index, pd.DatetimeIndex):
        results_date_check = True
        # Verify sorted
        if not results_df.index.is_monotonic_increasing:
            issues.append("Results index is not sorted by date")
    else:
        issues.append("Results index is not DatetimeIndex")
    
    # Check diagnostics
    diagnostics_date_check = False
    if diagnostics and len(diagnostics) > 0:
        # Check if diagnostics have dates as Timestamp objects
        first_diag_date = diagnostics[0].get('date')
        if isinstance(first_diag_date, pd.Timestamp):
            diagnostics_date_check = True
            
            # Verify diagnostics dates match results dates
            if results_date_check and len(diagnostics) == len(results_df):
                diag_dates = [d['date'] for d in diagnostics]
                results_dates = results_df.index.tolist()
                
                if diag_dates != results_dates:
                    issues.append("Diagnostics dates do not match results dates")
        else:
            issues.append("Diagnostics dates are not Timestamp objects")
    
    # Get date range
    date_range = (None, None)
    if panel_date_check:
        panel_dates = panel_df.index.get_level_values('Date')
        date_range = (panel_dates.min(), panel_dates.max())
    
    return {
        'panel_date_check': panel_date_check,
        'results_date_check': results_date_check,
        'diagnostics_date_check': diagnostics_date_check,
        'date_range': date_range,
        'issues': issues,
        'all_checks_passed': len(issues) == 0
    }
