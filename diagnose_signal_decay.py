"""
Diagnostic script to understand why feature selection is collapsing to 0-1 features.

Key questions:
1. What is the IC distribution in Formation vs Training?
2. Are features that are significant in Formation still significant in Training?
3. Is there temporal decay of predictive power?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys
sys.path.insert(0, '.')

from config import get_default_config

def compute_daily_ic(X: pd.DataFrame, y: pd.Series, dates: pd.Index) -> pd.DataFrame:
    """Compute daily cross-sectional IC (Spearman rank correlation)."""
    unique_dates = dates.unique()
    ic_series = []
    
    for date in unique_dates:
        mask = dates == date
        X_day = X[mask]
        y_day = y[mask]
        
        if len(y_day) < 10:
            continue
            
        ic_row = {'date': date}
        for col in X.columns:
            x_col = X_day[col].values
            valid = ~(np.isnan(x_col) | np.isnan(y_day.values))
            if valid.sum() >= 10:
                ic, _ = stats.spearmanr(x_col[valid], y_day.values[valid])
                ic_row[col] = ic
            else:
                ic_row[col] = np.nan
        ic_series.append(ic_row)
    
    return pd.DataFrame(ic_series).set_index('date')


def main():
    print("=" * 80)
    print("SIGNAL DECAY DIAGNOSTIC")
    print("=" * 80)
    
    # Load data
    panel_path = Path(r'D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet')
    panel_df = pd.read_parquet(panel_path)
    print(f"Panel shape: {panel_df.shape}")
    
    config = get_default_config()
    t0 = pd.Timestamp('2021-04-19')
    
    # Define windows
    formation_days = int(config.features.formation_years * 252)  # 5 years
    training_days = int(config.features.training_years * 252)    # 1 year
    embargo_days = config.time.HOLDING_PERIOD_DAYS               # 21 days
    
    # Formation: 5 years before Training
    t_training_end = t0 - pd.Timedelta(days=embargo_days)
    t_training_start = t_training_end - pd.Timedelta(days=training_days)
    t_formation_end = t_training_start - pd.Timedelta(days=1)
    t_formation_start = t_formation_end - pd.Timedelta(days=formation_days)
    
    print(f"\nFormation:  [{t_formation_start.date()}, {t_formation_end.date()}]")
    print(f"Training:   [{t_training_start.date()}, {t_training_end.date()}]")
    print(f"Test (t0):  {t0.date()}")
    
    # Get feature columns
    target_col = 'FwdRet_21'
    exclude_cols = ['Close', 'Ticker', 'ADV_63', 'ADV_63_Rank', 'market_cap'] + \
                   [c for c in panel_df.columns if c.startswith('FwdRet')]
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('_')]
    
    # Extract Formation data
    formation_mask = (
        (panel_df.index.get_level_values('Date') >= t_formation_start) &
        (panel_df.index.get_level_values('Date') <= t_formation_end)
    )
    panel_formation = panel_df[formation_mask]
    
    # Extract Training data
    training_mask = (
        (panel_df.index.get_level_values('Date') >= t_training_start) &
        (panel_df.index.get_level_values('Date') <= t_training_end)
    )
    panel_training = panel_df[training_mask]
    
    print(f"\nFormation samples: {len(panel_formation)}")
    print(f"Training samples: {len(panel_training)}")
    
    # Sample features for analysis (top 50 by variance to speed up)
    X_formation = panel_formation[feature_cols].astype(np.float32)
    y_formation = panel_formation[target_col].astype(np.float64)
    dates_formation = panel_formation.index.get_level_values('Date')
    
    X_training = panel_training[feature_cols].astype(np.float32)
    y_training = panel_training[target_col].astype(np.float64)
    dates_training = panel_training.index.get_level_values('Date')
    
    # Compute aggregate IC for all features in both windows
    print("\n" + "=" * 80)
    print("COMPUTING IC STATISTICS...")
    print("=" * 80)
    
    # Formation IC (simple mean, no time decay for diagnostic)
    formation_ic = {}
    for col in feature_cols[:100]:  # Sample first 100 for speed
        x = X_formation[col].values
        y = y_formation.values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 100:
            ic, _ = stats.spearmanr(x[valid], y[valid])
            formation_ic[col] = ic
    
    # Training IC
    training_ic = {}
    for col in feature_cols[:100]:
        x = X_training[col].values
        y = y_training.values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 100:
            ic, _ = stats.spearmanr(x[valid], y[valid])
            training_ic[col] = ic
    
    # Convert to DataFrames
    formation_ic_df = pd.Series(formation_ic).rename('formation_ic')
    training_ic_df = pd.Series(training_ic).rename('training_ic')
    
    # Merge
    ic_comparison = pd.concat([formation_ic_df, training_ic_df], axis=1).dropna()
    
    print(f"\nFeatures analyzed: {len(ic_comparison)}")
    
    # IC statistics
    print("\n--- FORMATION IC STATS ---")
    print(f"  Mean:   {ic_comparison['formation_ic'].mean():.4f}")
    print(f"  Median: {ic_comparison['formation_ic'].median():.4f}")
    print(f"  Std:    {ic_comparison['formation_ic'].std():.4f}")
    print(f"  |IC|>0.02: {(ic_comparison['formation_ic'].abs() > 0.02).sum()}")
    print(f"  |IC|>0.05: {(ic_comparison['formation_ic'].abs() > 0.05).sum()}")
    
    print("\n--- TRAINING IC STATS ---")
    print(f"  Mean:   {ic_comparison['training_ic'].mean():.4f}")
    print(f"  Median: {ic_comparison['training_ic'].median():.4f}")
    print(f"  Std:    {ic_comparison['training_ic'].std():.4f}")
    print(f"  |IC|>0.02: {(ic_comparison['training_ic'].abs() > 0.02).sum()}")
    print(f"  |IC|>0.05: {(ic_comparison['training_ic'].abs() > 0.05).sum()}")
    
    # Correlation between Formation and Training IC
    corr = ic_comparison['formation_ic'].corr(ic_comparison['training_ic'])
    print(f"\n--- IC PERSISTENCE ---")
    print(f"  Correlation(Formation IC, Training IC): {corr:.4f}")
    
    # Sign consistency
    same_sign = (np.sign(ic_comparison['formation_ic']) == np.sign(ic_comparison['training_ic'])).mean()
    print(f"  Sign consistency: {same_sign:.1%}")
    
    # Top features in Formation vs Training
    print("\n--- TOP 10 BY FORMATION IC ---")
    top_formation = ic_comparison.nlargest(10, 'formation_ic')
    for idx, row in top_formation.iterrows():
        print(f"  {idx[:40]:40s}  Form={row['formation_ic']:+.4f}  Train={row['training_ic']:+.4f}")
    
    print("\n--- TOP 10 BY TRAINING IC ---")
    top_training = ic_comparison.nlargest(10, 'training_ic')
    for idx, row in top_training.iterrows():
        print(f"  {idx[:40]:40s}  Form={row['formation_ic']:+.4f}  Train={row['training_ic']:+.4f}")
    
    # Analyze multicollinearity among top features
    print("\n" + "=" * 80)
    print("MULTICOLLINEARITY ANALYSIS (TOP 20 BY |IC|)")
    print("=" * 80)
    
    top_features = ic_comparison['training_ic'].abs().nlargest(20).index.tolist()
    X_top = X_training[top_features]
    
    # Correlation matrix
    corr_matrix = X_top.corr().abs()
    
    # Count high correlations (excluding diagonal)
    np.fill_diagonal(corr_matrix.values, 0)
    high_corr = (corr_matrix > 0.8).sum().sum() // 2  # Each pair counted twice
    total_pairs = len(top_features) * (len(top_features) - 1) // 2
    
    print(f"  High correlation pairs (|r|>0.8): {high_corr}/{total_pairs}")
    print(f"  Mean pairwise correlation: {corr_matrix.values[np.triu_indices(len(top_features), 1)].mean():.3f}")
    
    # Fit a simple OLS to see R² with all features
    print("\n" + "=" * 80)
    print("MULTIVARIATE SIGNAL STRENGTH")
    print("=" * 80)
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    # Use top 20 features
    X_train_top = X_training[top_features].dropna()
    y_train_aligned = y_training.loc[X_train_top.index]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_top)
    
    # Fit Ridge
    ridge = Ridge(alpha=0.01)
    ridge.fit(X_scaled, y_train_aligned)
    
    # Score
    r2 = ridge.score(X_scaled, y_train_aligned)
    print(f"  Ridge R² (top 20 features): {r2:.4f}")
    print(f"  Coefficients (sorted by |coef|):")
    
    coef_df = pd.DataFrame({
        'feature': top_features,
        'coef': ridge.coef_
    }).sort_values('coef', key=abs, ascending=False)
    
    for _, row in coef_df.head(10).iterrows():
        print(f"    {row['feature'][:40]:40s}  coef={row['coef']:+.4f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if corr < 0.3:
        print("\n[PROBLEM] LOW IC PERSISTENCE:")
        print("  Features significant in Formation don't stay significant in Training.")
        print("  This explains why multivariate models collapse to 0 features.")
        print("\n  POSSIBLE CAUSES:")
        print("  1. 5-year Formation is too far from 1-year Training (regime change)")
        print("  2. Momentum signals are non-stationary")
        print("  3. Feature interactions change over time")
        
    if same_sign < 0.6:
        print("\n[PROBLEM] SIGN INSTABILITY:")
        print("  Feature IC signs flip between Formation and Training.")
        print("  Model can't learn stable relationships.")
        
    if high_corr > total_pairs * 0.3:
        print("\n[PROBLEM] HIGH MULTICOLLINEARITY:")
        print("  Top features are highly correlated with each other.")
        print("  This makes coefficient estimation unstable.")
        
    if r2 < 0.01:
        print("\n[PROBLEM] WEAK MULTIVARIATE SIGNAL:")
        print(f"  Ridge R² = {r2:.4f} - the combined features explain almost nothing.")
        print("  This is consistent with Lasso/ElasticNet selecting 0 features.")


if __name__ == '__main__':
    main()
