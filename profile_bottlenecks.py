"""
Targeted Profiling Script for V3 Feature Selection Bottlenecks
==============================================================

Based on initial profiling, we identified two main bottlenecks:
1. Daily IC computation in formation_fdr: 191s (3.2 min)
2. ElasticNet tuning: convergence issues, very slow

This script profiles each stage independently to identify optimization opportunities.

Run from: D:\REPOSITORY\morias\Quant\strategies\crosssecmom2
Usage: python profile_bottlenecks.py
"""

import os
for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS", "LAPACK_NUM_THREADS"):
    os.environ.setdefault(var, "4")  # Allow some parallelism for numpy

import time
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from config import get_default_config


def load_test_data():
    """Load panel and prepare formation window data."""
    config = get_default_config()
    
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    
    t0 = time.time()
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    print(f"Panel loaded: {panel_df.shape} in {time.time()-t0:.2f}s")
    
    # Get formation window (5 years before first rebalance)
    # Use a fixed test date for reproducibility
    test_rebalance_date = pd.Timestamp('2021-04-19')
    formation_days = int(config.features.formation_years * 252)
    
    # Get dates in panel
    dates = panel_df.index.get_level_values('Date').unique().sort_values()
    # Find closest date <= test_rebalance_date
    mask = dates <= test_rebalance_date
    rebalance_idx = mask.sum() - 1 if mask.any() else 0
    formation_start_idx = max(0, rebalance_idx - formation_days)
    
    formation_start = dates[formation_start_idx]
    formation_end = dates[rebalance_idx - 1]  # Day before rebalance
    
    print(f"Formation window: {formation_start.date()} to {formation_end.date()}")
    
    # Filter to formation window
    mask = (panel_df.index.get_level_values('Date') >= formation_start) & \
           (panel_df.index.get_level_values('Date') <= formation_end)
    formation_df = panel_df.loc[mask].copy()
    
    print(f"Formation data: {formation_df.shape}")
    
    # Identify feature columns (exclude metadata)
    exclude_cols = {'Close', 'Open', 'High', 'Low', 'Volume', 'FwdRet_21', 
                    'Ticker', 'Date', 'ADV_63_rank'}
    feature_cols = [c for c in formation_df.columns if c not in exclude_cols 
                    and not c.startswith('FwdRet_')]
    
    print(f"Feature columns: {len(feature_cols)}")
    
    return formation_df, feature_cols, config


def profile_daily_ic_computation(formation_df, feature_cols, target_col='FwdRet_21'):
    """
    Profile the daily IC computation - the main bottleneck.
    
    Current approach: For each day, compute Spearman correlation between
    each feature and forward returns across all tickers.
    
    This is O(days × features × tickers) which is expensive.
    """
    print("\n" + "=" * 70)
    print("PROFILING: Daily IC Computation")
    print("=" * 70)
    
    # Get unique dates
    dates = formation_df.index.get_level_values('Date').unique().sort_values()
    n_dates = len(dates)
    n_features = len(feature_cols)
    
    print(f"Dates: {n_dates}")
    print(f"Features: {n_features}")
    print(f"Total IC computations: {n_dates * n_features:,}")
    
    # Approach 1: Current naive approach (loop over dates, compute per-date IC)
    print("\n[Method 1] Naive loop (current implementation)...")
    t0 = time.time()
    
    ic_matrix = np.full((n_dates, n_features), np.nan)
    
    for i, date in enumerate(dates[:50]):  # Only first 50 dates for timing
        day_data = formation_df.xs(date, level='Date')
        y = day_data[target_col].values
        
        for j, feat in enumerate(feature_cols[:100]):  # Only first 100 features
            x = day_data[feat].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 10:
                # Spearman correlation
                from scipy.stats import spearmanr
                ic_matrix[i, j] = spearmanr(x[mask], y[mask])[0]
    
    naive_time = time.time() - t0
    naive_per_ic = naive_time / (50 * 100)
    estimated_full = naive_per_ic * n_dates * n_features
    
    print(f"  50 dates × 100 features: {naive_time:.2f}s")
    print(f"  Per IC: {naive_per_ic*1000:.3f}ms")
    print(f"  Estimated full: {estimated_full:.1f}s ({estimated_full/60:.1f}min)")
    
    # Approach 2: Vectorized per-date (compute all features at once per date)
    print("\n[Method 2] Vectorized per-date...")
    t0 = time.time()
    
    def rank_array(arr):
        """Fast ranking using argsort."""
        temp = arr.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(arr))
        return ranks
    
    ic_matrix2 = np.full((n_dates, n_features), np.nan)
    
    for i, date in enumerate(dates[:50]):
        day_data = formation_df.xs(date, level='Date')
        y = day_data[target_col].values
        
        # Rank y once
        y_valid = ~np.isnan(y)
        if y_valid.sum() < 10:
            continue
        y_ranks = rank_array(y[y_valid])
        
        # Vectorized correlation for all features
        X = day_data[feature_cols[:100]].values[y_valid]
        
        for j in range(X.shape[1]):
            x = X[:, j]
            x_valid = ~np.isnan(x)
            if x_valid.sum() >= 10:
                x_ranks = rank_array(x[x_valid])
                # Pearson on ranks = Spearman
                ic_matrix2[i, j] = np.corrcoef(x_ranks, y_ranks[x_valid])[0, 1]
    
    vec_time = time.time() - t0
    vec_per_ic = vec_time / (50 * 100)
    estimated_full_vec = vec_per_ic * n_dates * n_features
    
    print(f"  50 dates × 100 features: {vec_time:.2f}s")
    print(f"  Per IC: {vec_per_ic*1000:.3f}ms")
    print(f"  Estimated full: {estimated_full_vec:.1f}s ({estimated_full_vec/60:.1f}min)")
    print(f"  Speedup vs naive: {naive_time/vec_time:.1f}x")
    
    # Approach 3: Fully vectorized using pandas groupby
    print("\n[Method 3] Pandas groupby + rank correlation...")
    t0 = time.time()
    
    # Subset for testing
    test_features = feature_cols[:100]
    test_df = formation_df[test_features + [target_col]].head(50 * 116)  # ~50 dates
    
    # Rank within each date
    def daily_rank_corr(group):
        y = group[target_col]
        results = {}
        for feat in test_features:
            x = group[feat]
            valid = x.notna() & y.notna()
            if valid.sum() >= 10:
                results[feat] = x[valid].rank().corr(y[valid].rank())
            else:
                results[feat] = np.nan
        return pd.Series(results)
    
    ic_df = test_df.groupby(level='Date').apply(daily_rank_corr)
    
    pandas_time = time.time() - t0
    print(f"  ~50 dates × 100 features: {pandas_time:.2f}s")
    print(f"  Speedup vs naive: {naive_time/pandas_time:.1f}x")
    
    return {
        'naive_time': naive_time,
        'vec_time': vec_time,
        'pandas_time': pandas_time,
        'estimated_full_naive': estimated_full,
        'estimated_full_vec': estimated_full_vec,
    }


def profile_elasticnet_tuning(formation_df, feature_cols, n_features=50):
    """
    Profile ElasticNet hyperparameter tuning.
    
    Issues identified:
    - Convergence warnings
    - Slow with many features
    """
    print("\n" + "=" * 70)
    print("PROFILING: ElasticNet Tuning")
    print("=" * 70)
    
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data (use subset of features)
    test_features = feature_cols[:n_features]
    
    # Get a single date's data for testing
    dates = formation_df.index.get_level_values('Date').unique()
    test_date = dates[-1]
    
    day_data = formation_df.xs(test_date, level='Date')
    X = day_data[test_features].values
    y = day_data['FwdRet_21'].values
    
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    print(f"Test data: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"X range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Test 1: Default ElasticNetCV
    print("\n[Test 1] Default ElasticNetCV...")
    t0 = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1 = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            n_alphas=20,
            cv=3,
            n_jobs=1,
            max_iter=1000,
        )
        model1.fit(X_scaled, y)
    
    default_time = time.time() - t0
    print(f"  Time: {default_time:.2f}s")
    print(f"  Best alpha: {model1.alpha_:.6f}")
    print(f"  Best l1_ratio: {model1.l1_ratio_}")
    print(f"  Non-zero coefs: {np.sum(model1.coef_ != 0)}/{len(model1.coef_)}")
    
    # Test 2: More iterations
    print("\n[Test 2] More iterations (max_iter=5000)...")
    t0 = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model2 = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            n_alphas=20,
            cv=3,
            n_jobs=1,
            max_iter=5000,
        )
        model2.fit(X_scaled, y)
    
    more_iter_time = time.time() - t0
    print(f"  Time: {more_iter_time:.2f}s")
    print(f"  Non-zero coefs: {np.sum(model2.coef_ != 0)}/{len(model2.coef_)}")
    
    # Test 3: Fewer alphas
    print("\n[Test 3] Fewer alphas (n_alphas=10)...")
    t0 = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model3 = ElasticNetCV(
            l1_ratio=[0.5],  # Single l1_ratio
            n_alphas=10,
            cv=3,
            n_jobs=1,
            max_iter=2000,
        )
        model3.fit(X_scaled, y)
    
    fewer_alpha_time = time.time() - t0
    print(f"  Time: {fewer_alpha_time:.2f}s")
    print(f"  Non-zero coefs: {np.sum(model3.coef_ != 0)}/{len(model3.coef_)}")
    
    # Test 4: Fixed alpha (no CV)
    print("\n[Test 4] Fixed alpha (no CV)...")
    from sklearn.linear_model import ElasticNet
    
    t0 = time.time()
    
    model4 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
    model4.fit(X_scaled, y)
    
    fixed_time = time.time() - t0
    print(f"  Time: {fixed_time:.2f}s")
    print(f"  Non-zero coefs: {np.sum(model4.coef_ != 0)}/{len(model4.coef_)}")
    
    # Scaling analysis
    print("\n[Scaling] Test with more features...")
    for n_feat in [50, 100, 200]:
        if n_feat > len(feature_cols):
            continue
            
        X_test = formation_df.xs(test_date, level='Date')[feature_cols[:n_feat]].values
        mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y)
        X_test = scaler.fit_transform(X_test[mask[:len(X_test)]])
        y_test = y[mask[:len(y)]]
        
        if len(X_test) < 20:
            continue
        
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ElasticNetCV(l1_ratio=[0.5], n_alphas=10, cv=3, max_iter=2000)
            m.fit(X_test, y_test[:len(X_test)])
        print(f"  {n_feat} features: {time.time()-t0:.2f}s")
    
    return {
        'default_time': default_time,
        'more_iter_time': more_iter_time,
        'fewer_alpha_time': fewer_alpha_time,
        'fixed_time': fixed_time,
    }


def profile_correlation_filter(formation_df, feature_cols, n_features=200):
    """Profile the correlation-based redundancy filter."""
    print("\n" + "=" * 70)
    print("PROFILING: Correlation Filter")
    print("=" * 70)
    
    # Get subset of features
    test_features = feature_cols[:n_features]
    
    # Use single date for testing
    dates = formation_df.index.get_level_values('Date').unique()
    test_date = dates[-1]
    
    day_data = formation_df.xs(test_date, level='Date')[test_features]
    
    print(f"Testing with {n_features} features, {len(day_data)} samples")
    
    # Method 1: Full correlation matrix
    print("\n[Method 1] Full correlation matrix (pandas)...")
    t0 = time.time()
    corr_matrix = day_data.corr()
    pandas_time = time.time() - t0
    print(f"  Time: {pandas_time:.4f}s")
    
    # Method 2: Numpy corrcoef
    print("\n[Method 2] Numpy corrcoef...")
    t0 = time.time()
    X = day_data.values.T  # features × samples
    # Handle NaN by filling with column mean
    X_filled = np.where(np.isnan(X), np.nanmean(X, axis=1, keepdims=True), X)
    corr_np = np.corrcoef(X_filled)
    numpy_time = time.time() - t0
    print(f"  Time: {numpy_time:.4f}s")
    
    # Greedy selection simulation
    print("\n[Method 3] Greedy selection (corr > 0.7)...")
    t0 = time.time()
    
    selected = []
    available = list(range(n_features))
    corr_threshold = 0.7
    
    while available:
        # Select first available
        idx = available.pop(0)
        selected.append(idx)
        
        # Remove correlated features
        to_remove = []
        for other_idx in available:
            if abs(corr_np[idx, other_idx]) > corr_threshold:
                to_remove.append(other_idx)
        
        for r in to_remove:
            available.remove(r)
    
    greedy_time = time.time() - t0
    print(f"  Time: {greedy_time:.4f}s")
    print(f"  Selected: {len(selected)}/{n_features} features")
    
    return {
        'pandas_corr_time': pandas_time,
        'numpy_corr_time': numpy_time,
        'greedy_time': greedy_time,
        'n_selected': len(selected),
    }


def main():
    """Run all profiling tests."""
    print("=" * 70)
    print("V3 FEATURE SELECTION BOTTLENECK PROFILER")
    print("=" * 70)
    
    # Load data
    formation_df, feature_cols, config = load_test_data()
    
    # Profile each stage
    results = {}
    
    # 1. Daily IC computation (main bottleneck)
    results['daily_ic'] = profile_daily_ic_computation(formation_df, feature_cols)
    
    # 2. ElasticNet tuning
    results['elasticnet'] = profile_elasticnet_tuning(formation_df, feature_cols)
    
    # 3. Correlation filter
    results['corr_filter'] = profile_correlation_filter(formation_df, feature_cols)
    
    # Summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    
    print("\n[Daily IC Computation]")
    print(f"  Current (naive): ~{results['daily_ic']['estimated_full_naive']:.0f}s ({results['daily_ic']['estimated_full_naive']/60:.1f}min)")
    print(f"  Potential (vectorized): ~{results['daily_ic']['estimated_full_vec']:.0f}s ({results['daily_ic']['estimated_full_vec']/60:.1f}min)")
    
    print("\n[ElasticNet Tuning]")
    print(f"  Default: {results['elasticnet']['default_time']:.1f}s")
    print(f"  Fixed alpha: {results['elasticnet']['fixed_time']:.1f}s")
    
    print("\n[Correlation Filter]")
    print(f"  Correlation matrix: {results['corr_filter']['pandas_corr_time']:.3f}s")
    print(f"  Greedy selection: {results['corr_filter']['greedy_time']:.3f}s")
    
    print("\n[RECOMMENDATIONS]")
    print("1. Vectorize daily IC computation - potential 3-5x speedup")
    print("2. Reduce ElasticNet search space or use fixed alpha")
    print("3. Pre-filter features with high NaN rate before IC computation")
    print("4. Consider downsampling formation period (weekly instead of daily)")
    
    return results


if __name__ == "__main__":
    results = main()
