"""
Integration test for walk-forward engine with feature selection pipeline.

This test verifies that:
1. per_window_pipeline integrates correctly with walk_forward_engine
2. Models are trained on selected features only (subset design)
3. Scoring works correctly at each rebalance
4. Signal features are consistently selected
5. No exceptions during multi-window walk-forward

Run with: python test_walk_forward_integration.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import functions to test
from walk_forward_engine import walk_forward_backtest
from config import ResearchConfig


# ============================================================================
# Test Fixtures: Synthetic Data with Known Signal
# ============================================================================

def create_synthetic_panel_with_signal():
    """
    Generate synthetic panel with known signal features.
    
    Features 0-4: Strong predictive signal (correlated with forward returns)
    Features 5-9: Pure noise (no predictive power)
    
    Returns:
        panel: DataFrame with multi-index (Date, Ticker)
        metadata: DataFrame with ticker information
        config: ResearchConfig object
    """
    np.random.seed(42)
    
    # Panel dimensions
    n_days = 500
    n_tickers = 20
    n_signal_features = 5
    n_noise_features = 5
    n_features = n_signal_features + n_noise_features
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # Create tickers
    tickers = [f'TICKER_{i:02d}' for i in range(n_tickers)]
    
    # Create multi-index
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['Date', 'Ticker']
    )
    
    # Initialize panel
    panel_data = {}
    
    # Generate signal features (0-4): correlated with future returns
    signal_strength = 0.3  # Moderate signal
    true_alpha = np.random.randn(n_tickers) * 0.02  # Ticker-specific alpha
    
    for feat_idx in range(n_signal_features):
        feature_vals = []
        for date_idx, date in enumerate(dates):
            # Feature value with some persistence and noise
            if date_idx == 0:
                vals = np.random.randn(n_tickers) * 0.5
            else:
                # Add persistence: 70% previous + 30% new
                vals = 0.7 * feature_vals[-1] + 0.3 * np.random.randn(n_tickers) * 0.5
            feature_vals.append(vals)
        
        feature_series = np.concatenate(feature_vals)
        panel_data[f'feature_{feat_idx}'] = feature_series
    
    # Generate noise features (5-9): no predictive power
    for feat_idx in range(n_signal_features, n_features):
        feature_vals = np.random.randn(len(index)) * 0.5
        panel_data[f'feature_{feat_idx}'] = feature_vals
    
    # Generate forward returns based on signal features
    returns = []
    for date_idx, date in enumerate(dates):
        # Combine signal features with weights
        signal = np.zeros(n_tickers)
        for feat_idx in range(n_signal_features):
            feat_slice = panel_data[f'feature_{feat_idx}'][
                date_idx * n_tickers:(date_idx + 1) * n_tickers
            ]
            signal += feat_slice * signal_strength / n_signal_features
        
        # Add ticker-specific alpha and noise
        ret = true_alpha + signal + np.random.randn(n_tickers) * 0.05
        returns.append(ret)
    
    returns_series = np.concatenate(returns)
    panel_data['ret_fwd_5d'] = returns_series
    
    # Add market cap for position sizing
    market_caps = np.tile(
        np.random.lognormal(mean=8, sigma=1.5, size=n_tickers),
        n_days
    )
    panel_data['market_cap'] = market_caps * 1e6  # In millions
    
    # Create panel DataFrame
    panel = pd.DataFrame(panel_data, index=index)
    
    # Create metadata
    metadata = pd.DataFrame({
        'ticker': tickers,
        'sector': [f'SECTOR_{i % 4}' for i in range(n_tickers)],  # 4 sectors
        'region': ['US'] * n_tickers
    })
    
    # Create config
    config = ResearchConfig()
    config.TRAINING_WINDOW_DAYS = 200
    config.STEP_SIZE_DAYS = 60
    config.HOLDING_PERIOD_DAYS = 5
    config.MIN_OBSERVATIONS = 100
    config.FORMATION_IC_THRESHOLD = 0.00  # Low threshold to ensure features pass
    config.MIN_STABILITY_PVALUE = 0.30  # Lower threshold for test
    config.MAX_FEATURES_SELECTED = 10
    config.ALPHA_ELASTICNET = [0.01, 0.1, 1.0]
    config.L1_RATIO = [0.5, 0.7, 0.9]
    
    return panel, metadata, config


# ============================================================================
# Integration Tests
# ============================================================================

def test_walk_forward_with_feature_selection():
    """
    Test walk-forward backtest with feature selection pipeline.
    
    Verifies:
    1. No exceptions during multi-window execution
    2. Models created at each rebalance
    3. Scores have correct shape
    4. Signal features (0-4) are frequently selected
    5. Diagnostics captured per window
    """
    panel, metadata, config = create_synthetic_panel_with_signal()
    
    # Run walk-forward backtest
    results = walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,  # Use default (earliest date + training window)
        end_date=None,    # Use default (latest date)
        rebalance_frequency='quarterly',  # Will produce 2-3 rebalances
        parallel=False,   # Sequential for easier debugging
        n_jobs=1,
        verbose=True
    )
    
    # Basic assertions
    assert results is not None, "Walk-forward should return results"
    assert 'returns' in results, "Results should contain returns DataFrame"
    assert 'diagnostics' in results, "Results should contain diagnostics"
    
    returns_df = results['returns']
    diagnostics = results['diagnostics']
    
    # Check we got multiple rebalances
    assert len(returns_df) >= 2, f"Expected at least 2 rebalances, got {len(returns_df)}"
    
    # Check returns DataFrame structure
    expected_cols = ['date', 'realized_return', 'n_long', 'n_short', 
                     'train_time', 'score_time', 'portfolio_time']
    for col in expected_cols:
        assert col in returns_df.columns, f"Missing column: {col}"
    
    # Check all rebalances produced valid returns
    assert returns_df['realized_return'].notna().all(), "All rebalances should have returns"
    
    # Check diagnostics captured per window
    assert len(diagnostics) >= 2, f"Expected at least 2 diagnostic entries, got {len(diagnostics)}"
    
    for diag in diagnostics:
        assert 'date' in diag, "Diagnostics should have date"
        assert 'n_features' in diag, "Diagnostics should have n_features"
        assert 'selected_features' in diag, "Diagnostics should have selected_features"
        
        # Check features were selected
        n_features = diag['n_features']
        selected_features = diag['selected_features']
        assert n_features > 0, "At least some features should be selected"
        assert len(selected_features) == n_features, "selected_features length should match n_features"
    
    # Check signal features are frequently selected
    all_selected = []
    for diag in diagnostics:
        all_selected.extend(diag['selected_features'])
    
    # Count feature occurrences
    from collections import Counter
    feature_counts = Counter(all_selected)
    
    # Signal features (0-4) should appear more often than noise features (5-9)
    signal_count = sum(feature_counts.get(f'feature_{i}', 0) for i in range(5))
    noise_count = sum(feature_counts.get(f'feature_{i}', 0) for i in range(5, 10))
    
    print(f"\nFeature selection statistics:")
    print(f"  Signal features (0-4) selected: {signal_count} times")
    print(f"  Noise features (5-9) selected: {noise_count} times")
    print(f"  Top 5 features: {feature_counts.most_common(5)}")
    
    # Signal features should dominate (at least 60% of selections)
    if signal_count + noise_count > 0:
        signal_ratio = signal_count / (signal_count + noise_count)
        assert signal_ratio >= 0.5, (
            f"Signal features should be selected frequently, "
            f"but only {signal_ratio:.1%} of selections were signal features"
        )
        print(f"  Signal ratio: {signal_ratio:.1%} (target: ≥50%)")


def test_walk_forward_model_subset_design():
    """
    Test that models use subset design (operate on selected features only).
    
    Verifies:
    1. Model's selected_features attribute is populated
    2. Model's scaling_params contain only selected features
    3. Scoring works with subset features
    """
    panel, metadata, config = create_synthetic_panel_with_signal()
    
    # Run single rebalance
    config.STEP_SIZE_DAYS = 999  # Large step to get only one rebalance
    
    results = walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,
        end_date=None,
        rebalance_frequency='quarterly',
        parallel=False,
        n_jobs=1,
        verbose=True
    )
    
    # Get diagnostics for first window
    diagnostics = results['diagnostics']
    assert len(diagnostics) >= 1, "Should have at least one diagnostic entry"
    
    first_diag = diagnostics[0]
    selected_features = first_diag['selected_features']
    n_features = first_diag['n_features']
    
    print(f"\nSubset design verification:")
    print(f"  Selected features: {selected_features}")
    print(f"  Number of features: {n_features}")
    
    # Basic checks
    assert n_features > 0, "Should select at least some features"
    assert len(selected_features) == n_features, "Feature count should match"
    assert n_features < 10, "Should not select all features (subset design)"


def test_walk_forward_parallel_execution():
    """
    Test parallel execution path produces same results as sequential.
    
    Note: Due to potential numerical differences in parallel execution,
    we check that results are similar, not identical.
    """
    panel, metadata, config = create_synthetic_panel_with_signal()
    
    # Run sequential
    results_seq = walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,
        end_date=None,
        rebalance_frequency='quarterly',
        parallel=False,
        n_jobs=1,
        verbose=False
    )
    
    # Run parallel
    results_par = walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,
        end_date=None,
        rebalance_frequency='quarterly',
        parallel=True,
        n_jobs=2,
        verbose=False
    )
    
    # Check both produced results
    assert results_seq is not None, "Sequential execution should succeed"
    assert results_par is not None, "Parallel execution should succeed"
    
    # Check same number of rebalances
    seq_returns = results_seq['returns']
    par_returns = results_par['returns']
    assert len(seq_returns) == len(par_returns), (
        f"Sequential and parallel should have same rebalances: "
        f"{len(seq_returns)} vs {len(par_returns)}"
    )
    
    # Check returns are similar (allow for numerical differences)
    seq_vals = seq_returns['realized_return'].values
    par_vals = par_returns['realized_return'].values
    correlation = np.corrcoef(seq_vals, par_vals)[0, 1]
    
    print(f"\nParallel execution verification:")
    print(f"  Sequential returns: {seq_vals}")
    print(f"  Parallel returns: {par_vals}")
    print(f"  Correlation: {correlation:.4f}")
    
    assert correlation > 0.95, (
        f"Parallel and sequential results should be highly correlated, "
        f"got correlation={correlation:.4f}"
    )


if __name__ == '__main__':
    print("="*70)
    print("Test 1: Walk-forward with feature selection")
    print("="*70)
    try:
        test_walk_forward_with_feature_selection()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    print("\n" + "="*70)
    print("Test 2: Model subset design")
    print("="*70)
    try:
        test_walk_forward_model_subset_design()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    print("\n" + "="*70)
    print("Test 3: Parallel execution")
    print("="*70)
    try:
        test_walk_forward_parallel_execution()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    print("\n" + "="*70)
    print("All tests completed")
    print("="*70)
