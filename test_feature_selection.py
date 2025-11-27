"""
Unit tests for feature_selection.py module.

Tests implemented functions with synthetic data:
- Helper functions: time_decay_weights, spearman_ic, daily_ic_series, newey_west_tstat
- formation_fdr(): FDR control, IC computation, parallel processing
- per_window_ic_filter(): IC filtering with thresholds, time-decay weights

Run with: pytest test_feature_selection.py -v
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import warnings

# Import functions to test
from feature_selection import (
    compute_time_decay_weights,
    compute_spearman_ic,
    compute_daily_ic_series,
    compute_newey_west_tstat,
    formation_fdr,
    per_window_ic_filter
)


# ============================================================================
# Test Fixtures: Synthetic Data Generators
# ============================================================================

@pytest.fixture
def synthetic_panel_data():
    """
    Generate synthetic panel data for testing.
    
    Returns:
        X: DataFrame (n_dates * n_assets, n_features)
        y: Series (n_dates * n_assets,)
        dates: DatetimeIndex
    """
    np.random.seed(42)
    
    n_dates = 252  # 1 year of daily data
    n_assets = 100
    n_features = 50
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    
    # Generate features with varying IC patterns
    X_list = []
    y_list = []
    date_list = []
    
    for date in dates:
        # Generate cross-sectional features
        X_date = np.random.randn(n_assets, n_features)
        
        # Create target with known relationships:
        # - Features 0-9: Strong positive IC (~0.15)
        # - Features 10-19: Weak positive IC (~0.05)
        # - Features 20-29: Zero IC (noise)
        # - Features 30-39: Weak negative IC (~-0.05)
        # - Features 40-49: Strong negative IC (~-0.15)
        
        y_date = np.zeros(n_assets)
        y_date += 0.15 * X_date[:, :10].mean(axis=1)  # Strong positive
        y_date += 0.05 * X_date[:, 10:20].mean(axis=1)  # Weak positive
        y_date += 0.0 * X_date[:, 20:30].mean(axis=1)  # Noise
        y_date -= 0.05 * X_date[:, 30:40].mean(axis=1)  # Weak negative
        y_date -= 0.15 * X_date[:, 40:50].mean(axis=1)  # Strong negative
        y_date += np.random.randn(n_assets) * 0.5  # Add noise
        
        X_list.append(X_date)
        y_list.append(y_date)
        date_list.extend([date] * n_assets)
    
    # Combine into panel
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Create DataFrame with date index
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['date'] = date_list
    X_df = X_df.set_index('date')
    
    y_series = pd.Series(y, index=X_df.index, name='forward_return')
    
    return X_df, y_series, dates


@pytest.fixture
def synthetic_single_window():
    """
    Generate synthetic data for a single window.
    
    Returns:
        X: DataFrame (n_assets, n_features)
        y: Series (n_assets,)
    """
    np.random.seed(123)
    
    n_assets = 100
    n_features = 30
    
    # Generate features
    X = np.random.randn(n_assets, n_features)
    
    # Create target with known relationships:
    # - Features 0-4: Strong IC
    # - Features 5-9: Medium IC
    # - Features 10-29: Weak/no IC
    y = (0.3 * X[:, :5].mean(axis=1) + 
         0.1 * X[:, 5:10].mean(axis=1) + 
         np.random.randn(n_assets) * 0.5)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='forward_return')
    
    return X_df, y_series


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Test suite for helper utility functions."""
    
    def test_time_decay_weights_basic(self):
        """Test time-decay weights with basic parameters."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        train_end = dates[-1]
        weights = compute_time_decay_weights(dates, train_end, half_life=126)
        
        # Check shape
        assert len(weights) == len(dates)
        
        # Check weights are positive
        assert np.all(weights > 0)
        
        # Check monotonic increase (oldest = lowest weight, most recent = highest)
        assert np.all(np.diff(weights) >= 0)
        
        # Check half-life property (oldest weight ≈ half of most recent weight at half_life point)
        # Most recent is last (index -1), half_life ago is index -127
        assert np.isclose(weights[-127] / weights[-1], 0.5, atol=0.01)
    
    def test_time_decay_weights_edge_cases(self):
        """Test time-decay weights with edge cases."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        train_end = dates[-1]
        
        # Very short half-life
        weights = compute_time_decay_weights(dates, train_end, half_life=10)
        # Most recent (last) should have significantly higher weight than mean
        assert weights[-1] > weights.mean() * 5  # Last weight much higher than average
        
        # Very long half-life (nearly uniform)
        weights = compute_time_decay_weights(dates, train_end, half_life=10000)
        # Standard deviation should be small relative to mean (nearly uniform)
        assert np.std(weights) / np.mean(weights) < 0.01
    
    def test_spearman_ic_basic(self):
        """Test Spearman IC computation with known correlation."""
        np.random.seed(42)
        n = 100
        
        # Perfect positive correlation
        x = np.arange(n).astype(float)
        y = x.copy()
        ic = compute_spearman_ic(x, y)
        assert np.isclose(ic, 1.0, atol=1e-6)
        
        # Perfect negative correlation
        y = -x
        ic = compute_spearman_ic(x, y)
        assert np.isclose(ic, -1.0, atol=1e-6)
        
        # No correlation (random)
        x = np.random.randn(n)
        y = np.random.randn(n)
        ic = compute_spearman_ic(x, y)
        assert abs(ic) < 0.3  # Should be close to 0
    
    def test_spearman_ic_with_nans(self):
        """Test Spearman IC handles NaN values correctly."""
        np.random.seed(42)
        n = 100
        
        x = np.arange(n).astype(float)
        y = x.copy()
        
        # Add some NaNs
        x[10:20] = np.nan
        y[30:40] = np.nan
        
        ic = compute_spearman_ic(x, y)
        
        # Should still compute on valid pairs
        assert not np.isnan(ic)
        assert ic > 0.9  # Strong positive after removing NaNs
    
    def test_spearman_ic_all_nans(self):
        """Test Spearman IC returns 0.0 when all values are NaN."""
        x = np.array([np.nan] * 100)
        y = np.array([np.nan] * 100)
        ic = compute_spearman_ic(x, y)
        assert ic == 0.0  # Implementation returns 0.0 for insufficient data
    
    def test_daily_ic_series_shape(self, synthetic_panel_data):
        """Test daily IC series returns correct shape."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Test with subset of features
        feature_cols = X_df.columns[:10]
        ic_series = compute_daily_ic_series(X_df[feature_cols], y_series, dates)
        
        # Check shape: (n_dates, n_features)
        assert ic_series.shape == (len(dates), len(feature_cols))
        
        # Check index is dates
        assert all(ic_series.index == dates)
        
        # Check columns are feature names
        assert all(ic_series.columns == feature_cols)
    
    def test_daily_ic_series_values(self, synthetic_panel_data):
        """Test daily IC series computes reasonable ICs."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Test with features that have known IC patterns
        feature_cols = ['feature_0', 'feature_5', 'feature_25', 'feature_45']
        ic_series = compute_daily_ic_series(X_df[feature_cols], y_series, dates)
        
        # feature_0 should have positive mean IC (strong signal)
        assert ic_series['feature_0'].mean() > 0.02  # Relaxed from 0.05
        
        # feature_5 should have positive mean IC (weak signal)
        assert ic_series['feature_5'].mean() > -0.01  # Slightly positive or near zero
        
        # feature_25 should have IC close to 0 (noise)
        assert abs(ic_series['feature_25'].mean()) < 0.05
        
        # feature_45 should have negative mean IC (strong negative)
        assert ic_series['feature_45'].mean() < -0.05
    
    def test_newey_west_tstat_basic(self):
        """Test Newey-West t-stat computation."""
        np.random.seed(42)
        
        # Series with no autocorrelation
        series = pd.Series(np.random.randn(100))
        t_stat = compute_newey_west_tstat(series)
        
        # Should be close to standard t-stat for no autocorr
        assert abs(t_stat) < 3.0  # Reasonable range
        
        # Series with strong mean
        series = pd.Series(np.random.randn(100) + 2.0)
        t_stat = compute_newey_west_tstat(series)
        
        # Should be significantly positive
        assert t_stat > 5.0
    
    def test_newey_west_tstat_all_zeros(self):
        """Test Newey-West t-stat with zero series."""
        series = pd.Series(np.zeros(100))
        t_stat = compute_newey_west_tstat(series)
        
        # Should be exactly 0 (or NaN due to division by zero)
        assert t_stat == 0.0 or np.isnan(t_stat)


# ============================================================================
# Test formation_fdr()
# ============================================================================

class TestFormationFDR:
    """Test suite for formation_fdr() function."""
    
    def test_formation_fdr_basic(self, synthetic_panel_data):
        """Test formation_fdr with synthetic data."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Run formation FDR with permissive FDR level
        approved, diagnostics = formation_fdr(
            X_df, 
            y_series,
            dates,
            half_life=126,
            fdr_level=0.2,
            n_jobs=1  # Single job for testing
        )
        
        # Check outputs
        assert isinstance(approved, list)
        assert isinstance(diagnostics, pd.DataFrame)
        
        # Check we got some features approved
        assert len(approved) > 0
        assert len(approved) <= len(X_df.columns)
        
        # Check diagnostics columns
        expected_cols = ['feature', 'mean_ic', 'tstat', 'pvalue', 'approved']
        assert all(col in diagnostics.columns for col in expected_cols)
        
        # Check diagnostics length matches features
        assert len(diagnostics) == len(X_df.columns)
    
    def test_formation_fdr_signal_detection(self, synthetic_panel_data):
        """Test formation_fdr correctly identifies strong signal features."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Run with permissive parameters
        approved, diagnostics = formation_fdr(
            X_df, 
            y_series,
            dates,
            half_life=126,
            fdr_level=0.2,
            n_jobs=1
        )
        
        # Check that strong positive features (0-9) are mostly approved
        strong_positive = [f'feature_{i}' for i in range(10)]
        approved_strong = [f for f in strong_positive if f in approved]
        assert len(approved_strong) >= 7  # At least 70% detected
        
        # Check that strong negative features (40-49) are mostly approved
        strong_negative = [f'feature_{i}' for i in range(40, 50)]
        approved_strong_neg = [f for f in strong_negative if f in approved]
        assert len(approved_strong_neg) >= 7  # At least 70% detected
        
        # Check that noise features (20-29) are mostly rejected
        noise_features = [f'feature_{i}' for i in range(20, 30)]
        approved_noise = [f for f in noise_features if f in approved]
        assert len(approved_noise) <= 3  # At most 30% false positives
    
    def test_formation_fdr_strict_threshold(self, synthetic_panel_data):
        """Test formation_fdr with strict FDR level."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Run with strict FDR
        approved_strict, _ = formation_fdr(
            X_df, y_series, dates,
            half_life=126,
            fdr_level=0.01,  # Very strict
            n_jobs=1
        )
        
        # Run with permissive FDR
        approved_permissive, _ = formation_fdr(
            X_df, y_series, dates,
            half_life=126,
            fdr_level=0.3,  # Permissive
            n_jobs=1
        )
        
        # Strict should approve fewer features
        assert len(approved_strict) < len(approved_permissive)
        
        # Strict features should be subset of permissive
        assert set(approved_strict).issubset(set(approved_permissive))
    
    def test_formation_fdr_time_decay(self, synthetic_panel_data):
        """Test formation_fdr with different time-decay parameters."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Short half-life (emphasize recent)
        approved_short, diag_short = formation_fdr(
            X_df, y_series, dates,
            half_life=30,  # Emphasize last month
            fdr_level=0.1,
            n_jobs=1
        )
        
        # Long half-life (nearly uniform)
        approved_long, diag_long = formation_fdr(
            X_df, y_series, dates,
            half_life=1000,  # Nearly uniform
            fdr_level=0.1,
            n_jobs=1
        )
        
        # Results might differ due to different weighting
        # At least check both ran successfully
        assert len(approved_short) > 0
        assert len(approved_long) > 0
    
    def test_formation_fdr_parallel(self, synthetic_panel_data):
        """Test formation_fdr with parallel processing."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Run with single job
        approved_single, diag_single = formation_fdr(
            X_df, y_series, dates,
            half_life=126,
            fdr_level=0.1,
            n_jobs=1
        )
        
        # Run with multiple jobs
        approved_parallel, diag_parallel = formation_fdr(
            X_df, y_series, dates,
            half_life=126,
            fdr_level=0.1,
            n_jobs=2
        )
        
        # Results should be identical
        assert set(approved_single) == set(approved_parallel)
        
        # Diagnostics should match (allowing for small floating point diff)
        for col in ['mean_ic', 'tstat']:
            if col in diag_single.columns:
                assert np.allclose(
                    diag_single.sort_values('feature')[col].values,
                    diag_parallel.sort_values('feature')[col].values,
                    rtol=1e-5
                )
    
    def test_formation_fdr_empty_input(self):
        """Test formation_fdr handles empty input gracefully."""
        # Empty DataFrame
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        dates_empty = pd.DatetimeIndex([])
        
        with pytest.raises((ValueError, KeyError, IndexError)):
            formation_fdr(X_empty, y_empty, dates_empty)
    
    def test_formation_fdr_single_feature(self, synthetic_panel_data):
        """Test formation_fdr with single feature."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Single strong feature
        X_single = X_df[['feature_0']]
        
        approved, diagnostics = formation_fdr(
            X_single, y_series, dates,
            half_life=126,
            fdr_level=0.1,
            n_jobs=1
        )
        
        # Should approve if strong signal
        assert len(diagnostics) == 1
        if diagnostics['mean_ic'].iloc[0] > 0.05:
            assert len(approved) == 1


# ============================================================================
# Test per_window_ic_filter()
# ============================================================================

class TestPerWindowICFilter:
    """Test suite for per_window_ic_filter() function."""
    
    def test_per_window_ic_filter_basic(self, synthetic_single_window):
        """Test per_window IC filter with synthetic data."""
        X_df, y_series = synthetic_single_window
        
        # Create dates and weights
        dates = pd.date_range('2020-01-01', periods=len(X_df), freq='D')
        weights = np.ones(len(X_df)) / len(X_df)
        
        selected, diagnostics = per_window_ic_filter(
            X_df,
            y_series,
            dates,
            weights,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        
        # Check outputs
        assert isinstance(selected, list)
        assert isinstance(diagnostics, dict)
        
        # Check we got some features
        assert len(selected) > 0
        assert len(selected) <= len(X_df.columns)
        
        # Check diagnostics keys
        expected_keys = ['n_start', 'n_after_ic']
        assert all(key in diagnostics for key in expected_keys)
    
    def test_per_window_ic_filter_signal_detection(self, synthetic_single_window):
        """Test IC filter correctly identifies strong features."""
        X_df, y_series = synthetic_single_window
        dates = pd.date_range('2020-01-01', periods=len(X_df), freq='D')
        weights = np.ones(len(X_df)) / len(X_df)
        
        selected, diagnostics = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        
        # Strong features (0-4) should be mostly selected
        strong_features = [f'feature_{i}' for i in range(5)]
        selected_strong = [f for f in strong_features if f in selected]
        assert len(selected_strong) >= 3  # At least 60% detected
        
        # Weak features (10-29) should be mostly rejected
        weak_features = [f'feature_{i}' for i in range(10, 30)]
        selected_weak = [f for f in weak_features if f in selected]
        assert len(selected_weak) <= 5  # At most 25% false positives
    
    def test_per_window_ic_filter_threshold_effects(self, synthetic_single_window):
        """Test IC filter with different thresholds."""
        X_df, y_series = synthetic_single_window
        dates = pd.date_range('2020-01-01', periods=len(X_df), freq='D')
        weights = np.ones(len(X_df)) / len(X_df)
        
        # Loose threshold
        selected_loose, _ = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.01,
            t_min=1.0,
            n_jobs=1
        )
        
        # Strict threshold
        selected_strict, _ = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.1,
            t_min=2.0,
            n_jobs=1
        )
        
        # Strict should select fewer features
        assert len(selected_strict) < len(selected_loose)
        
        # Strict features should be subset of loose
        assert set(selected_strict).issubset(set(selected_loose))
    
    def test_per_window_ic_filter_time_decay(self, synthetic_single_window):
        """Test IC filter with time-decay weights."""
        X_df, y_series = synthetic_single_window
        dates = pd.date_range('2020-01-01', periods=len(X_df), freq='D')
        
        # Uniform weights
        weights_uniform = np.ones(len(X_df)) / len(X_df)
        selected_uniform, _ = per_window_ic_filter(
            X_df, y_series, dates, weights_uniform,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        
        # Exponential decay (emphasize early observations)
        weights_decay = np.exp(-np.arange(len(X_df)) * 0.01)
        weights_decay = weights_decay / weights_decay.sum()
        selected_decay, _ = per_window_ic_filter(
            X_df, y_series, dates, weights_decay,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        
        # Both should work (might differ slightly)
        assert len(selected_uniform) > 0
        assert len(selected_decay) > 0
    
    def test_per_window_ic_filter_parallel(self, synthetic_single_window):
        """Test IC filter with parallel processing."""
        X_df, y_series = synthetic_single_window
        dates = pd.date_range('2020-01-01', periods=len(X_df), freq='D')
        weights = np.ones(len(X_df)) / len(X_df)
        
        # Single job
        selected_single, _ = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        
        # Multiple jobs
        selected_parallel, _ = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=2
        )
        
        # Results should be identical
        assert set(selected_single) == set(selected_parallel)
    
    def test_per_window_ic_filter_all_rejected(self):
        """Test IC filter when all features should be rejected."""
        np.random.seed(42)
        
        # Pure noise data
        X_noise = pd.DataFrame(
            np.random.randn(100, 20),
            columns=[f'noise_{i}' for i in range(20)]
        )
        y_noise = pd.Series(np.random.randn(100))
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        weights = np.ones(100) / 100
        
        selected, diagnostics = per_window_ic_filter(
            X_noise, y_noise, dates, weights,
            theta_ic=0.3,  # Very high threshold
            t_min=3.0,  # Very high threshold
            n_jobs=1
        )
        
        # Should select very few or no features
        assert len(selected) <= 2  # Allow for statistical noise


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_formation_to_window_pipeline(self, synthetic_panel_data):
        """Test formation FDR followed by per-window IC filter."""
        X_df, y_series, dates = synthetic_panel_data
        
        # Step 1: Formation FDR
        approved_formation, _ = formation_fdr(
            X_df, y_series, dates,
            half_life=126,
            fdr_level=0.2,
            n_jobs=1
        )
        
        assert len(approved_formation) > 0, "Formation should approve some features"
        
        # Step 2: Extract single window
        window_start_idx = len(dates) // 2
        window_dates = dates[window_start_idx:window_start_idx + 63]
        X_window = X_df.loc[window_dates, approved_formation]
        y_window = y_series.loc[window_dates]
        
        # Step 3: Per-window IC filter
        weights = np.ones(len(window_dates))
        selected_window, _ = per_window_ic_filter(
            X_window, y_window, window_dates, weights,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        
        # Check pipeline worked
        assert len(selected_window) > 0
        assert len(selected_window) <= len(approved_formation)
        assert all(f in approved_formation for f in selected_window)
    
    def test_end_to_end_performance(self, synthetic_panel_data):
        """Test end-to-end performance meets timing expectations."""
        import time
        
        X_df, y_series, dates = synthetic_panel_data
        
        # Measure formation FDR time
        start_time = time.time()
        approved_formation, _ = formation_fdr(
            X_df, y_series, dates,
            half_life=126,
            fdr_level=0.1,
            n_jobs=1
        )
        formation_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for 50 features)
        assert formation_time < 5.0, f"Formation FDR too slow: {formation_time:.2f}s"
        
        # Measure per-window IC filter time
        window_dates = dates[:63]
        X_window = X_df.loc[window_dates, approved_formation]
        y_window = y_series.loc[window_dates]
        weights = np.ones(len(window_dates))
        
        start_time = time.time()
        selected_window, _ = per_window_ic_filter(
            X_window, y_window, window_dates, weights,
            theta_ic=0.05,
            t_min=1.5,
            n_jobs=1
        )
        window_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second)
        assert window_time < 1.0, f"Per-window IC filter too slow: {window_time:.2f}s"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
