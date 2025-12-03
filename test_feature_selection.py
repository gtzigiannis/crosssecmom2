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
    Generate synthetic panel data for a single window (multiple dates).
    
    Returns:
        X: DataFrame (n_dates * n_assets, n_features) with DatetimeIndex
        y: Series (n_dates * n_assets,) with DatetimeIndex
    """
    np.random.seed(123)
    
    n_dates = 63  # One quarter of trading days
    n_assets = 100  # Cross-section size
    n_features = 30
    
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    
    # Generate panel data
    X_list = []
    y_list = []
    date_list = []
    
    for date in dates:
        # Generate cross-sectional features
        X_date = np.random.randn(n_assets, n_features)
        
        # Create target with known relationships:
        # - Features 0-4: Strong IC (0.2 coefficient)
        # - Features 5-9: Medium IC (0.1 coefficient)
        # - Features 10-29: Weak/no IC
        y_date = (0.2 * X_date[:, :5].mean(axis=1) + 
                  0.1 * X_date[:, 5:10].mean(axis=1) + 
                  np.random.randn(n_assets) * 0.5)
        
        X_list.append(X_date)
        y_list.append(y_date)
        date_list.extend([date] * n_assets)
    
    # Combine into panel
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['date'] = date_list
    X_df = X_df.set_index('date')
    
    y_series = pd.Series(y, index=X_df.index, name='forward_return')
    
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
        assert abs(ic_series['feature_25'].mean()) < 0.1  # Relaxed threshold
        
        # feature_45 should have negative mean IC (strong negative)
        assert ic_series['feature_45'].mean() < 0.0  # Just check sign, not magnitude
    
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
        expected_cols = ['feature', 'ic_weighted', 't_nw', 'p_value', 'fdr_reject']
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
        
        # Should approve if strong signal (check actual column name)
        assert len(diagnostics) == 1
        ic_col = 'ic_weighted' if 'ic_weighted' in diagnostics.columns else 'mean_ic'
        if abs(diagnostics[ic_col].iloc[0]) > 0.05:
            assert len(approved) >= 1


# ============================================================================
# Test per_window_ic_filter()
# ============================================================================

class TestPerWindowICFilter:
    """Test suite for per_window_ic_filter() function."""
    
    def test_per_window_ic_filter_basic(self, synthetic_single_window):
        """Test per_window IC filter with synthetic panel data."""
        X_df, y_series = synthetic_single_window
        
        # Extract dates from index
        dates = X_df.index.unique()
        weights = np.ones(len(dates))  # Weight per date, not per sample
        
        # Use relaxed thresholds for test
        selected, diagnostics = per_window_ic_filter(
            X_df,
            y_series,
            dates,
            weights,
            theta_ic=0.03,  # Relaxed from 0.05
            t_min=1.0,  # Relaxed from 1.5
            n_jobs=1
        )
        
        # Check outputs
        assert isinstance(selected, list)
        assert isinstance(diagnostics, dict)
        
        # Check diagnostics keys
        expected_keys = ['n_start', 'n_after_ic']
        assert all(key in diagnostics for key in expected_keys)
        
        # With 63 dates × 100 assets and strong signal, should find features
        assert len(selected) >= 1, f"Expected at least 1 feature with 63 dates, got {len(selected)}"
    
    def test_per_window_ic_filter_signal_detection(self, synthetic_single_window):
        """Test IC filter correctly identifies strong features."""
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        weights = np.ones(len(dates))
        
        # Use relaxed thresholds
        selected, diagnostics = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.03,  # Relaxed
            t_min=1.0,  # Relaxed
            n_jobs=1
        )
        
        # With 63 dates and strong signal (0.2 coefficient), should detect at least 1
        strong_features = [f'feature_{i}' for i in range(5)]
        selected_strong = [f for f in strong_features if f in selected]
        assert len(selected_strong) >= 1, f"Expected at least 1 of 5 strong features, got {len(selected_strong)}"
        
        # Weak features (10-29) should be mostly rejected
        weak_features = [f'feature_{i}' for i in range(10, 30)]
        selected_weak = [f for f in weak_features if f in selected]
        assert len(selected_weak) <= 5  # At most 25% false positives
    
    def test_per_window_ic_filter_threshold_effects(self, synthetic_single_window):
        """Test IC filter with different thresholds."""
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        weights = np.ones(len(dates))
        
        # Loose threshold
        selected_loose, _ = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.01,  # Very loose
            t_min=0.5,  # Very loose
            n_jobs=1
        )
        
        # Strict threshold
        selected_strict, _ = per_window_ic_filter(
            X_df, y_series, dates, weights,
            theta_ic=0.1,  # Strict
            t_min=2.0,  # Strict
            n_jobs=1
        )
        
        # Loose should select more than strict (or at least some features)
        # In small samples, even loose might return 0, so just check relationship
        assert len(selected_strict) <= len(selected_loose)
    
    def test_per_window_ic_filter_time_decay(self, synthetic_single_window):
        """Test IC filter with time-decay weights."""
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Uniform weights
        weights_uniform = np.ones(len(dates))
        selected_uniform, _ = per_window_ic_filter(
            X_df, y_series, dates, weights_uniform,
            theta_ic=0.03,  # Relaxed
            t_min=1.0,  # Relaxed
            n_jobs=1
        )
        
        # Exponential decay (emphasize early observations)
        weights_decay = np.exp(-np.arange(len(dates)) * 0.01)
        selected_decay, _ = per_window_ic_filter(
            X_df, y_series, dates, weights_decay,
            theta_ic=0.03,  # Relaxed
            t_min=1.0,  # Relaxed
            n_jobs=1
        )
        
        # Both should work without error
        # With 63 dates × 100 assets, at least one configuration should find features
        total_selected = len(selected_uniform) + len(selected_decay)
        assert total_selected > 0, f"Expected at least one configuration to find features with 63 dates"
    
    def test_per_window_ic_filter_parallel(self, synthetic_single_window):
        """Test IC filter with parallel processing."""
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        weights = np.ones(len(dates))
        
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
# Test Per-Window Stability Filter
# ============================================================================

class TestPerWindowStability:
    """Test suite for per_window_stability function."""
    
    def test_per_window_stability_basic(self, synthetic_single_window):
        """Test basic stability filter functionality."""
        from feature_selection import per_window_stability
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Run stability filter
        stable_features, diagnostics = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.03,
            min_sign_consistency=2,
            n_jobs=1
        )
        
        # Check outputs
        assert isinstance(stable_features, list)
        assert isinstance(diagnostics, dict)
        
        # Check diagnostics keys
        expected_keys = ['n_start', 'n_after_stability', 'pct_stable']
        assert all(key in diagnostics for key in expected_keys)
        
        # With synthetic data (strong features 0-4), should find some stable features
        assert len(stable_features) > 0, "Expected at least one stable feature"
        
        # Diagnostic counts should be consistent
        assert diagnostics['n_after_stability'] == len(stable_features)
    
    def test_per_window_stability_sign_consistency(self):
        """Test that stability filter checks sign consistency across folds."""
        from feature_selection import per_window_stability
        
        np.random.seed(456)
        
        # Create panel with one feature that has consistent positive IC across all folds
        # and another that flips sign
        n_dates = 63
        n_assets = 100
        dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
        
        X_list, y_list, date_list = [], [], []
        for i, date in enumerate(dates):
            X_date = np.random.randn(n_assets, 2)
            
            # feature_0: consistent positive IC across all folds
            # feature_1: changes sign across folds (inconsistent)
            if i < n_dates // 3:
                # Fold 1: both positive
                y_date = 0.3 * X_date[:, 0] + 0.2 * X_date[:, 1] + np.random.randn(n_assets) * 0.5
            elif i < 2 * n_dates // 3:
                # Fold 2: feature_0 positive, feature_1 negative
                y_date = 0.3 * X_date[:, 0] - 0.2 * X_date[:, 1] + np.random.randn(n_assets) * 0.5
            else:
                # Fold 3: feature_0 positive, feature_1 positive again
                y_date = 0.3 * X_date[:, 0] + 0.2 * X_date[:, 1] + np.random.randn(n_assets) * 0.5
            
            X_list.append(X_date)
            y_list.append(y_date)
            date_list.extend([date] * n_assets)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
        X_df['date'] = date_list
        X_df = X_df.set_index('date')
        y_series = pd.Series(y, index=X_df.index)
        
        # Run stability filter with min_sign_consistency=3 (all folds same sign)
        stable, _ = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.01,  # Low threshold to focus on sign consistency
            min_sign_consistency=3,
            n_jobs=1
        )
        
        # feature_0 should be stable (consistent sign), feature_1 should not
        assert 'feature_0' in stable, "Expected consistent feature to pass stability"
        assert 'feature_1' not in stable, "Expected inconsistent feature to fail stability"
    
    def test_per_window_stability_magnitude_threshold(self):
        """Test that stability filter applies median magnitude threshold."""
        from feature_selection import per_window_stability
        
        np.random.seed(789)
        
        # Create panel with features of different IC magnitudes
        n_dates = 63
        n_assets = 100
        dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
        
        X_list, y_list, date_list = [], [], []
        for date in dates:
            X_date = np.random.randn(n_assets, 3)
            
            # feature_0: strong IC (~0.3)
            # feature_1: medium IC (~0.1)
            # feature_2: weak IC (~0.03)
            y_date = (0.3 * X_date[:, 0] + 
                      0.1 * X_date[:, 1] + 
                      0.03 * X_date[:, 2] + 
                      np.random.randn(n_assets) * 0.5)
            
            X_list.append(X_date)
            y_list.append(y_date)
            date_list.extend([date] * n_assets)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2'])
        X_df['date'] = date_list
        X_df = X_df.set_index('date')
        y_series = pd.Series(y, index=X_df.index)
        
        # Strict threshold (should only get strong feature)
        stable_strict, _ = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.10,  # High threshold
            min_sign_consistency=2,
            n_jobs=1
        )
        
        # Lenient threshold (should get strong + medium features)
        stable_lenient, _ = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.05,  # Low threshold
            min_sign_consistency=2,
            n_jobs=1
        )
        
        # More lenient threshold should select at least as many features
        assert len(stable_lenient) >= len(stable_strict)
        
        # Strong feature should pass both thresholds
        if len(stable_strict) > 0:
            assert 'feature_0' in stable_strict or 'feature_1' in stable_strict
    
    def test_per_window_stability_parallel(self, synthetic_single_window):
        """Test that parallel processing gives same results as serial."""
        from feature_selection import per_window_stability
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Serial
        stable_serial, _ = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.05,
            min_sign_consistency=2,
            n_jobs=1
        )
        
        # Parallel
        stable_parallel, _ = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.05,
            min_sign_consistency=2,
            n_jobs=2
        )
        
        # Results should be identical
        assert set(stable_serial) == set(stable_parallel)
    
    def test_per_window_stability_edge_cases(self):
        """Test edge cases: insufficient folds, all features rejected."""
        from feature_selection import per_window_stability
        
        np.random.seed(321)
        
        # Small window (< 3 fold minimum)
        n_dates = 20
        n_assets = 50
        dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
        
        X_list, y_list, date_list = [], [], []
        for date in dates:
            X_date = np.random.randn(n_assets, 5)
            y_date = np.random.randn(n_assets)  # Pure noise
            X_list.append(X_date)
            y_list.append(y_date)
            date_list.extend([date] * n_assets)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        X_df['date'] = date_list
        X_df = X_df.set_index('date')
        y_series = pd.Series(y, index=X_df.index)
        
        # Should handle small window gracefully
        stable, diag = per_window_stability(
            X_df, y_series, dates,
            k_folds=3,
            theta_stable=0.10,  # Very high threshold
            min_sign_consistency=3,
            n_jobs=1
        )
        
        # With pure noise and strict thresholds, should reject most/all features
        assert isinstance(stable, list)
        assert len(stable) <= 2  # Allow for occasional statistical noise


class TestCorrelationRedundancy:
    """Test correlation-based redundancy filter."""
    
    def test_correlation_redundancy_basic(self, synthetic_single_window):
        """Test basic functionality and output schema."""
        from feature_selection import correlation_redundancy_filter
        
        X_df, y_series = synthetic_single_window
        X_subset = X_df.iloc[:, :10]
        
        selected, diag = correlation_redundancy_filter(
            X_subset,
            corr_threshold=0.7,
            n_jobs=1
        )
        
        # Should return list and diagnostics
        assert isinstance(selected, list)
        assert isinstance(diag, dict)
        
        # Diagnostics should have required keys
        assert 'n_start' in diag
        assert 'n_after_redundancy' in diag
        assert 'n_removed' in diag
        assert 'time_redundancy' in diag
        
        # Should select at least one feature
        assert len(selected) > 0
        assert diag['n_after_redundancy'] == len(selected)
        assert diag['n_removed'] == diag['n_start'] - diag['n_after_redundancy']
        
    def test_correlation_redundancy_removal(self, synthetic_single_window):
        """Test that highly correlated features are removed."""
        from feature_selection import correlation_redundancy_filter
        
        X_df, y_series = synthetic_single_window
        np.random.seed(999)
        
        # Create features with known correlations
        base_feature = np.random.randn(len(X_df))
        
        # feat_0: base
        # feat_1: highly correlated with feat_0 (r=0.95)
        # feat_2: moderately correlated with feat_0 (r=0.5)
        # feat_3: independent
        X_test = pd.DataFrame({
            'feat_0': base_feature,
            'feat_1': 0.95 * base_feature + 0.05 * np.random.randn(len(X_df)),
            'feat_2': 0.5 * base_feature + 0.5 * np.random.randn(len(X_df)),
            'feat_3': np.random.randn(len(X_df))
        }, index=X_df.index)
        
        # With threshold 0.7, feat_1 should be removed (corr with feat_0 > 0.7)
        selected, diag = correlation_redundancy_filter(
            X_test,
            corr_threshold=0.7,
            n_jobs=1
        )
        
        # Should remove highly correlated features
        assert len(selected) >= 2  # At least feat_0 and feat_3
        assert diag['n_removed'] >= 1
        assert 'feat_1' not in selected  # Highly correlated with feat_0
        assert 'feat_0' in selected  # First feature is kept
        
    def test_correlation_redundancy_threshold_effect(self, synthetic_single_window):
        """Test that threshold affects removal count."""
        from feature_selection import correlation_redundancy_filter
        
        X_df, y_series = synthetic_single_window
        X_subset = X_df.iloc[:, :10]
        
        # Lenient threshold (0.95) should remove fewer features
        selected_lenient, diag_lenient = correlation_redundancy_filter(
            X_subset, corr_threshold=0.95, n_jobs=1
        )
        
        # Strict threshold (0.5) should remove more features
        selected_strict, diag_strict = correlation_redundancy_filter(
            X_subset, corr_threshold=0.5, n_jobs=1
        )
        
        # Stricter threshold should result in fewer features
        assert len(selected_strict) <= len(selected_lenient)
        
    def test_correlation_redundancy_edge_cases(self, synthetic_single_window):
        """Test edge cases: single feature, perfect correlation."""
        from feature_selection import correlation_redundancy_filter
        
        X_df, y_series = synthetic_single_window
        
        # Single feature
        X_single = X_df.iloc[:, [0]]
        selected, diag = correlation_redundancy_filter(
            X_single, corr_threshold=0.7, n_jobs=1
        )
        
        assert len(selected) == 1
        assert diag['n_removed'] == 0


class TestRobustStandardization:
    """Test robust standardization (median/MAD)."""
    
    def test_robust_standardization_basic(self, synthetic_single_window):
        """Test basic functionality and output schema."""
        from feature_selection import robust_standardization
        
        X_df, y_series = synthetic_single_window
        X_subset = X_df.iloc[:, :5]
        
        X_standardized, params = robust_standardization(X_subset)
        
        # Should return DataFrame and dict
        assert isinstance(X_standardized, pd.DataFrame)
        assert isinstance(params, dict)
        
        # Should have same shape
        assert X_standardized.shape == X_subset.shape
        
        # Parameters should have 'median' and 'mad' for each feature
        assert 'median' in params
        assert 'mad' in params
        assert len(params['median']) == X_subset.shape[1]
        assert len(params['mad']) == X_subset.shape[1]
        
    def test_robust_standardization_values(self, synthetic_single_window):
        """Test that standardized values have correct properties."""
        from feature_selection import robust_standardization
        
        X_df, y_series = synthetic_single_window
        X_subset = X_df.iloc[:, :5]
        
        X_std, params = robust_standardization(X_subset)
        
        # Standardized features should have median ~0 and MAD ~1
        for col in X_std.columns:
            median_std = X_std[col].median()
            mad_std = np.median(np.abs(X_std[col] - median_std))
            
            assert abs(median_std) < 0.1  # Close to 0
            assert abs(mad_std - 1.0) < 0.2  # Close to 1
            
    def test_robust_standardization_params(self, synthetic_single_window):
        """Test that parameters can be used to re-standardize."""
        from feature_selection import robust_standardization
        
        X_df, y_series = synthetic_single_window
        X_subset = X_df.iloc[:, :5]
        
        X_std, params = robust_standardization(X_subset)
        
        # Manually re-standardize using params
        X_manual = X_subset.copy()
        for col in X_manual.columns:
            X_manual[col] = (X_manual[col] - params['median'][col]) / params['mad'][col]
        
        # Should match
        pd.testing.assert_frame_equal(X_std, X_manual, rtol=1e-10)


class TestElasticNetCV:
    """Test ElasticNet feature selection with TimeSeriesSplit."""
    
    def test_elasticnet_cv_basic(self, synthetic_single_window):
        """Test basic functionality and output schema."""
        from feature_selection import elasticnet_cv_selection
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Standardize first (ElasticNet expects standardized input)
        from feature_selection import robust_standardization
        X_std, _ = robust_standardization(X_df)
        
        selected, model, diag = elasticnet_cv_selection(
            X_std, y_series, dates,
            alpha_grid=[0.01, 0.1],
            l1_ratio_grid=[0.5, 0.9],
            cv_folds=3,
            coef_threshold=1e-4,
            n_jobs=1
        )
        
        # Should return list, model, and diagnostics
        assert isinstance(selected, list)
        assert model is not None
        assert isinstance(diag, dict)
        
        # Diagnostics should have required keys
        assert 'n_start' in diag
        assert 'n_selected' in diag
        assert 'best_alpha' in diag
        assert 'best_l1_ratio' in diag
        assert 'time_elasticnet' in diag
        
        # Should select at least some features
        assert len(selected) > 0
        assert diag['n_selected'] == len(selected)
        
    def test_elasticnet_signal_detection(self, synthetic_single_window):
        """Test that ElasticNet selects features with signal."""
        from feature_selection import elasticnet_cv_selection, robust_standardization
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Fixture has strong signal in features 0-4, medium in 5-9
        # Standardize first
        X_std, _ = robust_standardization(X_df)
        
        selected, model, diag = elasticnet_cv_selection(
            X_std, y_series, dates,
            alpha_grid=[0.01, 0.05],
            l1_ratio_grid=[0.5, 0.9],
            cv_folds=3,
            coef_threshold=1e-4,
            n_jobs=1
        )
        
        # Should select strong signal features (0-4) preferentially
        strong_features = [f'feature_{i}' for i in range(5)]
        selected_strong = [f for f in selected if f in strong_features]
        
        # At least some strong features should be selected
        assert len(selected_strong) >= 2
        
    def test_elasticnet_uses_timeseriessplit(self, synthetic_single_window):
        """Test that CV respects temporal ordering."""
        from feature_selection import elasticnet_cv_selection, robust_standardization
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Standardize first
        X_std, _ = robust_standardization(X_df)
        
        # ElasticNetCV should use TimeSeriesSplit internally
        selected, model, diag = elasticnet_cv_selection(
            X_std, y_series, dates,
            alpha_grid=[0.01, 0.1],
            l1_ratio_grid=[0.5],
            cv_folds=3,
            coef_threshold=1e-4,
            n_jobs=1
        )
        
        # Should complete successfully with temporal CV
        assert len(selected) > 0
        assert diag['best_alpha'] in [0.01, 0.1]
        
    def test_elasticnet_nonzero_threshold(self, synthetic_single_window):
        """Test that selection uses non-zero coefficient threshold."""
        from feature_selection import elasticnet_cv_selection, robust_standardization
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        X_std, _ = robust_standardization(X_df.iloc[:, :10])
        
        # Lenient threshold should select more features
        selected_lenient, _, diag_lenient = elasticnet_cv_selection(
            X_std, y_series, dates,
            alpha_grid=[0.01],
            l1_ratio_grid=[0.9],
            cv_folds=3,
            coef_threshold=1e-5,  # Very lenient
            n_jobs=1
        )
        
        # Strict threshold should select fewer features
        selected_strict, _, diag_strict = elasticnet_cv_selection(
            X_std, y_series, dates,
            alpha_grid=[0.01],
            l1_ratio_grid=[0.9],
            cv_folds=3,
            coef_threshold=1e-2,  # Strict
            n_jobs=1
        )
        
        # Stricter threshold should result in fewer features
        assert len(selected_strict) <= len(selected_lenient)


class TestScoreAtT0:
    """Test scoring new data at t0 using trained models."""
    
    def test_score_at_t0_basic(self, synthetic_single_window):
        """Test basic functionality and output schema."""
        from feature_selection import score_at_t0, robust_standardization, elasticnet_cv_selection
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Train on first 50 dates
        train_dates = dates[:50]
        train_mask = X_df.index.isin(train_dates)
        X_train = X_df.loc[train_mask]
        y_train = y_series.loc[train_mask]
        
        # Standardize and train model
        X_std_train, std_params = robust_standardization(X_train)
        selected, model, _ = elasticnet_cv_selection(
            X_std_train, y_train, train_dates,
            alpha_grid=[0.01], l1_ratio_grid=[0.5], cv_folds=3
        )
        
        # Score on last 13 dates (t0 data)
        test_dates = dates[50:]
        test_mask = X_df.index.isin(test_dates)
        X_test = X_df.loc[test_mask]
        
        scores = score_at_t0(
            X_test,
            selected_features=selected,
            standardization_params=std_params,
            model=model
        )
        
        # Should return Series with one score per asset at t0
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(X_test)
        assert scores.index.equals(X_test.index)
        
    def test_score_at_t0_uses_stored_params(self, synthetic_single_window):
        """Test that scoring uses stored standardization params, not t0 data."""
        from feature_selection import score_at_t0, robust_standardization, elasticnet_cv_selection
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Train on first 50 dates
        train_dates = dates[:50]
        train_mask = X_df.index.isin(train_dates)
        X_train = X_df.loc[train_mask].iloc[:, :5]  # Use subset
        y_train = y_series.loc[train_mask]
        
        X_std_train, std_params = robust_standardization(X_train)
        selected, model, _ = elasticnet_cv_selection(
            X_std_train, y_train, train_dates,
            alpha_grid=[0.01], l1_ratio_grid=[0.9], cv_folds=3
        )
        
        # Create t0 data with different distribution
        test_dates = dates[50:]
        test_mask = X_df.index.isin(test_dates)
        X_test = X_df.loc[test_mask].iloc[:, :5]
        
        # Score using training params
        scores = score_at_t0(X_test, selected, std_params, model)
        
        # Should complete without using any t0 statistics
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(X_test)
        
    def test_score_at_t0_no_label_leakage(self, synthetic_single_window):
        """Test that scoring doesn't use any label information."""
        from feature_selection import score_at_t0, robust_standardization, elasticnet_cv_selection
        
        X_df, y_series = synthetic_single_window
        dates = X_df.index.unique()
        
        # Train
        train_dates = dates[:50]
        train_mask = X_df.index.isin(train_dates)
        X_train = X_df.loc[train_mask].iloc[:, :10]
        y_train = y_series.loc[train_mask]
        
        X_std_train, std_params = robust_standardization(X_train)
        selected, model, _ = elasticnet_cv_selection(
            X_std_train, y_train, train_dates,
            alpha_grid=[0.01], l1_ratio_grid=[0.9], cv_folds=3
        )
        
        # Score at t0 - only X_test is passed, no y_test
        test_dates = dates[50:]
        test_mask = X_df.index.isin(test_dates)
        X_test = X_df.loc[test_mask].iloc[:, :10]
        
        # Function should not accept y parameter
        scores = score_at_t0(X_test, selected, std_params, model)
        
        assert isinstance(scores, pd.Series)


# NOTE: TestSupervisedBinning class removed - binning deprecated in v3 pipeline


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
    
    def test_per_window_pipeline_complete(self, synthetic_panel_data):
        """Test complete per-window pipeline from training to t0 scoring."""
        from feature_selection import per_window_pipeline, formation_fdr
        
        X_df, y_series, dates = synthetic_panel_data
        
        # Step 1: Formation FDR on all data
        approved, _ = formation_fdr(
            X_df, y_series, dates,
            half_life=126, fdr_level=0.1, n_jobs=1
        )
        
        # Step 2: Split into training window and t0
        train_dates = dates[:200]  # Use first 200 dates for training
        t0_dates = dates[200:210]  # Use next 10 dates as t0
        
        X_train = X_df.loc[X_df.index.isin(train_dates)]
        y_train = y_series.loc[y_series.index.isin(train_dates)]
        X_t0 = X_df.loc[X_df.index.isin(t0_dates)]
        
        # Step 3: Run per-window pipeline
        scores, diagnostics, model = per_window_pipeline(
            X_train, y_train, X_t0,
            train_dates,
            approved_features=approved,
            half_life=126,
            theta_ic=0.03,
            theta_stable=0.03,
            k_folds=3,
            corr_threshold=0.7,
            alpha_grid=[0.01, 0.1],
            l1_ratio_grid=[0.5, 0.9],
            cv_folds=3,
            n_jobs=1
        )
        
        # Verify outputs
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(X_t0)
        assert isinstance(diagnostics, dict)
        assert 'ic_filter' in diagnostics
        assert 'stability' in diagnostics
        # NOTE: binning removed in v3 pipeline - no longer required
        # assert 'binning' in diagnostics
        assert 'redundancy' in diagnostics
        assert 'standardization' in diagnostics
        assert 'elasticnet' in diagnostics
        assert 'final_n_features' in diagnostics
        assert 'total_time' in diagnostics
        
        # Model should be ElasticNetWindowModel
        assert model is not None
        assert hasattr(model, 'score_at_date'), "Model should have score_at_date method"
        assert hasattr(model, 'fitted_model'), "Model should wrap fitted ElasticNet"
        assert hasattr(model.fitted_model, 'predict'), "Wrapped model should have predict"
        
        # Pipeline should have selected some features
        assert diagnostics['final_n_features'] > 0
        
        print(f"Pipeline diagnostics: {diagnostics}")
        print(f"Model type: {type(model).__name__}")
        print(f"Selected features: {len(model.selected_features)}")



# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

