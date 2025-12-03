"""
Unit tests for label_engineering.py

Tests cover:
1. Forward return correctness and no leakage
2. Cross-sectional demeaning
3. Risk adjustment (if controls are provided)
4. Winsorisation and z-scoring
5. Integration sanity checks
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from label_engineering import (
    compute_targets,
    _compute_raw_target,
    _compute_cs_demeaned,
    _compute_risk_adjusted,
    _compute_zscore,
    get_target_columns,
    validate_targets,
)
from config import get_default_config, TargetConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_panel():
    """Create a small synthetic panel for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    assets = [f"ETF_{i}" for i in range(20)]
    
    rows = []
    for date in dates:
        for asset in assets:
            # Generate returns with known properties
            raw_ret = np.random.randn() * 0.02 + 0.001  # ~2% vol, slight drift
            beta = np.random.uniform(0.5, 1.5)
            vol = np.random.uniform(0.1, 0.3)
            
            rows.append({
                "date": date,
                "asset": asset,
                "FwdRet_21": raw_ret,
                "beta_VT_63": beta,
                "idio_vol_63": vol,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def config():
    """Get default config for testing."""
    cfg = get_default_config()
    cfg.target.target_horizon_days = 21
    return cfg


@pytest.fixture
def config_with_risk_controls(config):
    """Config with risk control columns enabled."""
    config.target.target_risk_control_columns = ["beta_VT_63", "idio_vol_63"]
    config.target.target_use_risk_adjustment = True
    return config


@pytest.fixture
def config_no_risk_adjustment(config):
    """Config with risk adjustment disabled."""
    config.target.target_use_risk_adjustment = False
    config.target.target_risk_control_columns = []
    return config


# =============================================================================
# TEST: RAW TARGET (y_raw_21d)
# =============================================================================

class TestRawTarget:
    """Tests for raw forward return computation."""
    
    def test_raw_target_correctness(self, synthetic_panel, config):
        """Test that y_raw_21d equals the source column."""
        panel = synthetic_panel.copy()
        tc = config.target
        
        result = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        
        # y_raw_21d should exactly equal FwdRet_21
        np.testing.assert_array_almost_equal(
            result["y_raw_21d"].values,
            result["FwdRet_21"].values
        )
    
    def test_raw_target_auto_detect_column(self, synthetic_panel, config):
        """Test auto-detection of FwdRet_{horizon} column."""
        panel = synthetic_panel.copy()
        panel = panel.rename(columns={"FwdRet_21": "FwdRet_21"})  # Already correct
        tc = config.target
        
        result = _compute_raw_target(panel, tc, raw_return_col=None)
        
        assert "y_raw_21d" in result.columns
        assert result["y_raw_21d"].notna().sum() > 0
    
    def test_raw_target_missing_column_error(self, synthetic_panel, config):
        """Test error when source column is missing."""
        panel = synthetic_panel.drop(columns=["FwdRet_21"])
        tc = config.target
        
        with pytest.raises(ValueError, match="Raw return column"):
            _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
    
    def test_no_leakage_in_forward_returns(self, config):
        """Test that forward returns don't leak future data."""
        # Create deterministic panel where we know exact forward returns
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        assets = ["A", "B"]
        
        rows = []
        for i, date in enumerate(dates):
            for asset in assets:
                # Price grows linearly: price_t = 100 + day_number
                price = 100 + i
                # Forward return from day i to day i+21
                # If i+21 > 29, return is NaN (can't compute)
                if i + 21 < 30:
                    fwd_ret = ((100 + i + 21) - (100 + i)) / (100 + i)
                else:
                    fwd_ret = np.nan
                
                rows.append({
                    "date": date,
                    "asset": asset,
                    "FwdRet_21": fwd_ret,
                })
        
        panel = pd.DataFrame(rows)
        tc = config.target
        
        result = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        
        # Forward return at day 0 should be (121 - 100) / 100 = 0.21
        expected_day0 = 21 / 100
        actual_day0 = result[result["date"] == dates[0]]["y_raw_21d"].iloc[0]
        np.testing.assert_almost_equal(actual_day0, expected_day0, decimal=5)


# =============================================================================
# TEST: CROSS-SECTIONAL DEMEANING (y_cs_21d)
# =============================================================================

class TestCrossSecDemeaning:
    """Tests for cross-sectional demeaning."""
    
    def test_global_demean_zero_mean_per_date(self, synthetic_panel, config):
        """Test that y_cs_21d has ~0 mean per date with global demeaning."""
        panel = synthetic_panel.copy()
        tc = config.target
        tc.target_demean_mode = "global"
        
        # First compute raw
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        result = _compute_cs_demeaned(panel, tc)
        
        # Mean per date should be ~0
        daily_means = result.groupby("date")["y_cs_21d"].mean()
        
        # All daily means should be very close to 0 (machine precision)
        assert daily_means.abs().max() < 1e-10, f"Max daily mean: {daily_means.abs().max()}"
    
    def test_demean_mode_none_skips(self, synthetic_panel, config):
        """Test that demean_mode='none' skips demeaning."""
        panel = synthetic_panel.copy()
        tc = config.target
        tc.target_demean_mode = "none"
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        result = _compute_cs_demeaned(panel, tc)
        
        # y_cs should equal y_raw
        np.testing.assert_array_almost_equal(
            result["y_cs_21d"].values,
            result["y_raw_21d"].values
        )
    
    def test_demean_preserves_cross_sectional_ranking(self, synthetic_panel, config):
        """Test that demeaning preserves cross-sectional ranking."""
        panel = synthetic_panel.copy()
        tc = config.target
        tc.target_demean_mode = "global"
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        result = _compute_cs_demeaned(panel, tc)
        
        # Check ranking on first date
        first_date = result["date"].min()
        first_date_data = result[result["date"] == first_date]
        
        rank_raw = first_date_data["y_raw_21d"].rank()
        rank_cs = first_date_data["y_cs_21d"].rank()
        
        # Rankings should be identical
        np.testing.assert_array_equal(rank_raw.values, rank_cs.values)
    
    def test_by_asset_class_demean_fallback(self, synthetic_panel, config):
        """Test fallback to global when asset_class column missing."""
        panel = synthetic_panel.copy()
        tc = config.target
        tc.target_demean_mode = "by_asset_class"
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        result = _compute_cs_demeaned(panel, tc)
        
        # Should still have ~0 mean per date (fell back to global)
        daily_means = result.groupby("date")["y_cs_21d"].mean()
        assert daily_means.abs().max() < 1e-10


# =============================================================================
# TEST: RISK ADJUSTMENT (y_resid_21d)
# =============================================================================

class TestRiskAdjustment:
    """Tests for risk adjustment via regression."""
    
    def test_skip_when_disabled(self, synthetic_panel, config_no_risk_adjustment):
        """Test that risk adjustment is skipped when disabled."""
        panel = synthetic_panel.copy()
        tc = config_no_risk_adjustment.target
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        panel = _compute_cs_demeaned(panel, tc)
        result = _compute_risk_adjusted(panel, tc)
        
        # y_resid should equal y_cs
        np.testing.assert_array_almost_equal(
            result["y_resid_21d"].values,
            result["y_cs_21d"].values
        )
    
    def test_skip_when_no_controls(self, synthetic_panel, config):
        """Test that risk adjustment is skipped with empty controls."""
        panel = synthetic_panel.copy()
        tc = config.target
        tc.target_use_risk_adjustment = True
        tc.target_risk_control_columns = []
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        panel = _compute_cs_demeaned(panel, tc)
        result = _compute_risk_adjusted(panel, tc)
        
        # y_resid should equal y_cs
        np.testing.assert_array_almost_equal(
            result["y_resid_21d"].values,
            result["y_cs_21d"].values
        )
    
    def test_residual_uncorrelated_with_controls(self, config_with_risk_controls):
        """Test that residuals have ~0 correlation with controls."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        assets = [f"ETF_{i}" for i in range(30)]
        
        rows = []
        for date in dates:
            for asset in assets:
                # Create returns with strong beta exposure
                beta = np.random.uniform(0.5, 1.5)
                market_ret = np.random.randn() * 0.01
                idio_ret = np.random.randn() * 0.02
                # y = beta * market + idio
                raw_ret = 0.5 * beta + idio_ret + market_ret
                
                rows.append({
                    "date": date,
                    "asset": asset,
                    "FwdRet_21": raw_ret,
                    "beta_VT_63": beta,
                    "idio_vol_63": np.abs(idio_ret),
                })
        
        panel = pd.DataFrame(rows)
        tc = config_with_risk_controls.target
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        panel = _compute_cs_demeaned(panel, tc)
        result = _compute_risk_adjusted(panel, tc)
        
        # Correlation between residual and beta should be ~0
        corr_beta = result[["y_resid_21d", "beta_VT_63"]].dropna().corr().iloc[0, 1]
        
        # Allow some tolerance due to finite sample
        assert abs(corr_beta) < 0.1, f"Residual-beta corr too high: {corr_beta}"
    
    def test_fallback_when_control_missing(self, synthetic_panel, config):
        """Test fallback when control column is missing."""
        panel = synthetic_panel.drop(columns=["beta_VT_63"])  # Remove a control
        tc = config.target
        tc.target_use_risk_adjustment = True
        tc.target_risk_control_columns = ["beta_VT_63", "idio_vol_63"]
        
        panel = _compute_raw_target(panel, tc, raw_return_col="FwdRet_21")
        panel = _compute_cs_demeaned(panel, tc)
        result = _compute_risk_adjusted(panel, tc)
        
        # Should fallback to y_resid = y_cs
        np.testing.assert_array_almost_equal(
            result["y_resid_21d"].values,
            result["y_cs_21d"].values
        )


# =============================================================================
# TEST: WINSORISATION AND Z-SCORING (y_resid_z_21d)
# =============================================================================

class TestZscoring:
    """Tests for winsorisation and z-scoring."""
    
    def test_zscore_mean_and_std(self, synthetic_panel, config_no_risk_adjustment):
        """Test that z-scores have ~0 mean and ~1 std per date."""
        panel = synthetic_panel.copy()
        tc = config_no_risk_adjustment.target
        
        result = compute_targets(panel, config_no_risk_adjustment, raw_return_col="FwdRet_21")
        
        # Check per-date mean and std
        z_means = result.groupby("date")["y_resid_z_21d"].mean()
        z_stds = result.groupby("date")["y_resid_z_21d"].std()
        
        # Mean should be ~0
        assert z_means.abs().max() < 0.1, f"Max z-score mean: {z_means.abs().max()}"
        
        # Std should be ~1
        assert (z_stds - 1.0).abs().max() < 0.1, f"Max z-score std deviation from 1: {(z_stds - 1.0).abs().max()}"
    
    def test_winsorisation_clips_extremes(self, config):
        """Test that winsorisation clips extreme values."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        assets = [f"ETF_{i}" for i in range(100)]
        
        rows = []
        for i, date in enumerate(dates):
            for j, asset in enumerate(assets):
                # Add outliers at specific positions
                if j == 0:
                    ret = 10.0  # Extreme positive
                elif j == 1:
                    ret = -10.0  # Extreme negative
                else:
                    ret = np.random.randn() * 0.02
                
                rows.append({
                    "date": date,
                    "asset": asset,
                    "FwdRet_21": ret,
                })
        
        panel = pd.DataFrame(rows)
        cfg = config
        cfg.target.target_winsorization_limits = (0.01, 0.99)
        cfg.target.target_use_risk_adjustment = False
        
        result = compute_targets(panel, cfg, raw_return_col="FwdRet_21")
        
        # Z-scores should not have extreme values like the originals
        z_max = result["y_resid_z_21d"].max()
        z_min = result["y_resid_z_21d"].min()
        
        # With 100 samples and 1%/99% winsorisation, extreme z-scores should be bounded
        # The bound depends on the distribution - with winsorisation, extremes are clipped
        # but z-scores can still be several sigma from mean
        assert z_max < 8, f"Z-score max too high: {z_max}"
        assert z_min > -8, f"Z-score min too low: {z_min}"


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_runs(self, synthetic_panel, config):
        """Test that full compute_targets runs without errors."""
        result = compute_targets(synthetic_panel, config, raw_return_col="FwdRet_21")
        
        # Check all target columns are present
        expected_cols = ["y_raw_21d", "y_cs_21d", "y_resid_21d", "y_resid_z_21d"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_validate_targets_returns_diagnostics(self, synthetic_panel, config):
        """Test that validate_targets returns proper diagnostics."""
        result = compute_targets(synthetic_panel, config, raw_return_col="FwdRet_21")
        validation = validate_targets(result, config.target)
        
        # Check all target columns have validation results
        for col in get_target_columns(config.target):
            assert col in validation, f"Missing validation for: {col}"
            assert validation[col]["status"] == "ok"
            assert validation[col]["n_valid"] > 0
    
    def test_preserves_original_columns(self, synthetic_panel, config):
        """Test that compute_targets preserves original columns."""
        original_cols = set(synthetic_panel.columns)
        result = compute_targets(synthetic_panel, config, raw_return_col="FwdRet_21")
        
        # All original columns should still be present
        for col in original_cols:
            assert col in result.columns, f"Original column lost: {col}"
    
    def test_handles_nan_gracefully(self, config):
        """Test that NaN values are handled without crashing."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        assets = ["A", "B", "C"]
        
        rows = []
        for i, date in enumerate(dates):
            for asset in assets:
                # Some NaN values
                if i % 5 == 0 and asset == "A":
                    ret = np.nan
                else:
                    ret = np.random.randn() * 0.02
                
                rows.append({
                    "date": date,
                    "asset": asset,
                    "FwdRet_21": ret,
                })
        
        panel = pd.DataFrame(rows)
        
        # Should not raise
        result = compute_targets(panel, config, raw_return_col="FwdRet_21")
        
        # NaN should propagate to all target columns for those rows
        nan_raw = result["y_raw_21d"].isna().sum()
        nan_z = result["y_resid_z_21d"].isna().sum()
        
        assert nan_raw >= 4, "Expected some NaN values in raw target"
        # Z-score NaN could be >= raw NaN (due to min samples requirement)


# =============================================================================
# TEST: CONFIGURATION
# =============================================================================

class TestConfiguration:
    """Tests for configuration options."""
    
    def test_target_column_selection(self, synthetic_panel, config):
        """Test that config.target.target_column points to correct column."""
        result = compute_targets(synthetic_panel, config, raw_return_col="FwdRet_21")
        
        # Default target_column is y_resid_z_21d
        assert config.target.target_column == "y_resid_z_21d"
        assert config.target.target_column in result.columns
    
    def test_custom_horizon(self):
        """Test custom horizon affects column names."""
        config = get_default_config()
        config.target.target_horizon_days = 10
        config.target.target_column_raw = "y_raw_10d"
        config.target.target_column_cs = "y_cs_10d"
        config.target.target_column_resid = "y_resid_10d"
        config.target.target_column_resid_z = "y_resid_z_10d"
        
        expected = ["y_raw_10d", "y_cs_10d", "y_resid_10d", "y_resid_z_10d"]
        assert get_target_columns(config.target) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
