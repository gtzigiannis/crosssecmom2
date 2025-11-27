"""
Unit tests for alpha_models.py

Tests ElasticNetWindowModel:
- Uses stored scaling params (not recomputed at t0)
- Works with subset of features
- Handles NaNs correctly
- Never touches labels at or after t0

Run with: pytest test_alpha_models.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

from alpha_models import ElasticNetWindowModel, MomentumRankModel


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def synthetic_panel():
    """
    Create synthetic panel for testing.
    
    Returns:
        panel: DataFrame with MultiIndex (Date, Ticker)
        Features: feat_0, feat_1, feat_2
        Target: FwdRet_21
    """
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = [f'ETF{i:02d}' for i in range(20)]
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['Date', 'Ticker'])
    
    # Features with varying signal strength
    n_samples = len(index)
    feat_0 = np.random.randn(n_samples)  # Strong signal
    feat_1 = np.random.randn(n_samples)  # Weak signal
    feat_2 = np.random.randn(n_samples)  # Noise
    
    # Target: FwdRet_21 = 0.3 * feat_0 + 0.1 * feat_1 + noise
    target = 0.3 * feat_0 + 0.1 * feat_1 + 0.2 * np.random.randn(n_samples)
    
    panel = pd.DataFrame({
        'feat_0': feat_0,
        'feat_1': feat_1,
        'feat_2': feat_2,
        'FwdRet_21': target
    }, index=index)
    
    return panel


@pytest.fixture
def trained_model(synthetic_panel):
    """
    Create a trained ElasticNetWindowModel from synthetic data.
    
    Trains on first 50 dates, tests on remaining dates.
    Uses proper subset design: refit on selected features only.
    """
    panel = synthetic_panel
    dates = panel.index.get_level_values('Date').unique()
    
    # Training window: first 50 dates
    train_dates = dates[:50]
    train_mask = panel.index.get_level_values('Date').isin(train_dates)
    train_data = panel[train_mask]
    
    # Features and target
    feature_cols = ['feat_0', 'feat_1', 'feat_2']
    X_train = train_data[feature_cols]
    y_train = train_data['FwdRet_21']
    
    # Compute scaling params for all features (median/MAD)
    medians = X_train.median().to_dict()
    mads = X_train.apply(lambda col: np.median(np.abs(col - col.median()))).to_dict()
    
    # Standardize training data
    X_train_scaled = X_train.copy()
    for col in feature_cols:
        X_train_scaled[col] = (X_train[col] - medians[col]) / mads[col]
    
    # Fit ElasticNet to determine feature selection
    model_full = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    model_full.fit(X_train_scaled.values, y_train.values)
    
    # Select features with non-zero coefficients
    non_zero_idx = np.abs(model_full.coef_) > 1e-4
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if non_zero_idx[i]]
    
    # CRITICAL: Refit ElasticNet on SELECTED features only
    X_train_selected = X_train_scaled[selected_features]
    model_refit = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    model_refit.fit(X_train_selected.values, y_train.values)
    
    # Extract scaling params ONLY for selected features
    selected_scaling_params = {
        'median': {feat: medians[feat] for feat in selected_features},
        'mad': {feat: mads[feat] for feat in selected_features}
    }
    
    # Create ElasticNetWindowModel with refit model and subset params
    wrapped_model = ElasticNetWindowModel(
        selected_features=selected_features,
        scaling_params=selected_scaling_params,  # Only selected features
        fitted_model=model_refit,  # Refit on subset
        binning_params=None
    )
    
    return wrapped_model, panel, train_dates


# ============================================================================
# Tests for ElasticNetWindowModel
# ============================================================================

class TestElasticNetWindowModel:
    """Tests for ElasticNetWindowModel."""
    
    def test_uses_stored_scaling_params(self, trained_model):
        """Verify that model uses stored median/MAD, not recomputed stats."""
        model, panel, train_dates = trained_model
        
        # Pick a test date (not in training)
        test_dates = panel.index.get_level_values('Date').unique()
        t0 = test_dates[60]  # Date 60 (after training window ends at date 50)
        
        # Get ONLY selected features (model operates on subset)
        X_t0 = panel.loc[t0, model.selected_features]
        
        # Compute what the standardized values SHOULD be using stored params
        expected_scaled = X_t0.copy()
        for col in model.selected_features:
            median_train = model.scaling_params['median'][col]
            mad_train = model.scaling_params['mad'][col]
            expected_scaled[col] = (X_t0[col] - median_train) / mad_train
        
        # Score using model
        scores = model.score_at_date(panel, t0, None, None)
        
        # Manually predict using expected_scaled to verify consistency
        expected_scores = model.fitted_model.predict(expected_scaled.values)
        
        # Check that scores match expected (within numerical precision)
        np.testing.assert_allclose(
            scores.values, 
            expected_scores, 
            rtol=1e-5,
            err_msg="Model did not use stored scaling params correctly"
        )
    
    def test_handles_subset_of_features(self, synthetic_panel):
        """Verify that model works when only a subset of features is selected."""
        panel = synthetic_panel
        dates = panel.index.get_level_values('Date').unique()
        
        # Create model with only feat_0 selected (not feat_1 or feat_2)
        selected_features = ['feat_0']
        
        scaling_params = {
            'median': {'feat_0': 0.0},
            'mad': {'feat_0': 1.0}
        }
        
        # Simple model: score = feat_0 (identity)
        class SimpleModel:
            def predict(self, X):
                return X[:, 0]  # Return first column
        
        model = ElasticNetWindowModel(
            selected_features=selected_features,
            scaling_params=scaling_params,
            fitted_model=SimpleModel(),
            binning_params=None
        )
        
        # Score at a test date
        t0 = dates[50]
        scores = model.score_at_date(panel, t0, None, None)
        
        # Verify scores were computed
        assert len(scores) == 20  # 20 tickers
        assert not scores.isna().all(), "All scores are NaN"
        
        # Verify scores are approximately equal to standardized feat_0
        X_t0 = panel.loc[t0, 'feat_0']
        expected = (X_t0 - 0.0) / 1.0
        np.testing.assert_allclose(scores.values, expected.values, rtol=1e-5)
    
    def test_handles_nan_features(self, synthetic_panel):
        """Verify that model handles NaN features correctly."""
        panel = synthetic_panel.copy()
        dates = panel.index.get_level_values('Date').unique()
        t0 = dates[50]
        
        # Inject NaNs into feat_0 for some tickers at t0
        tickers = panel.loc[t0].index
        nan_tickers = tickers[:5]  # First 5 tickers get NaN
        
        for ticker in nan_tickers:
            panel.loc[(t0, ticker), 'feat_0'] = np.nan
        
        # Create simple model
        selected_features = ['feat_0']
        scaling_params = {
            'median': {'feat_0': 0.0},
            'mad': {'feat_0': 1.0}
        }
        
        class SimpleModel:
            def predict(self, X):
                return X[:, 0]
        
        model = ElasticNetWindowModel(
            selected_features=selected_features,
            scaling_params=scaling_params,
            fitted_model=SimpleModel(),
            binning_params=None
        )
        
        # Score at t0
        scores = model.score_at_date(panel, t0, None, None)
        
        # Verify NaN tickers have NaN scores
        for ticker in nan_tickers:
            assert np.isnan(scores.loc[ticker]), f"Ticker {ticker} should have NaN score"
        
        # Verify non-NaN tickers have valid scores
        valid_tickers = [t for t in tickers if t not in nan_tickers]
        for ticker in valid_tickers:
            assert not np.isnan(scores.loc[ticker]), f"Ticker {ticker} should have valid score"
    
    def test_no_label_leakage(self, trained_model):
        """Verify that score_at_date never uses labels at or after t0."""
        model, panel, train_dates = trained_model
        
        # Pick a test date
        test_dates = panel.index.get_level_values('Date').unique()
        t0 = test_dates[60]
        
        # Score at t0
        scores_original = model.score_at_date(panel, t0, None, None)
        
        # Corrupt all labels at and after t0
        panel_corrupted = panel.copy()
        future_mask = panel_corrupted.index.get_level_values('Date') >= t0
        panel_corrupted.loc[future_mask, 'FwdRet_21'] = np.nan
        
        # Score again with corrupted labels
        scores_corrupted = model.score_at_date(panel_corrupted, t0, None, None)
        
        # Scores should be IDENTICAL (model shouldn't use labels at t0)
        np.testing.assert_array_equal(
            scores_original.values,
            scores_corrupted.values,
            err_msg="Model used labels at or after t0 (label leakage detected)"
        )
    
    def test_score_at_date_api(self, trained_model):
        """Verify that score_at_date returns clean pd.Series."""
        model, panel, train_dates = trained_model
        
        test_dates = panel.index.get_level_values('Date').unique()
        t0 = test_dates[60]
        
        scores = model.score_at_date(panel, t0, None, None)
        
        # Check return type
        assert isinstance(scores, pd.Series), "score_at_date should return pd.Series"
        
        # Check index
        expected_index = panel.loc[t0].index
        assert scores.index.equals(expected_index), "Scores index should match t0 cross-section"
        
        # Check name
        assert scores.name == 'score', "Series should be named 'score'"


class TestMomentumRankModel:
    """Tests for MomentumRankModel baseline."""
    
    def test_momentum_rank_baseline(self, synthetic_panel):
        """Verify MomentumRankModel ranks a feature correctly."""
        panel = synthetic_panel
        dates = panel.index.get_level_values('Date').unique()
        t0 = dates[50]
        
        model = MomentumRankModel(feature='feat_0')
        scores = model.score_at_date(panel, t0, None, None)
        
        # Verify scores are ranks (0 to 1)
        assert scores.min() >= 0.0, "Ranks should be >= 0"
        assert scores.max() <= 1.0, "Ranks should be <= 1"
        
        # Verify ranking is correct (higher feat_0 -> higher rank)
        X_t0 = panel.loc[t0, 'feat_0']
        expected_ranks = X_t0.rank(pct=True, method='average')
        
        pd.testing.assert_series_equal(scores, expected_ranks, check_names=False)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
