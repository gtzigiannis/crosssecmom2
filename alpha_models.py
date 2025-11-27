"""
Alpha Models Module (Refactored)
==================================
Provides model wrappers that store trained parameters and apply them at t0.

This module has been refactored to separate "models only" from feature selection:
- feature_selection.py: Contains the canonical feature selection pipeline
- alpha_models.py: Contains model wrappers that store and apply trained parameters

Model Interface:
    All models implement: score_at_date(panel, t0, universe_metadata, config) -> pd.Series

Available Models:
    - MomentumRankModel: Simple baseline (rank a single momentum feature)
    - ElasticNetWindowModel: Linear model with stored scaling params and ElasticNet coefficients
    
DEPRECATED FUNCTIONS:
    - train_alpha_model(): Use feature_selection.per_window_pipeline() instead
    - All Phase 2 feature selection helpers (fit_supervised_bins, compute_cv_ic, etc.)
      are superseded by feature_selection.py
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings


class AlphaModel:
    """
    Base class for alpha models.
    
    All alpha models must implement:
    - score_at_date(panel, t0, universe_metadata, config) -> pd.Series
    """
    
    def score_at_date(
        self,
        panel: pd.DataFrame,
        t0: pd.Timestamp,
        universe_metadata: pd.DataFrame,
        config
    ) -> pd.Series:
        """
        Generate cross-sectional scores at date t0.
        
        Parameters
        ----------
        panel : pd.DataFrame
            Full panel with MultiIndex (Date, Ticker)
        t0 : pd.Timestamp
            Scoring date
        universe_metadata : pd.DataFrame
            ETF metadata (families, clusters, caps)
        config : ResearchConfig
            Configuration object
            
        Returns
        -------
        pd.Series
            Scores indexed by ticker (higher = more attractive)
        """
        raise NotImplementedError


class MomentumRankModel(AlphaModel):
    """
    Simple baseline: score = rank of a single momentum feature.
    
    This serves as a minimal working example and baseline for more complex models.
    """
    
    def __init__(self, feature: str = 'Close%-63'):
        """
        Parameters
        ----------
        feature : str
            Feature name to rank (e.g., 'Close%-63' for 3-month momentum)
        """
        self.feature = feature
    
    def score_at_date(
        self,
        panel: pd.DataFrame,
        t0: pd.Timestamp,
        universe_metadata: pd.DataFrame,
        config
    ) -> pd.Series:
        """Score = percentile rank of momentum feature."""
        try:
            cross_section = panel.loc[t0]
        except KeyError:
            # Date not in panel
            return pd.Series(dtype=float)
        
        if self.feature not in cross_section.columns:
            warnings.warn(f"Feature {self.feature} not found in panel at {t0}")
            return pd.Series(dtype=float)
        
        scores = cross_section[self.feature].rank(pct=True, method='average')
        return scores


class ElasticNetWindowModel(AlphaModel):
    """
    Per-window linear model storing selected features, scaling params, and fitted ElasticNet.
    
    This model is returned by feature_selection.per_window_pipeline() and contains:
    - selected_features: List of feature names selected by the pipeline
    - scaling_params: Dict with 'median' and 'mad' ONLY for selected features
    - binning_params: Optional dict with binning boundaries for supervised-binned features
    - fitted_model: ElasticNet model REFIT on selected features only
    
    Scoring at t0:
    - Extracts ONLY selected features at t0
    - Applies stored binning (if any)
    - Standardizes using stored median/MAD from training
    - Predicts using refit ElasticNet (trained on selected features)
    - Returns pd.Series of scores indexed by ticker
    
    CRITICAL: All transformations use ONLY parameters learned during training.
    No label information or statistics from t0 are used.
    The model operates on the SUBSET of selected features only.
    """
    
    def __init__(
        self,
        selected_features: List[str],
        scaling_params: Dict,
        fitted_model: object,
        binning_params: Optional[Dict] = None
    ):
        """
        Parameters
        ----------
        selected_features : list
            Feature names selected by ElasticNet (non-zero coefficients)
        scaling_params : dict
            {'median': {feat: value}, 'mad': {feat: value}}
            Scaling parameters for SELECTED features only (computed on training window)
        fitted_model : object
            ElasticNet model REFIT on selected features only
        binning_params : dict, optional
            {feature_name: bin_boundaries} for supervised-binned features
            If None, no binning is applied
        """
        self.selected_features = selected_features
        self.scaling_params = scaling_params
        self.fitted_model = fitted_model
        self.binning_params = binning_params or {}
    
    def _apply_binning(self, cross_section: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored binning boundaries to create binned features.
        
        Uses digitize to assign each value to a bin based on training-window boundaries.
        """
        cs = cross_section.copy()
        
        for feat, boundaries in self.binning_params.items():
            if feat not in cs.columns:
                continue
            
            # Digitize: bin i if boundaries[i-1] <= x < boundaries[i]
            binned = np.digitize(cs[feat].values, boundaries, right=False)
            cs[f'{feat}_binned'] = binned.astype('float32')
        
        return cs
    
    def _apply_scaling(self, cross_section: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored robust standardization (median/MAD) from training window.
        
        CRITICAL: Uses training-window statistics, not t0 statistics.
        """
        cs = cross_section.copy()
        
        for feat in cross_section.columns:
            if feat in self.scaling_params['median']:
                median_train = self.scaling_params['median'][feat]
                mad_train = self.scaling_params['mad'][feat]
                cs[feat] = (cross_section[feat] - median_train) / mad_train
        
        return cs
    
    def score_at_date(
        self,
        panel: pd.DataFrame,
        t0: pd.Timestamp,
        universe_metadata: pd.DataFrame,
        config
    ) -> pd.Series:
        """
        Compute scores at t0 using stored model and parameters.
        
        Steps:
        1. Extract t0 cross-section
        2. Apply stored binning (if any)
        3. Extract ONLY selected features (model was refit on these)
        4. Apply stored standardization (median/MAD from training)
        5. Predict using refit ElasticNet
        6. Return scores as pd.Series
        
        CRITICAL: Uses only selected_features (the model was refit on this subset).
        scaling_params contains ONLY these features.
        
        Returns
        -------
        pd.Series
            Scores indexed by ticker (NaN for tickers with missing features)
        """
        try:
            cross_section = panel.loc[t0].copy()
        except KeyError:
            return pd.Series(dtype=float)
        
        # Apply binning if configured
        if self.binning_params:
            cross_section = self._apply_binning(cross_section)
        
        # Extract ONLY selected features (model was refit on these)
        available_features = [f for f in self.selected_features if f in cross_section.columns]
        
        if len(available_features) == 0:
            warnings.warn(f"No selected features available at {t0}")
            return pd.Series(dtype=float)
        
        if len(available_features) < len(self.selected_features):
            warnings.warn(f"Only {len(available_features)}/{len(self.selected_features)} selected features available at {t0}")
        
        # Extract feature matrix in training order
        X_t0 = cross_section[self.selected_features].copy()
        
        # Apply stored standardization (only for selected features)
        X_t0_scaled = self._apply_scaling(X_t0)
        
        # Check for NaNs (assets with missing features)
        has_nan = X_t0_scaled.isna().any(axis=1)
        
        # Predict scores
        scores = pd.Series(np.nan, index=cross_section.index, name='score')
        
        if not has_nan.all():
            X_clean = X_t0_scaled[~has_nan].values
            scores[~has_nan] = self.fitted_model.predict(X_clean)
        
        return scores


# ============================================================================
# DEPRECATED: Old Phase 2 Feature Selection Code
# ============================================================================
# 
# The functions below (fit_supervised_bins, compute_ic, compute_cv_ic, etc.)
# are DEPRECATED and superseded by feature_selection.py.
#
# DO NOT USE these functions in new code. They are kept temporarily for 
# backward compatibility with existing walk-forward code that hasn't been
# migrated yet.
#
# For new feature selection, use:
#   from feature_selection import per_window_pipeline
#
# ============================================================================


class SupervisedBinnedModel(AlphaModel):
    """
    DEPRECATED: Use ElasticNetWindowModel instead.
    
    This class is kept for backward compatibility with existing walk-forward code.
    New code should use ElasticNetWindowModel returned by feature_selection.per_window_pipeline().
    
    Alpha model with supervised binning and feature selection inside training window.
    
    Training phase:
    1. Fit decision trees to find optimal bin boundaries for selected features
    2. Create binned features using these boundaries
    3. Compute IC for all features (raw + binned)
    4. Select top features by absolute IC
    5. Fit final model on selected features (for now, equal-weight combination)
    
    Scoring phase:
    1. Apply stored bin boundaries to create binned features at t0
    2. Compute score using stored feature weights
    """
    
    def __init__(
        self,
        binning_dict: Dict[str, np.ndarray],
        selected_features: List[str],
        feature_weights: Optional[pd.Series] = None,
        feature_ics: Optional[pd.Series] = None
    ):
        """
        Parameters
        ----------
        binning_dict : dict
            {feature_name: bin_boundaries} for each binned feature
            bin_boundaries is array of cutpoints defining bins
        selected_features : list
            List of feature names (raw or binned) selected for the model
        feature_weights : pd.Series, optional
            Weights for each selected feature (if None, equal weight)
        feature_ics : pd.Series, optional
            Information coefficients for features (for diagnostics)
        """
        self.binning_dict = binning_dict
        self.selected_features = selected_features
        self.feature_weights = feature_weights
        self.feature_ics = feature_ics
    
    def _apply_binning(self, cross_section: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored bin boundaries to create binned features.
        
        Parameters
        ----------
        cross_section : pd.DataFrame
            Cross-section data at one date
            
        Returns
        -------
        pd.DataFrame
            Cross-section with added binned features
        """
        cs = cross_section.copy()
        
        for feat, boundaries in self.binning_dict.items():
            if feat not in cs.columns:
                continue
            
            # Digitize: bin i if boundaries[i-1] <= x < boundaries[i]
            # boundaries includes -inf and +inf at ends
            binned = np.digitize(cs[feat].values, boundaries, right=False)
            cs[f'{feat}_Bin'] = binned.astype('float32')
        
        return cs
    
    def score_at_date(
        self,
        panel: pd.DataFrame,
        t0: pd.Timestamp,
        universe_metadata: pd.DataFrame,
        config
    ) -> pd.Series:
        """
        Compute scores at t0 using selected features and stored weights.
        """
        try:
            cross_section = panel.loc[t0].copy()
        except KeyError:
            return pd.Series(dtype=float)
        
        # Apply binning
        cross_section = self._apply_binning(cross_section)
        
        # Check which selected features are available
        available_features = [f for f in self.selected_features if f in cross_section.columns]
        
        if len(available_features) == 0:
            warnings.warn(f"No selected features available at {t0}")
            return pd.Series(dtype=float)
        
        # Compute scores
        if self.feature_weights is not None:
            # Weighted combination with IC sign handling
            # CRITICAL: IC sign determines ranking direction
            #   Positive IC: high feature -> rank high -> good (normal)
            #   Negative IC: high feature -> rank low -> bad (flip by using negative weight)
            scores = pd.Series(0.0, index=cross_section.index)
            for feat in available_features:
                # Always rank ascending (0 to 1)
                feat_rank = cross_section[feat].rank(pct=True, method='average')
                # Weight is SIGNED IC - negative weights flip the contribution
                weight = self.feature_weights.get(feat, 1.0)
                scores += weight * feat_rank
        else:
            # Equal-weight combination
            scores = cross_section[available_features].rank(pct=True, method='average').mean(axis=1)
        
        return scores


def fit_supervised_bins(
    feature_values: pd.Series,
    target_values: pd.Series,
    max_depth: int = 3,
    min_samples_leaf: int = 100,
    n_bins: int = 10,
    **kwargs
) -> np.ndarray:
    """
    Fit supervised bins using decision tree regressor.
    
    Fits a shallow decision tree to find cutpoints that partition the feature
    into bins with different expected target values.
    
    Parameters
    ----------
    feature_values : pd.Series
        Feature values in training window
    target_values : pd.Series
        Target values (FwdRet_H) aligned with features
    max_depth : int
        Maximum tree depth
    min_samples_leaf : int
        Minimum samples per leaf node
    n_bins : int
        Target number of bins (max_leaf_nodes)
        
    Returns
    -------
    np.ndarray
        Bin boundaries including -inf and +inf at ends
        Length = n_actual_bins + 1
    """
    # Remove NaN
    valid_mask = feature_values.notna() & target_values.notna()
    X = feature_values[valid_mask].values.reshape(-1, 1)
    y = target_values[valid_mask].values
    
    if len(X) < min_samples_leaf * 2:
        # Insufficient data, return single bin
        return np.array([-np.inf, np.inf])
    
    # Get random_state from kwargs if provided
    random_state = kwargs.get('random_state', None)
    
    # Fit decision tree
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=n_bins,
        random_state=random_state
    )
    
    try:
        tree.fit(X, y)
    except Exception as e:
        warnings.warn(f"Tree fitting failed: {e}, using single bin")
        return np.array([-np.inf, np.inf])
    
    # Extract thresholds from tree
    thresholds = []
    
    def extract_thresholds(node_id, tree_obj):
        """Recursively extract split thresholds."""
        if tree_obj.feature[node_id] != -2:  # Not a leaf
            thresholds.append(tree_obj.threshold[node_id])
            # Recurse on children
            extract_thresholds(tree_obj.children_left[node_id], tree_obj)
            extract_thresholds(tree_obj.children_right[node_id], tree_obj)
    
    extract_thresholds(0, tree.tree_)
    
    if len(thresholds) == 0:
        # Tree didn't split (all same target value?)
        return np.array([-np.inf, np.inf])
    
    # Sort thresholds and add -inf, +inf at ends
    boundaries = np.sort(thresholds)
    boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
    
    return boundaries


def compute_ic(
    feature_values: pd.Series,
    target_values: pd.Series,
    method: str = 'spearman'
) -> float:
    """
    Compute Information Coefficient (rank correlation between feature and target).
    
    Parameters
    ----------
    feature_values : pd.Series
        Feature values
    target_values : pd.Series
        Target values (FwdRet_H)
    method : str
        'spearman' or 'pearson'
        
    Returns
    -------
    float
        IC value (NaN if insufficient data)
    """
    valid_mask = feature_values.notna() & target_values.notna()
    
    if valid_mask.sum() < 10:
        return np.nan
    
    feat = feature_values[valid_mask]
    tgt = target_values[valid_mask]
    
    # Check if feature is constant (prevents correlation warnings)
    if feat.nunique() <= 1:
        return np.nan
    
    # Check if target is constant (shouldn't happen, but defensive)
    if tgt.nunique() <= 1:
        return np.nan
    
    if method == 'spearman':
        ic, _ = spearmanr(feat, tgt)
    else:
        ic = np.corrcoef(feat, tgt)[0, 1]
    
    return ic


def compute_cv_ic(
    feature_values: pd.Series,
    target_values: pd.Series,
    config,
    n_splits: int = 5,
    is_already_binned: bool = False
) -> float:
    """
    Compute cross-validated IC to prevent overfitting in feature selection.
    
    For binned features:
    1. Split training data into K folds
    2. For each fold:
       - Fit bins on K-1 folds
       - Compute IC on held-out fold using those bins
    3. Return mean IC across folds
    
    For raw features:
    - Simply compute IC on each fold directly
    - Return mean IC across folds
    
    Parameters
    ----------
    feature_values : pd.Series
        Raw feature values (will be binned if necessary)
    target_values : pd.Series
        Target values (FwdRet_H) aligned with features
    config : ResearchConfig
        Configuration for binning parameters
    n_splits : int
        Number of CV folds (default 5)
    is_already_binned : bool
        If True, feature is already binned, skip re-binning in CV
        
    Returns
    -------
    float
        Mean IC across folds (out-of-sample)
    """
    # Remove NaN
    valid_mask = feature_values.notna() & target_values.notna()
    X = feature_values[valid_mask].values
    y = target_values[valid_mask].values
    idx = feature_values[valid_mask].index
    
    if len(X) < n_splits * 10:
        # Insufficient data for CV, fall back to simple IC
        return compute_ic(
            pd.Series(X, index=idx),
            pd.Series(y, index=idx),
            method='spearman'
        )
    
    # If feature is already binned, just compute IC directly without re-binning
    if is_already_binned:
        kf = KFold(n_splits=n_splits, shuffle=False)
        fold_ics = []
        for train_idx, test_idx in kf.split(X):
            X_test = X[test_idx]
            y_test = y[test_idx]
            ic = compute_ic(
                pd.Series(X_test),
                pd.Series(y_test),
                method='spearman'
            )
            fold_ics.append(ic)
        return np.mean(fold_ics)
    
    # K-Fold cross-validation with binning
    kf = KFold(n_splits=n_splits, shuffle=False)  # No shuffle to preserve time order
    fold_ics = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit bins on training fold
        boundaries = fit_supervised_bins(
            pd.Series(X_train),
            pd.Series(y_train),
            max_depth=config.features.bin_max_depth,
            min_samples_leaf=config.features.bin_min_samples_leaf,
            n_bins=config.features.n_bins,
            random_state=config.features.random_state
        )
        
        # Apply bins to test fold
        X_test_binned = np.digitize(X_test, boundaries, right=False)
        
        # Compute IC on test fold
        if len(X_test) > 2:
            ic = compute_ic(
                pd.Series(X_test_binned),
                pd.Series(y_test),
                method='spearman'
            )
            fold_ics.append(ic)
    
    # Return mean IC across folds
    if len(fold_ics) > 0:
        return np.mean(fold_ics)
    else:
        return 0.0


def compute_cv_ic_with_folds(
    feature_values: pd.Series,
    target_values: pd.Series,
    config,
    n_splits: int = 5,
    is_already_binned: bool = False
) -> Dict[str, any]:
    """
    PHASE 0: Enhanced CV-IC computation that returns fold-level ICs for sign consistency check.
    
    Returns both mean IC and individual fold ICs to enable sign consistency filtering.
    This helps identify features with unstable IC direction across folds.
    
    Parameters
    ----------
    feature_values : pd.Series
        Raw feature values
    target_values : pd.Series
        Target values (FwdRet_H)
    config : ResearchConfig
        Configuration
    n_splits : int
        Number of CV folds
    is_already_binned : bool
        Whether feature is pre-binned
        
    Returns
    -------
    dict
        {
            'mean_ic': float,           # Mean IC across folds
            'fold_ics': list,           # Individual fold ICs
            'sign_consistency': float   # Fraction of folds with same sign as median
        }
    """
    # Remove NaN
    valid_mask = feature_values.notna() & target_values.notna()
    X = feature_values[valid_mask].values
    y = target_values[valid_mask].values
    idx = feature_values[valid_mask].index
    
    if len(X) < n_splits * 10:
        # Insufficient data for CV
        ic = compute_ic(pd.Series(X, index=idx), pd.Series(y, index=idx), method='spearman')
        return {
            'mean_ic': ic,
            'fold_ics': [ic],
            'sign_consistency': 1.0
        }
    
    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_ics = []
    
    if is_already_binned:
        # Feature already binned, just compute IC per fold
        for train_idx, test_idx in kf.split(X):
            X_test = X[test_idx]
            y_test = y[test_idx]
            if len(X_test) > 2:
                ic = compute_ic(pd.Series(X_test), pd.Series(y_test), method='spearman')
                fold_ics.append(ic)
    else:
        # Fit bins on each training fold, evaluate on test fold
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit bins on training fold
            boundaries = fit_supervised_bins(
                pd.Series(X_train),
                pd.Series(y_train),
                max_depth=config.features.bin_max_depth,
                min_samples_leaf=config.features.bin_min_samples_leaf,
                n_bins=config.features.n_bins,
                random_state=config.features.random_state
            )
            
            # Apply bins to test fold
            X_test_binned = np.digitize(X_test, boundaries, right=False)
            
            # Compute IC on test fold
            if len(X_test) > 2:
                ic = compute_ic(pd.Series(X_test_binned), pd.Series(y_test), method='spearman')
                fold_ics.append(ic)
    
    if len(fold_ics) == 0:
        return {'mean_ic': 0.0, 'fold_ics': [], 'sign_consistency': 0.0}
    
    # Compute sign consistency
    median_ic = np.median(fold_ics)
    if median_ic == 0:
        sign_consistency = 0.0
    else:
        same_sign_count = sum(np.sign(ic) == np.sign(median_ic) for ic in fold_ics)
        sign_consistency = same_sign_count / len(fold_ics)
    
    return {
        'mean_ic': np.mean(fold_ics),
        'fold_ics': fold_ics,
        'sign_consistency': sign_consistency
    }


def compute_cv_mi(
    feature: pd.Series,
    target: pd.Series,
    config,
    n_splits: int = 5
) -> Dict:
    """
    Compute mutual information across cross-validation folds.
    
    Returns:
    --------
    dict with keys:
        - 'mean_mi': Average MI across folds
        - 'fold_mis': List of MI values from each fold
        - 'mi_positivity_rate': Fraction of folds with MI > 0
    """
    feature_values = feature.values.reshape(-1, 1)
    target_values = target.values
    
    # Remove NaN rows
    valid_mask = ~(np.isnan(feature_values).any(axis=1) | np.isnan(target_values))
    feature_clean = feature_values[valid_mask]
    target_clean = target_values[valid_mask]
    
    if len(feature_clean) < 50:
        return {
            'mean_mi': 0.0,
            'fold_mis': [],
            'mi_positivity_rate': 0.0
        }
    
    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_mis = []
    
    for train_idx, test_idx in kf.split(feature_clean):
        X_test = feature_clean[test_idx]
        y_test = target_clean[test_idx]
        
        if len(X_test) < 20:
            continue
        
        # Compute MI on test fold
        mi = mutual_info_regression(
            X_test, y_test, 
            discrete_features=False,
            random_state=42,
            n_neighbors=min(3, len(X_test) // 2)
        )[0]
        
        fold_mis.append(mi)
    
    if len(fold_mis) == 0:
        return {
            'mean_mi': 0.0,
            'fold_mis': [],
            'mi_positivity_rate': 0.0
        }
    
    # Calculate positivity rate
    positive_count = sum(mi > 0 for mi in fold_mis)
    mi_positivity_rate = positive_count / len(fold_mis)
    
    return {
        'mean_mi': np.mean(fold_mis),
        'fold_mis': fold_mis,
        'mi_positivity_rate': mi_positivity_rate
    }


def filter_multicollinear_features(
    train_data: pd.DataFrame,
    candidate_features: List[str],
    mi_values: Dict[str, float],
    corr_threshold: float = 0.75
) -> List[str]:
    """
    Remove redundant features using pairwise Spearman correlation.
    
    Among correlated feature groups (correlation > threshold),
    keep the feature with highest MI.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    candidate_features : List[str]
        Features to filter
    mi_values : Dict[str, float]
        MI value for each feature
    corr_threshold : float
        Correlation threshold (default 0.75)
        
    Returns:
    --------
    List[str]
        Filtered feature list with multicollinearity removed
    """
    if len(candidate_features) <= 1:
        return candidate_features
    
    # Build correlation matrix
    feature_matrix = train_data[candidate_features].dropna()
    
    if len(feature_matrix) < 50:
        return candidate_features
    
    corr_matrix = feature_matrix.corr(method='spearman').abs()
    
    # Identify correlated groups
    to_drop = set()
    
    for i, feat_i in enumerate(candidate_features):
        if feat_i in to_drop:
            continue
        
        for j in range(i + 1, len(candidate_features)):
            feat_j = candidate_features[j]
            
            if feat_j in to_drop:
                continue
            
            # Check correlation
            corr_val = corr_matrix.loc[feat_i, feat_j]
            
            if corr_val > corr_threshold:
                # Drop the one with lower MI
                mi_i = mi_values.get(feat_i, 0.0)
                mi_j = mi_values.get(feat_j, 0.0)
                
                if mi_j > mi_i:
                    to_drop.add(feat_i)
                    break  # feat_i is dropped, no need to check more
                else:
                    to_drop.add(feat_j)
    
    # Also drop features with MI = 0
    for feat in candidate_features:
        if mi_values.get(feat, 0.0) <= 0:
            to_drop.add(feat)
    
    filtered = [f for f in candidate_features if f not in to_drop]
    
    print(f"[filter_multicol] Dropped {len(to_drop)} features due to multicollinearity or zero MI")
    
    return filtered


def elastic_net_feature_selection(
    train_data: pd.DataFrame,
    candidate_features: List[str],
    target_col: str,
    cv_folds: int = 5,
    l1_ratio_values: List[float] = None,
    max_features: int = 15
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select features using Elastic Net with cross-validated regularization.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    candidate_features : List[str]
        Features to consider
    target_col : str
        Target column name
    cv_folds : int
        Number of CV folds (default 5)
    l1_ratio_values : List[float]
        L1 ratios to test (default [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0])
    max_features : int
        Maximum features to return (default 15)
        
    Returns:
    --------
    Tuple[List[str], Dict[str, float]]
        - Selected feature names
        - Feature coefficients (non-zero)
    """
    if l1_ratio_values is None:
        l1_ratio_values = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    
    # Prepare data
    X = train_data[candidate_features].copy()
    y = train_data[target_col].copy()
    
    # Drop rows with NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 100:
        print(f"[elastic_net] Insufficient data ({len(X)} rows), skipping")
        return candidate_features[:max_features], {}
    
    # Standardize features (important for regularization)
    X_mean = X.mean()
    X_std = X.std()
    X_std[X_std == 0] = 1.0  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    
    # Fit Elastic Net with CV
    print(f"[elastic_net] Fitting ElasticNetCV on {len(candidate_features)} features...")
    
    enet = ElasticNetCV(
        l1_ratio=l1_ratio_values,
        cv=cv_folds,
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    )
    
    enet.fit(X_scaled, y)
    
    # Extract non-zero coefficients
    coefficients = pd.Series(enet.coef_, index=candidate_features)
    non_zero_features = coefficients[coefficients != 0].abs().sort_values(ascending=False)
    
    print(f"[elastic_net] Optimal l1_ratio: {enet.l1_ratio_:.3f}, alpha: {enet.alpha_:.6f}")
    print(f"[elastic_net] Selected {len(non_zero_features)} features with non-zero coefficients")
    
    # Limit to max_features
    selected_features = non_zero_features.head(max_features).index.tolist()
    feature_coefs = non_zero_features.head(max_features).to_dict()
    
    return selected_features, feature_coefs


def train_alpha_model(
    panel: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    t_train_start: pd.Timestamp,
    t_train_end: pd.Timestamp,
    config,
    model_type: str = 'supervised_binned'
) -> AlphaModel:
    """
    DEPRECATED: Use feature_selection.per_window_pipeline() instead.
    
    This function is kept for backward compatibility with existing walk-forward code
    that hasn't been migrated to the new pipeline yet.
    
    NEW CODE SHOULD USE:
        from feature_selection import per_window_pipeline
        scores, diagnostics, model = per_window_pipeline(
            X_train, y_train, X_t0, dates_train, approved_features, ...
        )
    
    Train an alpha model on the specified training window.
    
    This function orchestrates:
    1. Extracting training window data
    2. Supervised binning on selected features
    3. Supervised feature selection based on IC
    4. Model fitting
    
    CRITICAL: All operations use ONLY data from [t_train_start, t_train_end]
    
    Parameters
    ----------
    panel : pd.DataFrame
        Full panel with MultiIndex (Date, Ticker)
    universe_metadata : pd.DataFrame
        ETF metadata
    t_train_start : pd.Timestamp
        Training window start
    t_train_end : pd.Timestamp
        Training window end (inclusive)
    config : ResearchConfig
        Configuration object
    model_type : str
        'momentum_rank' or 'supervised_binned'
        
    Returns
    -------
    tuple
        (model, selected_features, ic_series)
        - model: AlphaModel object
        - selected_features: List of feature names used
        - ic_series: pd.Series of IC values for all features
    """
    if model_type == 'momentum_rank':
        # Simple baseline: just return momentum rank model with empty diagnostics
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    # ===== Supervised binned model =====
    
    # Extract training window
    dates = panel.index.get_level_values('Date')
    train_mask = (dates >= t_train_start) & (dates <= t_train_end)
    train_data = panel[train_mask].copy()
    
    if len(train_data) == 0:
        warnings.warn(f"No training data in [{t_train_start}, {t_train_end}]")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    # ===== CRITICAL: Restrict to core universe (eligible tickers) =====
    # Only train on tickers that are in the core universe after duplicate removal
    if 'ticker' in universe_metadata.columns:
        metadata_idx = universe_metadata.set_index('ticker')
    else:
        metadata_idx = universe_metadata
    
    if 'in_core_after_duplicates' in metadata_idx.columns:
        core_tickers = metadata_idx[
            metadata_idx['in_core_after_duplicates'] == True
        ].index
        
        # Filter training data to core tickers only
        tickers_in_train = train_data.index.get_level_values('Ticker')
        core_mask = tickers_in_train.isin(core_tickers)
        train_data = train_data[core_mask].copy()
        
        if len(train_data) == 0:
            warnings.warn(f"No core ticker data in training window")
            baseline_feature = 'Close%-63'
            return (MomentumRankModel(feature=baseline_feature), 
                    [baseline_feature], 
                    pd.Series({baseline_feature: np.nan}))
    
    # Target column
    target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    if target_col not in train_data.columns:
        warnings.warn(f"Target {target_col} not found in panel")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    # ===== 1. Supervised binning =====
    print(f"[train] Fitting supervised bins on training window...")
    
    binning_dict = {}
    
    # Determine which features to bin
    if config.features.use_manual_binning_candidates:
        # Manual mode: Use hardcoded list
        features_to_bin = config.features.binning_candidates
        print(f"[train] Manual binning mode: Using {len(features_to_bin)} specified features")
    else:
        # Auto mode: Bin ALL base features (except metadata/targets)
        # Use same logic as base_features discovery
        excluded_cols = {'Ticker', 'Close', 'Date'}
        features_to_bin = [
            col for col in train_data.columns 
            if col not in excluded_cols
            and not col.startswith('FwdRet_')
            and not col.endswith('_Rank')
            and not col.endswith('_Bin')
        ]
        print(f"[train] Auto binning mode: Binning ALL {len(features_to_bin)} base features")
    
    for feat in features_to_bin:
        if feat not in train_data.columns:
            continue
        
        # Fit bins
        boundaries = fit_supervised_bins(
            train_data[feat],
            train_data[target_col],
            max_depth=config.features.bin_max_depth,
            min_samples_leaf=config.features.bin_min_samples_leaf,
            n_bins=config.features.n_bins,
            random_state=config.features.random_state
        )
        
        binning_dict[feat] = boundaries
        
        # Create binned feature in training data
        binned = np.digitize(train_data[feat].values, boundaries, right=False)
        train_data[f'{feat}_Bin'] = binned.astype('float32')
    
    print(f"[train] Created {len(binning_dict)} binned features")
    
    # ===== 2. Feature selection via CV-IC (out-of-sample IC to prevent overfitting) =====
    print(f"[train] Computing cross-validated ICs for feature selection...")
    
    # ===== AUTO-DISCOVER base features if not specified =====
    if config.features.base_features is None:
        # Automatically use ALL features in panel (except metadata and targets)
        # Exclude: Ticker (string), Close (price level), Date, FwdRet_* (targets), *_Rank (cross-sectional), *_Bin (will be created)
        excluded_cols = {'Ticker', 'Close', 'Date'}
        base_features = [
            col for col in train_data.columns 
            if col not in excluded_cols
            and not col.startswith('FwdRet_')
            and not col.endswith('_Rank')
            and not col.endswith('_Bin')
        ]
        print(f"[train] Auto-discovered {len(base_features)} base features from panel")
    else:
        base_features = config.features.base_features
    
    # Candidate features = base features + binned features (all compete for selection)
    candidate_features = []
    
    # Add base features that exist
    for feat in base_features:
        if feat in train_data.columns:
            candidate_features.append(feat)
    
    # Add binned features that were created
    for feat in binning_dict.keys():
        binned_name = f'{feat}_Bin'
        if binned_name in train_data.columns:
            candidate_features.append(binned_name)
    
    print(f"[train] Evaluating {len(candidate_features)} candidate features ({len(base_features)} raw + {len(binning_dict)} binned)")
    
    # ===== PHASE 2: ENHANCED FEATURE SELECTION PIPELINE =====
    # Stage 0: Univariate filters (IC + MI) + correlation pruning
    # Stage 1: Multivariate selection (Elastic Net) if needed
    
    print(f"[train] ===== PHASE 2 FEATURE SELECTION PIPELINE =====")
    
    from joblib import Parallel, delayed
    
    def compute_single_feature_stats(feat, train_data, target_col, config):
        """Compute IC and MI statistics for a single feature."""
        is_binned = feat.endswith('_Bin')
        
        # Compute IC with fold details
        cv_ic_result = compute_cv_ic_with_folds(
            train_data[feat],
            train_data[target_col],
            config,
            n_splits=5,
            is_already_binned=is_binned
        )
        
        # Compute MI with fold details
        cv_mi_result = compute_cv_mi(
            train_data[feat],
            train_data[target_col],
            config,
            n_splits=5
        )
        
        # Compute IC t-statistic for significance
        fold_ics = cv_ic_result['fold_ics']
        if len(fold_ics) > 1:
            ic_mean = np.mean(fold_ics)
            ic_std = np.std(fold_ics, ddof=1)
            ic_tstat = abs(ic_mean) / (ic_std / np.sqrt(len(fold_ics))) if ic_std > 0 else 0
        else:
            ic_tstat = 0
        
        return feat, {
            'ic': cv_ic_result,
            'mi': cv_mi_result,
            'ic_tstat': ic_tstat
        }
    
    # Parallel execution across features
    n_jobs = config.compute.n_jobs if hasattr(config.compute, 'n_jobs') else -1
    feature_stats = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_single_feature_stats)(feat, train_data, target_col, config)
        for feat in candidate_features
    )
    
    # Extract statistics into structured dictionaries
    cv_ic_values = {feat: stats['ic']['mean_ic'] for feat, stats in feature_stats}
    sign_consistency_values = {feat: stats['ic']['sign_consistency'] for feat, stats in feature_stats}
    ic_tstat_values = {feat: stats['ic_tstat'] for feat, stats in feature_stats}
    cv_mi_values = {feat: stats['mi']['mean_mi'] for feat, stats in feature_stats}
    mi_positivity_values = {feat: stats['mi']['mi_positivity_rate'] for feat, stats in feature_stats}
    
    cv_ic_series = pd.Series(cv_ic_values).dropna()
    sign_consistency_series = pd.Series(sign_consistency_values)
    ic_tstat_series = pd.Series(ic_tstat_values)
    cv_mi_series = pd.Series(cv_mi_values).dropna()
    mi_positivity_series = pd.Series(mi_positivity_values)
    
    print(f"[train] Computed IC and MI statistics for {len(candidate_features)} features")
    
    # ===== STAGE 0: UNIVARIATE FILTERS =====
    
    # Step 1: IC Filter (magnitude + t-stat + sign consistency)
    ic_threshold = config.features.ic_threshold  # 0.02
    min_sign_consistency = 0.80
    min_ic_tstat = 1.96  # 95% confidence
    
    abs_cv_ic = cv_ic_series.abs()
    ic_magnitude_mask = abs_cv_ic >= ic_threshold
    ic_tstat_mask = ic_tstat_series >= min_ic_tstat
    sign_stable_mask = sign_consistency_series >= min_sign_consistency
    
    ic_filter_mask = ic_magnitude_mask & ic_tstat_mask & sign_stable_mask
    
    features_after_ic = cv_ic_series[ic_filter_mask].index.tolist()
    
    print(f"[train] Step 1 (IC Filter): {len(features_after_ic)}/{len(candidate_features)} passed")
    print(f"  - |IC| >= {ic_threshold}: {ic_magnitude_mask.sum()} features")
    print(f"  - IC t-stat >= {min_ic_tstat}: {ic_tstat_mask.sum()} features")
    print(f"  - Sign consistency >= {min_sign_consistency:.0%}: {sign_stable_mask.sum()} features")
    
    if len(features_after_ic) == 0:
        warnings.warn("No features passed IC filter, using momentum rank")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    # Step 2: MI Filter (80% positivity + mean MI > 0)
    min_mi_positivity = 0.80
    
    mi_positivity_mask = mi_positivity_series >= min_mi_positivity
    mean_mi_positive_mask = cv_mi_series > 0
    
    mi_filter_mask = mi_positivity_mask & mean_mi_positive_mask
    mi_filter_mask = mi_filter_mask.reindex(features_after_ic, fill_value=False)
    
    features_after_mi = [f for f in features_after_ic if mi_filter_mask.get(f, False)]
    
    print(f"[train] Step 2 (MI Filter): {len(features_after_mi)}/{len(features_after_ic)} passed")
    print(f"  - MI > 0 in >= {min_mi_positivity:.0%} folds: {mi_positivity_mask.sum()} features")
    print(f"  - Mean MI > 0: {mean_mi_positive_mask.sum()} features")
    
    if len(features_after_mi) == 0:
        warnings.warn("No features passed MI filter, using momentum rank")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    # Step 3: Multicollinearity Filter (correlation threshold 0.75)
    mi_values_dict = cv_mi_series[features_after_mi].to_dict()
    
    features_after_multicol = filter_multicollinear_features(
        train_data=train_data,
        candidate_features=features_after_mi,
        mi_values=mi_values_dict,
        corr_threshold=0.75
    )
    
    print(f"[train] Step 3 (Multicollinearity): {len(features_after_multicol)}/{len(features_after_mi)} passed")
    
    if len(features_after_multicol) == 0:
        warnings.warn("No features passed multicollinearity filter, using momentum rank")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    # ===== STAGE 1: MULTIVARIATE SELECTION (if needed) =====
    
    if len(features_after_multicol) > 15:
        print(f"[train] Stage 1: {len(features_after_multicol)} features exceed threshold, running Elastic Net...")
        
        selected_features, enet_coefs = elastic_net_feature_selection(
            train_data=train_data,
            candidate_features=features_after_multicol,
            target_col=target_col,
            cv_folds=5,
            max_features=15
        )
        
        print(f"[train] Elastic Net selected {len(selected_features)} features")
        
        if len(selected_features) == 0:
            # Fall back to top 15 by IC × MI
            print(f"[train] Elastic Net returned 0 features, falling back to IC × MI ranking")
            composite_scores = abs_cv_ic[features_after_multicol] * cv_mi_series[features_after_multicol]
            selected_features = composite_scores.nlargest(15).index.tolist()
    else:
        # Use all features from Stage 0
        selected_features = features_after_multicol
        print(f"[train] Stage 1: {len(selected_features)} features <= 15, using all from univariate filters")
    
    if len(selected_features) == 0:
        warnings.warn("No features selected after Phase 2 pipeline, using momentum rank")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    print(f"[train] ===== PHASE 2 SUMMARY =====")
    print(f"[train] Final selection: {len(selected_features)} features")
    print(f"[train] Top 5 by |IC|: {abs_cv_ic[selected_features].nlargest(5).to_dict()}")
    print(f"[train] Top 5 by MI: {cv_mi_series[selected_features].nlargest(5).to_dict()}")
    
    # DEBUG: Show how many raw vs binned features were selected
    raw_selected = [f for f in selected_features if not f.endswith('_Bin')]
    binned_selected = [f for f in selected_features if f.endswith('_Bin')]
    print(f"[train] Breakdown: {len(raw_selected)} raw + {len(binned_selected)} binned features")
    
    # DEBUG: Show IC and MI ranges
    ic_values = cv_ic_series[selected_features]
    mi_values = cv_mi_series[selected_features]
    print(f"[train] IC range: [{ic_values.min():.4f}, {ic_values.max():.4f}]")
    print(f"[train] MI range: [{mi_values.min():.4f}, {mi_values.max():.4f}]")
    
    # Selected features already include raw and/or binned versions based on their IC
    # No need to swap - use selected features directly
    final_features = selected_features
    
    # ===== 3. Feature weights (CV-IC-weighted combination) =====
    # CRITICAL: Store SIGNED IC values to preserve direction
    # Positive IC: high feature values -> high returns (rank ascending)
    # Negative IC: high feature values -> low returns (rank descending / flip sign)
    feature_weights = cv_ic_series[final_features].copy()
    # Normalize by absolute values (preserve sign)
    feature_weights = feature_weights / abs(feature_weights).sum()
    
    # ===== 4. Create model object =====
    model = SupervisedBinnedModel(
        binning_dict=binning_dict,
        selected_features=final_features,  # Use final_features (with _Bin suffix where applicable)
        feature_weights=feature_weights,
        feature_ics=cv_ic_series  # Store CV-IC values for diagnostics
    )
    
    # Return model along with diagnostics
    return model, selected_features, cv_ic_series


if __name__ == "__main__":
    print("Alpha models module loaded.")
    print("\nAvailable model types:")
    print("  - MomentumRankModel: Simple baseline")
    print("  - SupervisedBinnedModel: Supervised binning + feature selection")
    print("\nUsage:")
    print("  model = train_alpha_model(panel, metadata, t_start, t_end, config)")
    print("  scores = model.score_at_date(panel, t0, metadata, config)")
