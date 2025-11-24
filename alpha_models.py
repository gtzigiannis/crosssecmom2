"""
Alpha Models Module
===================
Generic alpha model interface for cross-sectional momentum strategies.

Key principles:
1. All supervised operations (binning, feature selection, model training) should use ONLY training window data
2. Look-ahead bias: binning cutpoints and feature selection from training window should be applied to test dates
3. Model-agnostic interface: score_at_date(panel, t0, ...) returns scores for portfolio construction
4. Binned features are treated on equal footing with raw features

Model lifecycle:
1. train_alpha_model(panel, t_train_start, t_train_end, ...) -> model_object
   - Extracts training window data
   - Performs supervised binning on selected features
   - Performs supervised feature selection (IC-based or other)
   - Fits model on selected features + binned features
   
2. model_object.score_at_date(panel, t0, ...) -> pd.Series
   - Applies stored binning cutpoints to features at t0
   - Computes scores using trained model
   - Returns cross-sectional scores for portfolio formation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
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


class SupervisedBinnedModel(AlphaModel):
    """
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


def train_alpha_model(
    panel: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    t_train_start: pd.Timestamp,
    t_train_end: pd.Timestamp,
    config,
    model_type: str = 'supervised_binned'
) -> AlphaModel:
    """
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
    
    # PHASE 0: Compute CV-IC with fold-level details for sign consistency filtering
    from joblib import Parallel, delayed
    
    def compute_single_cv_ic_detailed(feat, train_data, target_col, config):
        """Helper function for parallel CV-IC computation with fold details."""
        is_binned = feat.endswith('_Bin')
        cv_ic_result = compute_cv_ic_with_folds(
            train_data[feat],
            train_data[target_col],
            config,
            n_splits=5,
            is_already_binned=is_binned
        )
        return feat, cv_ic_result
    
    # Parallel execution across features
    n_jobs = config.compute.n_jobs if hasattr(config.compute, 'n_jobs') else -1
    cv_ic_results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_single_cv_ic_detailed)(feat, train_data, target_col, config)
        for feat in candidate_features
    )
    
    # Extract mean ICs and sign consistency
    cv_ic_values = {feat: result['mean_ic'] for feat, result in cv_ic_results}
    sign_consistency_values = {feat: result['sign_consistency'] for feat, result in cv_ic_results}
    
    cv_ic_series = pd.Series(cv_ic_values).dropna()
    sign_consistency_series = pd.Series(sign_consistency_values)
    
    # PHASE 0 FILTER: Require 80%+ sign consistency to prevent signal reversal
    min_sign_consistency = 0.80
    sign_stable_mask = sign_consistency_series >= min_sign_consistency
    
    n_unstable = (~sign_stable_mask).sum()
    if n_unstable > 0:
        print(f"[train] PHASE 0: Filtered out {n_unstable} features with sign consistency < {min_sign_consistency:.0%}")
    
    # Apply both filters: IC threshold AND sign consistency
    abs_cv_ic = cv_ic_series.abs()
    ic_threshold_mask = abs_cv_ic >= config.features.ic_threshold
    combined_mask = ic_threshold_mask & sign_stable_mask
    
    # Select top features that pass both filters
    selected_features = abs_cv_ic[combined_mask].nlargest(config.features.max_features).index.tolist()
    
    if len(selected_features) == 0:
        warnings.warn("No features passed CV-IC threshold, using momentum rank")
        baseline_feature = 'Close%-63'
        return (MomentumRankModel(feature=baseline_feature), 
                [baseline_feature], 
                pd.Series({baseline_feature: np.nan}))
    
    print(f"[train] Selected {len(selected_features)} features by CV-IC")
    print(f"[train] Top 5 by |CV-IC|: {abs_cv_ic[selected_features].nlargest(5).to_dict()}")
    
    # DEBUG: Show how many raw vs binned features were selected
    raw_selected = [f for f in selected_features if not f.endswith('_Bin')]
    binned_selected = [f for f in selected_features if f.endswith('_Bin')]
    print(f"[train] Breakdown: {len(raw_selected)} raw + {len(binned_selected)} binned features selected")
    
    # DEBUG: Show IC range
    if len(selected_features) > 0:
        ic_values = cv_ic_series[selected_features]
        print(f"[train] IC range: [{ic_values.min():.4f}, {ic_values.max():.4f}]")
    
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
