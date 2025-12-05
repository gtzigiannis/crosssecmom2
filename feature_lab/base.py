"""
Shared base components for Alpha and Risk feature selection pipelines.

Includes:
- Feature role classification (alpha vs risk)
- DataQualityGate: Coverage, outliers, degenerate features
- RedundancyFilter: Hierarchical clustering + pairwise elimination
- Numba-optimized computation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# Feature Role Classification (Alpha vs Risk)
# =============================================================================

# Risk feature patterns - features matching these are RISK, not ALPHA
RISK_PATTERNS = [
    # Volatility family
    'std', 'ATR', 'BBW', 'parkinson', 'garman_klass', 'rogers_satchell',
    'realized_vol', 'RV_', 'vol_', 'Ret1dZ', 'volatility',
    # Beta family
    'beta', 'corr_mkt', 'corr_VT', 'corr_BNDW', 'r_squared', 'downside_beta',
    'idio', 'corr_VIX', 'corr_MOVE', 'beta_VT', 'beta_BNDW',
    # Liquidity family
    'amihud', 'spread', 'kyle', 'illiq', 'roll_spread', 'ADV_',
    'volume', 'rel_vol', 'turnover', 'obv', 'pv_corr',
    # Regime/tail family
    'regime', 'vol_of_vol', 'Hurst', 'crash_flag', 'meltup_flag',
    'high_vol', 'low_vol', 'streak', 'days_since',
    'max_dd', 'DD', 'var_', 'cvar', 'semi_vol', 'skew', 'kurt',
    'down_corr', 'drawdown_corr',
]


def classify_feature_role(feature_name: str) -> str:
    """
    Classify a feature as 'alpha' or 'risk' based on name patterns.
    
    Rules (from feature_alpha.md, feature_risk.md):
    - Risk features: volatility, beta, liquidity, regime patterns
    - Interaction features: classified based on participants
      - If ANY participant is risk -> risk
      - Otherwise -> alpha
    - All other features -> alpha
    
    Returns: 'alpha' or 'risk'
    """
    feature_lower = feature_name.lower()
    
    # Check for interaction patterns
    interaction_markers = ['_x_', '_div_', '_minus_', '_in_']
    is_interaction = any(marker in feature_lower for marker in interaction_markers)
    
    if is_interaction:
        # For interactions, check if ANY participant is a risk feature
        # Split by interaction markers to get participants
        participants = feature_name
        for marker in interaction_markers:
            participants = participants.replace(marker, '|||')
        participant_list = [p.strip() for p in participants.split('|||') if p.strip()]
        
        # If any participant matches risk patterns -> risk interaction
        for participant in participant_list:
            if _matches_risk_patterns(participant):
                return 'risk'
        return 'alpha'
    
    # For base features, check risk patterns directly
    if _matches_risk_patterns(feature_name):
        return 'risk'
    
    return 'alpha'


def _matches_risk_patterns(name: str) -> bool:
    """Check if feature name matches any risk pattern."""
    name_lower = name.lower()
    for pattern in RISK_PATTERNS:
        if pattern.lower() in name_lower:
            return True
    return False


def get_alpha_features(feature_columns: List[str]) -> List[str]:
    """Get all alpha features from a list of feature columns."""
    return [f for f in feature_columns if classify_feature_role(f) == 'alpha']


def get_risk_features(feature_columns: List[str]) -> List[str]:
    """Get all risk features from a list of feature columns."""
    return [f for f in feature_columns if classify_feature_role(f) == 'risk']


def partition_features(feature_columns: List[str]) -> Dict[str, List[str]]:
    """
    Partition features into alpha and risk categories.
    
    Returns: {'alpha': [...], 'risk': [...]}
    """
    alpha_features = []
    risk_features = []
    
    for f in feature_columns:
        if classify_feature_role(f) == 'risk':
            risk_features.append(f)
        else:
            alpha_features.append(f)
    
    return {'alpha': alpha_features, 'risk': risk_features}


# =============================================================================
# Numba-Optimized Utilities
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_coverage_numba(arr: np.ndarray) -> float:
    """Compute coverage (fraction of non-NaN values) using numba."""
    n = len(arr)
    if n == 0:
        return 0.0
    valid = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            valid += 1
    return valid / n


@jit(nopython=True, cache=True)
def _compute_outlier_frac_numba(arr: np.ndarray, sigma_threshold: float) -> float:
    """Compute fraction of outliers beyond sigma_threshold std devs."""
    n = len(arr)
    if n == 0:
        return 0.0
    
    # Compute mean and std ignoring NaNs
    total = 0.0
    count = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            total += arr[i]
            count += 1
    
    if count < 2:
        return 0.0
    
    mean = total / count
    
    # Compute std
    sq_sum = 0.0
    for i in range(n):
        if not np.isnan(arr[i]):
            sq_sum += (arr[i] - mean) ** 2
    
    std = np.sqrt(sq_sum / (count - 1))
    if std < 1e-10:
        return 0.0
    
    # Count outliers
    threshold = sigma_threshold * std
    outliers = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            if abs(arr[i] - mean) > threshold:
                outliers += 1
    
    return outliers / count


@jit(nopython=True, cache=True)
def _is_degenerate_numba(arr: np.ndarray, min_std: float = 1e-8) -> bool:
    """Check if array has near-zero variance (degenerate)."""
    n = len(arr)
    if n < 2:
        return True
    
    # Compute mean and std ignoring NaNs
    total = 0.0
    count = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            total += arr[i]
            count += 1
    
    if count < 2:
        return True
    
    mean = total / count
    
    sq_sum = 0.0
    for i in range(n):
        if not np.isnan(arr[i]):
            sq_sum += (arr[i] - mean) ** 2
    
    std = np.sqrt(sq_sum / (count - 1))
    return std < min_std


@jit(nopython=True, parallel=True, cache=True)
def _compute_daily_ic_batch_numba(
    X: np.ndarray,  # (n_features, n_samples)
    y: np.ndarray,  # (n_samples,)
    date_starts: np.ndarray,  # Start indices for each date
    date_ends: np.ndarray,    # End indices for each date
    min_samples: int
) -> np.ndarray:
    """
    Compute daily Spearman IC for multiple features in parallel.
    
    Returns: (n_features, n_dates) array of daily ICs
    """
    n_features = X.shape[0]
    n_dates = len(date_starts)
    
    result = np.full((n_features, n_dates), np.nan)
    
    for d in prange(n_dates):
        start = date_starts[d]
        end = date_ends[d]
        y_date = y[start:end]
        
        for f in range(n_features):
            X_date = X[f, start:end]
            
            # Count valid pairs
            valid_count = 0
            for i in range(len(X_date)):
                if not np.isnan(X_date[i]) and not np.isnan(y_date[i]):
                    valid_count += 1
            
            if valid_count < min_samples:
                continue
            
            # Compute Spearman correlation manually
            # Extract valid pairs
            X_valid = np.empty(valid_count)
            y_valid = np.empty(valid_count)
            idx = 0
            for i in range(len(X_date)):
                if not np.isnan(X_date[i]) and not np.isnan(y_date[i]):
                    X_valid[idx] = X_date[i]
                    y_valid[idx] = y_date[i]
                    idx += 1
            
            # Rank transform
            X_ranks = _rank_array_numba(X_valid)
            y_ranks = _rank_array_numba(y_valid)
            
            # Pearson on ranks = Spearman
            ic = _pearson_numba(X_ranks, y_ranks)
            result[f, d] = ic
    
    return result


@jit(nopython=True, cache=True)
def _rank_array_numba(arr: np.ndarray) -> np.ndarray:
    """Compute ranks of array values (average rank for ties)."""
    n = len(arr)
    ranks = np.empty(n)
    
    # Get sorted indices
    sorted_idx = np.argsort(arr)
    
    # Assign ranks (1-based, with average for ties)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[sorted_idx[j]] == arr[sorted_idx[j + 1]]:
            j += 1
        # Average rank for tied values
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    
    return ranks


@jit(nopython=True, cache=True)
def _pearson_numba(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return np.nan
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    
    if var_x < 1e-10 or var_y < 1e-10:
        return np.nan
    
    return cov / np.sqrt(var_x * var_y)


# =============================================================================
# DataQualityGate: Coverage, outliers, degenerate detection
# =============================================================================

@dataclass
class DataQualityResult:
    """Result of data quality check for a single feature."""
    feature: str
    passed: bool
    coverage: float
    outlier_frac: float
    is_degenerate: bool
    reason: Optional[str] = None


class DataQualityGate:
    """
    Data quality gate for feature selection.
    
    Checks:
    1. Coverage: Minimum fraction of non-null values
    2. Outliers: Maximum fraction of values beyond N sigma
    3. Degenerate: Near-zero variance detection
    
    Shared by both Alpha and Risk pipelines.
    """
    
    def __init__(
        self,
        min_coverage: float = 0.85,
        max_outlier_frac: float = 0.01,
        outlier_sigma: float = 10.0,
        n_jobs: int = -1
    ):
        self.min_coverage = min_coverage
        self.max_outlier_frac = max_outlier_frac
        self.outlier_sigma = outlier_sigma
        self.n_jobs = n_jobs
    
    def check_feature(self, values: np.ndarray, feature_name: str) -> DataQualityResult:
        """
        Check data quality for a single feature.
        
        Args:
            values: 1D array of feature values
            feature_name: Feature name for reporting
            
        Returns:
            DataQualityResult with pass/fail and metrics
        """
        # Coverage check
        if HAS_NUMBA:
            coverage = _compute_coverage_numba(values)
        else:
            coverage = np.sum(~np.isnan(values)) / len(values) if len(values) > 0 else 0.0
        
        if coverage < self.min_coverage:
            return DataQualityResult(
                feature=feature_name,
                passed=False,
                coverage=coverage,
                outlier_frac=np.nan,
                is_degenerate=False,
                reason=f"coverage={coverage:.2%} < {self.min_coverage:.0%}"
            )
        
        # Outlier check
        if HAS_NUMBA:
            outlier_frac = _compute_outlier_frac_numba(values, self.outlier_sigma)
        else:
            valid = values[~np.isnan(values)]
            if len(valid) > 1:
                mean, std = np.mean(valid), np.std(valid)
                if std > 1e-10:
                    outliers = np.abs(valid - mean) > self.outlier_sigma * std
                    outlier_frac = np.mean(outliers)
                else:
                    outlier_frac = 0.0
            else:
                outlier_frac = 0.0
        
        if outlier_frac > self.max_outlier_frac:
            return DataQualityResult(
                feature=feature_name,
                passed=False,
                coverage=coverage,
                outlier_frac=outlier_frac,
                is_degenerate=False,
                reason=f"outliers={outlier_frac:.2%} > {self.max_outlier_frac:.0%}"
            )
        
        # Degenerate check
        if HAS_NUMBA:
            is_degenerate = _is_degenerate_numba(values)
        else:
            valid = values[~np.isnan(values)]
            is_degenerate = len(valid) < 2 or np.std(valid) < 1e-8
        
        if is_degenerate:
            return DataQualityResult(
                feature=feature_name,
                passed=False,
                coverage=coverage,
                outlier_frac=outlier_frac,
                is_degenerate=True,
                reason="degenerate (near-zero variance)"
            )
        
        return DataQualityResult(
            feature=feature_name,
            passed=True,
            coverage=coverage,
            outlier_frac=outlier_frac,
            is_degenerate=False
        )
    
    def check_features(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str],
        verbose: bool = True
    ) -> Tuple[List[str], List[DataQualityResult]]:
        """
        Check data quality for multiple features in parallel.
        
        Args:
            panel_df: Panel DataFrame with features
            feature_cols: List of feature column names
            verbose: Print progress
            
        Returns:
            Tuple of (passed_features, all_results)
        """
        def check_one(col):
            return self.check_feature(panel_df[col].values, col)
        
        if self.n_jobs == 1:
            results = [check_one(col) for col in feature_cols]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(check_one)(col) for col in feature_cols
            )
        
        passed = [r.feature for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        if verbose:
            print(f"[DataQualityGate] {len(passed)}/{len(feature_cols)} features passed")
            if failed:
                # Use Counter to properly aggregate failure reasons
                from collections import Counter
                reason_counts = Counter(
                    r.reason.split('=')[0] if r.reason else 'unknown' 
                    for r in failed
                )
                print(f"  Failed reasons: {dict(reason_counts)}")
        
        return passed, results


# =============================================================================
# RedundancyFilter: Hierarchical clustering + pairwise elimination
# =============================================================================

class RedundancyFilter:
    """
    Redundancy filter using hierarchical clustering and pairwise elimination.
    
    Two-stage process:
    1. Within-family: Hierarchical clustering, keep highest-scoring per cluster
    2. Cross-family: Pairwise elimination of highly correlated features
    
    Shared by both Alpha and Risk pipelines.
    """
    
    def __init__(
        self,
        within_family_max_corr: float = 0.70,
        cross_family_max_corr: float = 0.75,
        max_per_family: int = 10,
        n_jobs: int = -1
    ):
        self.within_family_max_corr = within_family_max_corr
        self.cross_family_max_corr = cross_family_max_corr
        self.max_per_family = max_per_family
        self.n_jobs = n_jobs
    
    def filter_within_family(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str],
        scores: Dict[str, float],
        verbose: bool = True
    ) -> List[str]:
        """
        Filter features within a single family using hierarchical clustering.
        
        Keeps the highest-scoring feature from each cluster (at threshold distance).
        Also enforces max_per_family limit.
        
        Args:
            panel_df: Panel DataFrame
            feature_cols: Features in this family
            scores: Dict mapping feature -> composite score
            
        Returns:
            List of non-redundant features
        """
        if len(feature_cols) <= 1:
            return feature_cols
        
        # Compute correlation matrix
        corr_matrix = self._compute_correlation_matrix(panel_df, feature_cols)
        
        # Convert to distance matrix for clustering
        distance_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        # Use condensed distance matrix
        n = len(feature_cols)
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(distance_matrix[i, j])
        
        if len(condensed) == 0:
            return feature_cols
        
        linkage_matrix = linkage(condensed, method='average')
        
        # Cut tree at threshold (1 - max_corr = distance threshold)
        distance_threshold = 1 - self.within_family_max_corr
        cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
        
        # For each cluster, keep the highest-scoring feature
        clusters: Dict[int, List[str]] = {}
        for feat, label in zip(feature_cols, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feat)
        
        selected = []
        for label, cluster_features in clusters.items():
            # Sort by score (descending) and take the best
            cluster_features.sort(key=lambda f: scores.get(f, 0), reverse=True)
            selected.append(cluster_features[0])
        
        # Sort by score and enforce max_per_family
        selected.sort(key=lambda f: scores.get(f, 0), reverse=True)
        selected = selected[:self.max_per_family]
        
        if verbose and len(selected) < len(feature_cols):
            print(f"    Within-family: {len(feature_cols)} -> {len(selected)} features")
        
        return selected
    
    def filter_cross_family(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str],
        scores: Dict[str, float],
        protected: Optional[Set[str]] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Filter features across families using pairwise elimination.
        
        For each highly-correlated pair, drops the lower-scoring feature.
        Protected features are never dropped.
        
        Args:
            panel_df: Panel DataFrame
            feature_cols: All features (from all families)
            scores: Dict mapping feature -> composite score
            protected: Set of features that cannot be dropped
            
        Returns:
            List of non-redundant features
        """
        if len(feature_cols) <= 1:
            return feature_cols
        
        protected = protected or set()
        
        # Compute correlation matrix
        corr_matrix = self._compute_correlation_matrix(panel_df, feature_cols)
        
        # Sort features by score (highest first)
        sorted_features = sorted(feature_cols, key=lambda f: scores.get(f, 0), reverse=True)
        
        # Greedy elimination
        selected = []
        dropped = set()
        
        for feat in sorted_features:
            if feat in dropped:
                continue
            
            selected.append(feat)
            
            # Mark highly correlated features for dropping (unless protected)
            feat_idx = feature_cols.index(feat)
            for other_idx, other in enumerate(feature_cols):
                if other == feat or other in dropped or other in protected:
                    continue
                if abs(corr_matrix[feat_idx, other_idx]) > self.cross_family_max_corr:
                    # Drop the other feature (lower score since we sorted)
                    dropped.add(other)
        
        if verbose and len(selected) < len(feature_cols):
            print(f"  Cross-family: {len(feature_cols)} -> {len(selected)} features")
        
        return selected
    
    def _compute_correlation_matrix(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise Pearson correlation matrix.
        
        Note: Uses Pearson correlation (not Spearman) for redundancy detection.
        This is intentional:
        - Pearson captures linear redundancy in raw feature values
        - Two features that are linearly related contain redundant information
        - Spearman (used for IC) measures rank predictiveness which is different
        
        The goal of redundancy filtering is to remove features that provide
        similar information content, which is better measured by linear correlation
        on the raw values rather than rank correlation.
        """
        n = len(feature_cols)
        corr_matrix = np.eye(n)
        
        # Extract feature values
        data = panel_df[feature_cols].values  # (n_samples, n_features)
        
        # Compute correlations (vectorized)
        for i in range(n):
            for j in range(i + 1, n):
                x = data[:, i]
                y = data[:, j]
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() > 10:
                    try:
                        corr, _ = pearsonr(x[valid], y[valid])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                    except:
                        corr_matrix[i, j] = 0
                        corr_matrix[j, i] = 0
        
        return corr_matrix
    
    def compute_final_pool_stats(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Compute final pool health statistics.
        
        Returns:
            Dict with max_pairwise_corr, mean_pairwise_corr, condition_number
        """
        if len(feature_cols) < 2:
            return {
                'max_pairwise_corr': 0.0,
                'mean_pairwise_corr': 0.0,
                'condition_number': 1.0
            }
        
        corr_matrix = self._compute_correlation_matrix(panel_df, feature_cols)
        
        # Extract upper triangle (excluding diagonal)
        n = len(feature_cols)
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(abs(corr_matrix[i, j]))
        
        max_corr = max(upper_triangle) if upper_triangle else 0.0
        mean_corr = np.mean(upper_triangle) if upper_triangle else 0.0
        
        # Condition number
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            condition_number = max(eigenvalues) / min(eigenvalues) if len(eigenvalues) > 0 else float('inf')
        except:
            condition_number = float('inf')
        
        return {
            'max_pairwise_corr': max_corr,
            'mean_pairwise_corr': mean_corr,
            'condition_number': condition_number
        }
