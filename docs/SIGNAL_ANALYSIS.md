# Feature Selection Architecture: Rethinking the Problem

## Diagnosis Summary

After running diagnostics, we discovered:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Formation→Training IC Corr | 0.84 | Signal persists well |
| Sign Consistency | 76.6% | Features don't flip |
| Top IC Values | 0.30-0.45 | Strong univariate signal |
| Ridge R² (20 features) | 3.35% | Weak multivariate signal |
| LassoLarsIC BIC selection | 1 feature | Correct - one signal source |

**The problem isn't signal decay - it's that 1300+ features measure the same thing: momentum.**

---

## Proposed Solutions

### Solution A: Factor-Based Ensemble (Recommended)

Instead of selecting individual features, identify **orthogonal factor groups** and select the best representative from each:

```
Step 1: Cluster features by correlation
        → Momentum cluster (709 features)
        → Volume cluster (23 features)
        → Volatility cluster (131 features)
        → Macro cluster (93 features)
        → ...

Step 2: Within each cluster, select top feature by IC
        → One representative per factor

Step 3: Build model on 5-10 orthogonal factors
        → No regularization collapse because factors are independent
```

**Why this works:**
- Eliminates multicollinearity by design
- Preserves each factor's signal
- Model complexity bounded by # of factors, not # of features
- More interpretable

### Solution B: Composite Factor Construction

Create composite factors as weighted averages within groups:

```
MomentumFactor = Σ IC_i × Feature_i (for all momentum features)
VolumeFactor = Σ IC_i × Feature_i (for all volume features)
...
```

Then use only composite factors in the model.

**Why this works:**
- Aggregates noisy signals into cleaner factors
- Noise cancellation through averaging
- 5-10 factors instead of 1300 features

### Solution C: PCA-Based Dimensionality Reduction

```
Step 1: Standardize all features
Step 2: Run PCA within each category
Step 3: Keep top N components (explaining 90% variance)
Step 4: Use ~20 PC components as features
```

**Why this works:**
- Mathematically optimal compression
- Components are orthogonal by construction
- Captures variance without redundancy

### Solution D: Relaxed Regularization + Stability Selection

If you want to keep the current architecture:

```
Step 1: Use AIC instead of BIC (less penalty)
Step 2: Run LARS multiple times with bootstrap
Step 3: Keep features selected in >50% of runs
Step 4: Refit Ridge on stable feature set
```

**Why this works:**
- AIC selects more features than BIC
- Bootstrap stability ensures robust selection
- Avoids single-run sensitivity

---

## Recommended Implementation: Factor-Based Ensemble

### Phase 1: Feature Clustering (Formation)

```python
def cluster_features_by_correlation(X, n_clusters=10):
    """Group features into correlated clusters using hierarchical clustering."""
    corr_matrix = X.corr()
    distance_matrix = 1 - corr_matrix.abs()
    
    from scipy.cluster.hierarchy import linkage, fcluster
    linkage_matrix = linkage(distance_matrix, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    return dict(zip(X.columns, clusters))
```

### Phase 2: Representative Selection (Training)

```python
def select_cluster_representatives(X, y, cluster_assignments):
    """For each cluster, select the feature with highest |IC|."""
    from scipy.stats import spearmanr
    
    representatives = {}
    for cluster_id in set(cluster_assignments.values()):
        cluster_features = [f for f, c in cluster_assignments.items() if c == cluster_id]
        
        # Find best feature by IC
        best_ic = 0
        best_feature = None
        for feat in cluster_features:
            ic, _ = spearmanr(X[feat], y)
            if abs(ic) > abs(best_ic):
                best_ic = ic
                best_feature = feat
        
        representatives[cluster_id] = (best_feature, best_ic)
    
    return representatives
```

### Phase 3: Ridge on Representatives

```python
def fit_factor_model(X, y, representatives):
    """Fit Ridge on cluster representatives (no Lasso needed - already decorrelated)."""
    selected_features = [rep[0] for rep in representatives.values()]
    
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.01)
    model.fit(X[selected_features], y)
    
    return model, selected_features
```

---

## Expected Outcomes

| Approach | Expected # Features | Why |
|----------|--------------------:|-----|
| Current (LassoLarsIC BIC) | 1 | Correctly identifies one signal |
| Factor-Based Ensemble | 8-12 | One per orthogonal factor |
| PCA Components | 15-20 | Based on variance explained |
| Relaxed AIC | 5-15 | Less penalty than BIC |

---

## Quick Win: Use AIC Instead of BIC

BIC: $\text{penalty} = k \log(n)$
AIC: $\text{penalty} = 2k$

For n=18,000 samples:
- BIC penalty: ~9.8 × k
- AIC penalty: 2 × k

**BIC is ~5× more aggressive than AIC for this sample size!**

Change in config.py:
```python
lars_criterion: str = 'aic'  # Was 'bic'
```

This alone should increase selected features from 1 to ~5-10.
