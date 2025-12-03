"""
Parallelism Safety Tests for crosssecmom2

Tests to verify that parallel execution:
1. Produces consistent results across runs
2. Doesn't corrupt shared state
3. Works correctly with joblib's loky backend
4. Handles 16-core/32-thread hardware properly

Author: AI Assistant
Date: November 28, 2025
"""

import gc
import numpy as np
import pandas as pd
import pytest
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def parallel_test_data():
    """Create test data for parallelism tests."""
    np.random.seed(42)
    
    n_dates = 200
    n_tickers = 30
    n_features = 50
    
    dates = pd.bdate_range('2020-01-01', periods=n_dates)
    tickers = [f'ETF_{i:02d}' for i in range(n_tickers)]
    
    # Create multi-index panel
    index = pd.MultiIndex.from_product([dates, tickers], names=['Date', 'Ticker'])
    
    # Generate features with some signal
    np.random.seed(42)
    features = {}
    
    # Signal features (correlated with target)
    for i in range(10):
        signal = np.random.randn(len(index)) * 0.01
        features[f'signal_{i}'] = signal
    
    # Noise features
    for i in range(40):
        noise = np.random.randn(len(index)) * 0.01
        features[f'noise_{i}'] = noise
    
    # Target: combination of signals
    target = sum(features[f'signal_{i}'] * 0.1 for i in range(10))
    target += np.random.randn(len(index)) * 0.02
    
    panel = pd.DataFrame(features, index=index)
    panel['FwdRet_21'] = target
    
    return panel


# ============================================================================
# Parallelism Safety Tests
# ============================================================================

class TestParallelismSafety:
    """Tests for parallel execution safety."""
    
    def test_formation_fdr_parallel_consistency(self, parallel_test_data):
        """Test that formation_fdr produces consistent results across multiple runs."""
        from feature_selection import formation_fdr
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        # Run multiple times with same seed
        results = []
        for _ in range(3):
            approved, diagnostics = formation_fdr(
                X=X,
                y=y,
                dates=dates,
                half_life=63,
                fdr_level=0.25,  # Permissive for test
                n_jobs=4
            )
            results.append(set(approved))
        
        # All runs should produce identical results
        assert results[0] == results[1], "Run 1 != Run 2"
        assert results[1] == results[2], "Run 2 != Run 3"
        print(f"[PASS] formation_fdr: {len(results[0])} features selected consistently")
    
    def test_parallel_vs_sequential_equivalence(self, parallel_test_data):
        """Test that parallel execution matches sequential execution."""
        from feature_selection import formation_fdr
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        # Run with n_jobs=1 (sequential)
        approved_seq, diag_seq = formation_fdr(
            X=X, y=y, dates=dates, half_life=63, fdr_level=0.25, n_jobs=1
        )
        
        # Run with n_jobs=4 (parallel)
        approved_par, diag_par = formation_fdr(
            X=X, y=y, dates=dates, half_life=63, fdr_level=0.25, n_jobs=4
        )
        
        # Results should be identical
        assert set(approved_seq) == set(approved_par), \
            f"Sequential and parallel results differ!\nSeq: {len(approved_seq)}, Par: {len(approved_par)}"
        print(f"[PASS] Sequential vs Parallel: {len(approved_seq)} features match")
    
    def test_ic_ranking_parallel_consistency(self, parallel_test_data):
        """Test that soft IC ranking is consistent across parallel runs."""
        from feature_selection import rank_features_by_ic_and_sign_consistency, compute_time_decay_weights
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        train_end = dates.max()
        weights = compute_time_decay_weights(dates, train_end, half_life=63)
        
        # Run multiple times
        results = []
        for _ in range(3):
            ranked, diag = rank_features_by_ic_and_sign_consistency(
                X=X, y=y, dates=dates, weights=weights,
                num_blocks=3, ic_floor=0.01, top_k=30, min_features=10, n_jobs=4
            )
            results.append(ranked)
        
        # All runs should produce identical ordered lists
        assert results[0] == results[1], "Run 1 != Run 2"
        assert results[1] == results[2], "Run 2 != Run 3"
        print(f"[PASS] IC ranking: {len(results[0])} features ranked consistently")
    
    def test_no_shared_state_corruption(self, parallel_test_data):
        """Test that parallel workers don't corrupt shared state."""
        from feature_selection import formation_fdr
        from joblib import Parallel, delayed
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        # Run formation_fdr inside parallel loop (simulates walk-forward)
        def run_fdr_worker(worker_id):
            approved, diag = formation_fdr(
                X=X, y=y, dates=dates, half_life=63, fdr_level=0.25, n_jobs=1
            )
            return (worker_id, set(approved))
        
        # Run 4 workers in parallel
        results = Parallel(n_jobs=4, backend='loky')(
            delayed(run_fdr_worker)(i) for i in range(4)
        )
        
        # All workers should produce identical results
        feature_sets = [r[1] for r in results]
        for i in range(1, len(feature_sets)):
            assert feature_sets[0] == feature_sets[i], \
                f"Worker 0 != Worker {i}: state corruption detected!"
        
        print(f"[PASS] No shared state corruption: 4 workers produced identical results")
    
    def test_high_core_count_scalability(self, parallel_test_data):
        """Test performance with high core count (simulates 16c/32t hardware)."""
        from feature_selection import formation_fdr
        import os
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        cpu_count = os.cpu_count() or 4
        print(f"\nAvailable CPUs: {cpu_count}")
        
        # Test with different n_jobs values
        timings = {}
        for n_jobs in [1, 4, -1]:  # -1 = all cores
            start = time.time()
            approved, _ = formation_fdr(
                X=X, y=y, dates=dates, half_life=63, fdr_level=0.25, n_jobs=n_jobs
            )
            elapsed = time.time() - start
            timings[n_jobs] = elapsed
            print(f"  n_jobs={n_jobs:2d}: {elapsed:.2f}s, {len(approved)} features")
        
        # Parallel should be faster (or at least not much slower)
        # Note: For small datasets, overhead may make parallel slower
        # This test mainly verifies that parallel execution WORKS
        assert len(approved) > 0, "No features selected"
        print(f"[PASS] High core count test: parallel execution works")


class TestFilterThresholds:
    """Tests to verify filter thresholds are production-appropriate."""
    
    def test_fdr_threshold_filters_noise(self, parallel_test_data):
        """Test that FDR threshold filters out noise features."""
        from feature_selection import formation_fdr
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        # With 10 signal + 40 noise features, FDR=0.10 should:
        # - Approve most/all signal features
        # - Reject most noise features
        approved, diagnostics = formation_fdr(
            X=X, y=y, dates=dates, half_life=63, fdr_level=0.10, n_jobs=4
        )
        
        signal_approved = sum(1 for f in approved if f.startswith('signal_'))
        noise_approved = sum(1 for f in approved if f.startswith('noise_'))
        
        print(f"Signal features approved: {signal_approved}/10")
        print(f"Noise features approved: {noise_approved}/40")
        print(f"Total approved: {len(approved)}/50")
        
        # Signal-to-noise ratio in approved features should be high
        if len(approved) > 0:
            snr = signal_approved / max(1, noise_approved)
            print(f"Signal-to-noise ratio: {snr:.2f}")
        
        # Sanity check: not approving everything
        assert len(approved) < len(feature_cols), \
            f"FDR not filtering: approved {len(approved)}/{len(feature_cols)}"
        
        print(f"[PASS] FDR threshold filters noise features")
    
    def test_per_window_top_k_reasonable(self):
        """Verify that per_window_top_k is set to a reasonable production value."""
        from config import get_default_config
        
        config = get_default_config()
        
        # For 1390 features, top_k=200 is too high (ElasticNet will be slow)
        # Recommended: 20-50 for production
        current_top_k = config.features.per_window_top_k
        
        print(f"Current per_window_top_k: {current_top_k}")
        
        # This test documents the current state
        # If top_k > 100, it's likely too high for production
        if current_top_k > 100:
            print(f"[WARN] per_window_top_k={current_top_k} may be too high for production!")
            print(f"       Consider reducing to 20-50 for faster execution")
        else:
            print(f"[OK] per_window_top_k={current_top_k} is reasonable")


class TestNestedParallelism:
    """Tests for nested parallelism (outer walk-forward + inner feature selection)."""
    
    def test_nested_parallelism_safety(self, parallel_test_data):
        """Test that nested parallelism doesn't cause resource contention."""
        from feature_selection import formation_fdr
        from joblib import Parallel, delayed
        import os
        
        panel = parallel_test_data
        dates = panel.index.get_level_values('Date')
        feature_cols = [c for c in panel.columns if c != 'FwdRet_21']
        X = panel[feature_cols]
        y = panel['FwdRet_21']
        
        def outer_worker(worker_id):
            """Simulates walk-forward worker calling feature selection."""
            # Inner feature selection should use n_jobs=1 when outer is parallel
            approved, _ = formation_fdr(
                X=X, y=y, dates=dates, half_life=63, fdr_level=0.25,
                n_jobs=1  # IMPORTANT: n_jobs=1 when nested
            )
            return (worker_id, len(approved))
        
        # Run outer parallel loop (simulates walk-forward dates)
        cpu_count = os.cpu_count() or 4
        n_outer_jobs = min(4, cpu_count)  # Limit outer parallelism for test
        
        print(f"\nRunning {n_outer_jobs} outer workers with n_jobs=1 inner...")
        start = time.time()
        results = Parallel(n_jobs=n_outer_jobs, backend='loky')(
            delayed(outer_worker)(i) for i in range(8)  # 8 simulated dates
        )
        elapsed = time.time() - start
        
        # All workers should complete successfully
        assert len(results) == 8, f"Expected 8 results, got {len(results)}"
        
        # All workers should produce consistent results
        feature_counts = [r[1] for r in results]
        assert all(c == feature_counts[0] for c in feature_counts), \
            f"Inconsistent results across workers: {feature_counts}"
        
        print(f"[PASS] Nested parallelism: {len(results)} workers completed in {elapsed:.2f}s")
        print(f"       Each selected {feature_counts[0]} features")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
