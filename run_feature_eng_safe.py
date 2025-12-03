"""
Safe feature engineering runner with proper MKL threading configuration
"""
import os
import sys

# MUST set these BEFORE importing numpy/scipy/pandas
# This prevents Intel MKL from using threading which conflicts with loky
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
os.environ['LAPACK_NUM_THREADS'] = '1'

# Now safe to import and run
from config import get_default_config
from feature_engineering import run_feature_engineering

if __name__ == '__main__':
    config = get_default_config()
    panel_df = run_feature_engineering(config)
    print(f"\n[SUCCESS] Feature engineering completed!")
    print(f"[SUCCESS] Panel shape: {panel_df.shape}")
