"""
Minimal test to isolate the crash location
"""
import os
for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS", "LAPACK_NUM_THREADS"):
    os.environ.setdefault(var, "1")

print("Step 1: Importing modules...")
import pandas as pd
from config import get_default_config
from data_manager import CrossSecMomDataManager
print("  ✓ Imports successful")

print("\nStep 2: Loading config...")
config = get_default_config()
print(f"  ✓ Config loaded: {config.time.start_date} to {config.time.end_date}")

print("\nStep 3: Loading universe...")
universe = pd.read_csv(config.paths.universe_csv)
tickers = universe['Ticker'].tolist()
print(f"  ✓ Universe loaded: {len(tickers)} tickers")

print("\nStep 4: Creating DataManager...")
dm = CrossSecMomDataManager(str(config.paths.data_dir))
print("  ✓ DataManager created")

print(f"\nStep 5: Fetching OHLCV data for {len(tickers)} tickers...")
print("  (This is where the crash usually happens)")
try:
    data = dm.fetch_ohlcv_data(tickers, config.time.start_date, config.time.end_date)
    print(f"  ✓ SUCCESS! Loaded {len(data)} tickers")
except Exception as e:
    print(f"  ✗ FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
