"""
Loky worker initialization - executed in each spawned worker process.
This ensures MKL threading is disabled in all worker processes.
"""
import os

# Set these in every worker process before any numerical libraries load
for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS", "LAPACK_NUM_THREADS"):
    os.environ.setdefault(var, "1")
