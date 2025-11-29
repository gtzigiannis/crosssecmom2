"""
Site-wide customization for MKL threading.
This file is automatically loaded by Python before any other imports.
Place in: D:\\REPOSITORY\\morias\\Quant\\strategies\\crosssecmom2\\sitecustomize.py
"""
import os

# Force single-threaded BLAS/MKL before ANY numerical library loads
for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
            "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS", "LAPACK_NUM_THREADS"):
    os.environ.setdefault(var, "1")
