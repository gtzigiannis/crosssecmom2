# PowerShell wrapper to run feature engineering with proper MKL configuration
# This fixes Windows MKL/Fortran signal handling issues with multiprocessing

$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:BLAS_NUM_THREADS = "1"
$env:LAPACK_NUM_THREADS = "1"

Write-Host "[wrapper] MKL threading disabled for loky compatibility" -ForegroundColor Green
Write-Host "[wrapper] Starting feature engineering..." -ForegroundColor Green

python run_feature_eng_safe.py

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "`n[wrapper] SUCCESS! Feature engineering completed" -ForegroundColor Green
} else {
    Write-Host "`n[wrapper] ERROR: Feature engineering failed with exit code $exitCode" -ForegroundColor Red
}

exit $exitCode
