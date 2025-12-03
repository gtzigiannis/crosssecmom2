# PowerShell script to run feature engineering with proper MKL configuration
# Set environment variables BEFORE Python starts

$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:BLAS_NUM_THREADS = "1"
$env:LAPACK_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"

Write-Host "[wrapper] MKL threading disabled" -ForegroundColor Green
Write-Host "[wrapper] Running: python main.py --step feature_eng" -ForegroundColor Green

python main.py --step feature_eng

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[wrapper] SUCCESS!" -ForegroundColor Green
} else {
    Write-Host "`n[wrapper] ERROR: Exit code $LASTEXITCODE" -ForegroundColor Red
}

exit $LASTEXITCODE
