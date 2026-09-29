param()

# Run this script in the current PowerShell session when possible:
# . .\scripts\activate_benchmark.ps1
# It never changes Conda's default environment.
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH"
}
conda activate benchmark_py3.12
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Write-Host "Activated benchmark_py3.12 for this PowerShell session."
