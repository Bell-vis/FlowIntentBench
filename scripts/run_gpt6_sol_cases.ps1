[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("prepare", "preflight", "collect", "status", "stop")]
    [string]$Command = "collect",
    [string]$OutputRoot,
    [int]$Workers = 2,
    [int]$MaxSlots,
    [double]$Watch = 0,
    [switch]$Json,
    [string[]]$Model,
    [switch]$UseWsl,
    [string]$WslDistribution,
    [switch]$WindowsLocal,
    [switch]$RetryInfrastructure
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repositoryRoot "outputs\expansion96_n3_gpt6_sol"
}
$arguments = @(
    (Join-Path $repositoryRoot "scripts\run_gpt6_sol_cases.py"),
    $Command,
    "--output-root", ([IO.Path]::GetFullPath($OutputRoot)),
    "--workers", $Workers
)
if ($MaxSlots -gt 0) { $arguments += @("--max-slots", $MaxSlots) }
if ($Watch -gt 0) { $arguments += @("--watch", $Watch) }
if ($Json) { $arguments += "--json" }
foreach ($value in ($Model | Where-Object { $_ })) { $arguments += @("--model", $value) }
if ($WindowsLocal) { $arguments += "--windows-local" }
if ($RetryInfrastructure) { $arguments += "--retry-infrastructure" }

if ($UseWsl) {
    if ($WindowsLocal) {
        throw "-WindowsLocal cannot be combined with -UseWsl; WSL should use the strict bubblewrap runtime."
    }
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) {
        throw "-UseWsl was requested, but wsl.exe is not available. Install/configure WSL first."
    }
    $wslPrefix = @()
    if ($WslDistribution) { $wslPrefix += @("-d", $WslDistribution) }
    $linuxRoot = (& $wsl.Source @wslPrefix wslpath -a -u $repositoryRoot).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxRoot) {
        throw "Could not translate the repository path into WSL. Check the selected WSL distribution."
    }
    $linuxOutputRoot = (& $wsl.Source @wslPrefix wslpath -a -u ([IO.Path]::GetFullPath($OutputRoot))).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxOutputRoot) {
        throw "Could not translate the output path into WSL. Check the selected WSL distribution."
    }
    $linuxArguments = @(
        "scripts/run_gpt6_sol_cases.py",
        $Command,
        "--output-root", $linuxOutputRoot,
        "--workers", $Workers
    )
    if ($MaxSlots -gt 0) { $linuxArguments += @("--max-slots", $MaxSlots) }
    if ($Watch -gt 0) { $linuxArguments += @("--watch", $Watch) }
    if ($Json) { $linuxArguments += "--json" }
    if ($RetryInfrastructure) { $linuxArguments += "--retry-infrastructure" }
    foreach ($value in ($Model | Where-Object { $_ })) { $linuxArguments += @("--model", $value) }
    $wslCommand = @()
    if ($WslDistribution) { $wslCommand += @("-d", $WslDistribution) }
    $wslCommand += @("--cd", $linuxRoot, "--", "conda", "run", "--no-capture-output", "-n", "benchmark_py3.12", "python")
    $wslCommand += $linuxArguments
    & $wsl.Source @wslCommand
    exit $LASTEXITCODE
}

$savedPythonPath = $env:PYTHONPATH
$savedPythonNoUserSite = $env:PYTHONNOUSERSITE
$python = $env:FLOWINTENT_PYTHON
$exitCode = 0
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = "1"
    if ($python) {
        & $python @arguments
        $exitCode = $LASTEXITCODE
    }
    # Keep the project runtime isolated from the base Conda/ParaView Python.
    # The benchmark package requires Python 3.12 and a clean PYTHONPATH.
    elseif ((Get-Command conda -ErrorAction SilentlyContinue) -and $env:CONDA_DEFAULT_ENV -ne "benchmark_py3.12") {
        & conda run --no-capture-output -n benchmark_py3.12 python @arguments
        $exitCode = $LASTEXITCODE
    }
    else {
        $python = (Get-Command python -ErrorAction Stop).Source
        & $python @arguments
        $exitCode = $LASTEXITCODE
    }
}
finally {
    if ($null -eq $savedPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
    else { $env:PYTHONPATH = $savedPythonPath }
    if ($null -eq $savedPythonNoUserSite) { Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue }
    else { $env:PYTHONNOUSERSITE = $savedPythonNoUserSite }
}
exit $exitCode
