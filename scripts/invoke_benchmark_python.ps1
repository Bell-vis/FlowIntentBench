[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,
    [string[]]$PythonArguments = @(),
    [string]$CondaPrefix
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $CondaPrefix) {
    $condaCommands = @(Get-Command conda.exe -CommandType Application -ErrorAction Stop)
    foreach ($condaCommand in $condaCommands) {
        $condaRoot = Split-Path (Split-Path $condaCommand.Source -Parent) -Parent
        $candidate = Join-Path $condaRoot "envs\benchmark_py3.12"
        if (Test-Path -LiteralPath (Join-Path $candidate "python.exe") -PathType Leaf) {
            $CondaPrefix = $candidate
            break
        }
    }
    if (-not $CondaPrefix) {
        throw "Cannot locate benchmark_py3.12. Set -CondaPrefix to its environment directory."
    }
}
$CondaPrefix = [IO.Path]::GetFullPath($CondaPrefix)
$python = Join-Path $CondaPrefix "python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Target Python missing: $python. Set -CondaPrefix to the benchmark_py3.12 environment directory."
}
$runtimeCache = Join-Path $repositoryRoot ".runtime-cache"
if ([IO.Path]::GetPathRoot($runtimeCache) -eq "C:\") {
    throw "Runtime files on C: are disabled. Use a repository on a data drive."
}

$names = @("PATH", "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "FLOWINTENTBENCH_RUNTIME_CACHE",
    "TEMP", "TMP", "TMPDIR")
$saved = @{}
foreach ($name in $names) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$code = 1
try {
    Remove-Item Env:PYTHONPATH, Env:PYTHONHOME -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = "1"
    $env:CONDA_PREFIX = $CondaPrefix
    $env:CONDA_DEFAULT_ENV = "benchmark_py3.12"
    $env:PATH = (@($CondaPrefix, (Join-Path $CondaPrefix "Library\mingw-w64\bin"),
        (Join-Path $CondaPrefix "Library\usr\bin"), (Join-Path $CondaPrefix "Library\bin"),
        (Join-Path $CondaPrefix "Scripts"), (Join-Path $CondaPrefix "bin"), $saved["PATH"]) -join ";")
    New-Item -ItemType Directory -Force -Path $runtimeCache | Out-Null
    $env:FLOWINTENTBENCH_RUNTIME_CACHE = $runtimeCache
    $env:TEMP = $runtimeCache
    $env:TMP = $runtimeCache
    $env:TMPDIR = $runtimeCache
    & $python -E -s -X utf8 -u (Join-Path $PSScriptRoot "benchmark_runtime.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Interpreter validation failed; benchmark was not started."
    }
    & $python -E -s -X utf8 -u $ScriptPath @PythonArguments
    $code = $LASTEXITCODE
}
finally {
    foreach ($name in $names) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
    }
}
exit $code
