[CmdletBinding()]
param(
    [string]$Output,
    [double]$Interval = 60,
    [switch]$Once,
    [string]$CondaPrefix
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $Output) {
    $Output = Join-Path $repositoryRoot "outputs\claude_resume_eval"
}
$arguments = @("--output", [IO.Path]::GetFullPath($Output), "--interval", "$Interval")
if ($Once) {
    $arguments += "--once"
}
& (Join-Path $PSScriptRoot "invoke_benchmark_python.ps1") `
    -ScriptPath (Join-Path $PSScriptRoot "monitor_claude_xhigh_eval.py") `
    -PythonArguments $arguments -CondaPrefix $CondaPrefix
exit $LASTEXITCODE
