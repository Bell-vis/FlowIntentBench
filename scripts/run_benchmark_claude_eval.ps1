[CmdletBinding()]
param(
    [string]$Output,
    [int]$MaxApiCalls = 8,
    [double]$MaxWallSeconds = 180,
    [double]$MaxNoScoreSeconds = 1800,
    [int]$JudgmentLimit = 16,
    [int]$ApiJudgeWorkers = 2,
    [int]$ApiSharedConcurrency = 4,
    [switch]$Resume,
    [string]$ReviewerModel = "gpt-6-astra",
    [switch]$FullAnsweredEvaluation,
    [switch]$CheckApi,
    [switch]$PrepareOnly,
    [string]$CondaPrefix,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ($CheckApi) {
    & (Join-Path $PSScriptRoot "invoke_benchmark_python.ps1") `
        -ScriptPath (Join-Path $PSScriptRoot "check_claude_reviewer_api.py") `
        -CondaPrefix $CondaPrefix
    exit $LASTEXITCODE
}
if (-not $Output) {
    $Output = Join-Path $repositoryRoot "outputs\claude_resume_eval"
}
$outputPath = [IO.Path]::GetFullPath($Output)
if ([IO.Path]::GetPathRoot($outputPath) -eq "C:\") {
    throw "Evaluation output on C: is disabled. Choose a path on E: or another data drive."
}

if ($FullAnsweredEvaluation) {
    if (-not $PSBoundParameters.ContainsKey("MaxApiCalls")) {
        $MaxApiCalls = 4096
    }
    if (-not $PSBoundParameters.ContainsKey("MaxWallSeconds")) {
        $MaxWallSeconds = 86400
    }
    if (-not $PSBoundParameters.ContainsKey("JudgmentLimit")) {
        $JudgmentLimit = 32
    }
}

$arguments = @(
    "--output", $outputPath,
    "--reviewer-model", $ReviewerModel,
    "--max-api-calls", $MaxApiCalls,
    "--max-wall-seconds", $MaxWallSeconds,
    "--max-no-score-seconds", $MaxNoScoreSeconds,
    "--judgment-limit", $JudgmentLimit,
    "--api-judge-workers", $ApiJudgeWorkers,
    "--api-shared-concurrency", $ApiSharedConcurrency
) + $RemainingArguments
if ($Resume) {
    $arguments = @("--resume") + $arguments
}
if ($PrepareOnly) {
    $arguments += "--prepare-only"
}
& (Join-Path $PSScriptRoot "invoke_benchmark_python.ps1") `
    -ScriptPath (Join-Path $PSScriptRoot "evaluate_existing_claude_xhigh.py") `
    -PythonArguments $arguments -CondaPrefix $CondaPrefix
exit $LASTEXITCODE
