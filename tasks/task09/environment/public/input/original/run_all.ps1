$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeRoot = Join-Path $env:USERPROFILE ".conda\envs\xtb-benchmark"
if (-not (Test-Path -LiteralPath (Join-Path $runtimeRoot "python.exe"))) {
    throw "xtb-benchmark Python is unavailable"
}
$condaExe = $null
if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) {
    $condaExe = $env:CONDA_EXE
} else {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) { $condaExe = $condaCommand.Source }
}
if (-not $condaExe) {
    throw "conda is required to establish the validated Windows DLL search order"
}
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

function Invoke-BenchmarkPython {
    param([Parameter(Mandatory = $true)][string[]]$PythonArgs)
    & $condaExe run -p $runtimeRoot --no-capture-output python @PythonArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Push-Location $projectRoot
try {
    Invoke-BenchmarkPython -PythonArgs @("-s", "-m", "scripts.validate_project", "--delivery")
    Invoke-BenchmarkPython -PythonArgs @("-s", "-m", "pytest", "-q")
    Invoke-BenchmarkPython -PythonArgs @(
        "-s", "-m", "scripts.replay_trajectories",
        "baseline", "kimi", "deepseek", "glm",
        "--output", ".test_tmp/replay_validation_delivery.json"
    )
    Invoke-BenchmarkPython -PythonArgs @("-s", "-m", "scripts.secret_scan")
} finally {
    Pop-Location
}
