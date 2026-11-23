$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.audit-venv\Scripts\python.exe'
$runs = @(
    @{ Profile = 'deepseek-v4-pro-official-task01-03-20260902.profile.yaml'; Output = 'deepseek-v4-pro-official-task01-03-20260902'; KeyEnv = 'DEEPSEEK_PRO_API_KEY' },
    @{ Profile = 'kimi-k26-siliconflow-task01-03-20260902.profile.yaml'; Output = 'kimi-k26-siliconflow-task01-03-20260902'; KeyEnv = 'SILICONFLOW_API_KEY' },
    @{ Profile = 'glm52-siliconflow-task01-03-20260902.profile.yaml'; Output = 'glm52-siliconflow-task01-03-20260902'; KeyEnv = 'SILICONFLOW_API_KEY' }
)
foreach ($run in $runs) {
    $profile = Join-Path $repo ('eval_runs\' + $run.Profile)
    $output = Join-Path $repo ('eval_runs\' + $run.Output)
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $logOut = Join-Path $output 'suite.stdout.log'
    $logErr = Join-Path $output 'suite.stderr.log'
    & $python 'scripts/run_llm_evaluation.py' '--profile' $profile '--engine' 'direct' '--output-dir' $output *>> $logOut 2>> $logErr
    $exitCode = $LASTEXITCODE
    Set-Content -LiteralPath (Join-Path $output 'suite.exit_code.txt') -Value ([string]$exitCode) -Encoding utf8
}
