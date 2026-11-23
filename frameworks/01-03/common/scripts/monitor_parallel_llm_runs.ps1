param(
    [string]$Root = (Join-Path (Split-Path -Parent $PSScriptRoot) 'eval_runs'),
    [int]$IntervalSeconds = 30
)

$runs = @(
    'deepseek-v4-pro-parallel-task01-03-20260902',
    'kimi-k26-parallel-task01-03-20260902',
    'glm52-parallel-task01-03-20260902'
)
$report = Join-Path $Root 'parallel_llm_monitor_20260902.json'

while ($true) {
    $items = foreach ($run in $runs) {
        $runDir = Join-Path $Root $run
        $tasks = foreach ($task in @('task01', 'task02', 'task03')) {
            $callLog = Join-Path $runDir ($task + '_llm_calls.jsonl')
            $workspaceResult = Join-Path $runDir ($task + '_workspace\\run_result.json')
            $reward = Join-Path $runDir ($task + '_verifier_logs\\verifier\\reward.json')
            [ordered]@{
                task = $task
                call_log_exists = Test-Path -LiteralPath $callLog
                call_log_bytes = if (Test-Path -LiteralPath $callLog) { (Get-Item -LiteralPath $callLog).Length } else { 0 }
                call_log_last_write = if (Test-Path -LiteralPath $callLog) { (Get-Item -LiteralPath $callLog).LastWriteTime.ToString('o') } else { $null }
                run_result_exists = Test-Path -LiteralPath $workspaceResult
                reward_exists = Test-Path -LiteralPath $reward
            }
        }
        $stderr = Join-Path $runDir 'runner.stderr.log'
        [ordered]@{
            run = $run
            suite_finished = Test-Path -LiteralPath (Join-Path $runDir 'suite.exit_code.txt')
            stderr_bytes = if (Test-Path -LiteralPath $stderr) { (Get-Item -LiteralPath $stderr).Length } else { 0 }
            tasks = @($tasks)
        }
    }
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        policy = 'observe only; never terminate a run based on elapsed time'
        runs = @($items)
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $report -Encoding utf8
    if (@($items | Where-Object { -not $_.suite_finished }).Count -eq 0) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
