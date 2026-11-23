param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$System32 = Join-Path $env:SystemRoot "System32"
$env:PATH = $System32 + [System.IO.Path]::PathSeparator + $env:PATH
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

& conda run -n xtb-benchmark python (Join-Path $Root "scripts\run_all.py") $Command @RemainingArgs
exit $LASTEXITCODE
