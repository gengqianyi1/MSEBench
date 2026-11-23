$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $root 'start_task.py') @args
exit $LASTEXITCODE
