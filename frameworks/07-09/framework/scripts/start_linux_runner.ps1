param([string]$Name = 'matbench-harbor-linux')
$ErrorActionPreference = 'Stop'
$releaseRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if ($releaseRoot -notmatch '^([A-Za-z]):\\(.*)$') { throw 'Expected a Windows drive path for Docker Desktop.' }
$linuxRoot = '/run/desktop/mnt/host/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\','/')
$tempPath = Join-Path $releaseRoot 'runner_tmp'
New-Item -ItemType Directory -Force -Path $tempPath | Out-Null
docker run -d --name $Name --add-host host.docker.internal:host-gateway --mount "type=bind,source=$releaseRoot,target=$linuxRoot" --mount 'type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock' -e "TMPDIR=$linuxRoot/runner_tmp" -e "PYTHONPATH=$linuxRoot/framework" -w "$linuxRoot/framework" matbench/harbor-runner:20260906 sleep infinity
if ($LASTEXITCODE -ne 0) { throw 'Linux runner failed to start.' }
docker exec $Name python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:20708/health',timeout=10).read().decode())"
if ($LASTEXITCODE -ne 0) { throw 'Runner cannot reach the Task 08 sidecar.' }
