$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw '请先运行 python -m venv .venv，再安装 backend/requirements.txt' }
New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot 'data') | Out-Null
$env:APP_MODE = 'demo'
$env:DATABASE_URL = 'sqlite:///./data/app.db'
Start-Process -FilePath $taskPython -ArgumentList @('-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'data\backend.log') -RedirectStandardError (Join-Path $taskRoot 'data\backend-error.log')
$taskNpm = (Get-Command npm.cmd).Source
Start-Process -FilePath $taskNpm -ArgumentList @('run','dev','--prefix','frontend') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'data\frontend.log') -RedirectStandardError (Join-Path $taskRoot 'data\frontend-error.log')
Write-Output '知序演示服务正在启动：http://localhost:5173'
Write-Output '服务日志位于 data。请勿重复运行此脚本；端口已被占用时先停止旧进程。'
