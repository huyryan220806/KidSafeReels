$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Can Python 3.11 hoac 3.12 de tao moi truong.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt -c requirements-app.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Cai thu vien that bai.' }
    & '.\.venv\Scripts\python.exe' -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Phu thuoc khong tuong thich.' }
    Write-Host 'Cai dat xong. Chay: powershell -ExecutionPolicy Bypass -File scripts/run.ps1'
} finally {
    Pop-Location
}
