$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        throw 'Chay scripts/setup.ps1 truoc.'
    }
    & '.\.venv\Scripts\python.exe' -m scripts.run_app
    if ($LASTEXITCODE -ne 0) { throw 'App da dung do loi. Xem thong bao phia tren.' }
} finally {
    Pop-Location
}
