$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
$previousBackend = $env:KIDSAFE_BACKEND
try {
    $modelPython = Join-Path $projectRoot '.venv-model\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $modelPython)) {
        throw 'Chay scripts/setup-model.ps1 truoc.'
    }
    & $modelPython -m ml.release_verify
    if ($LASTEXITCODE -ne 0) { throw 'Model thieu hoac sai checksum. Hay nhan lai goi model tu Huy.' }
    $env:KIDSAFE_BACKEND = 'model'
    & $modelPython -m scripts.run_app
    if ($LASTEXITCODE -ne 0) { throw 'App da dung do loi. Xem thong bao phia tren.' }
} finally {
    $env:KIDSAFE_BACKEND = $previousBackend
    Pop-Location
}
