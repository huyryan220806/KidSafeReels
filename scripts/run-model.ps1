param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
$previousBackend = $env:KIDSAFE_BACKEND
$previousModelDir = $env:KIDSAFE_MODEL_DIR
try {
    if ($PythonPath) {
        $modelPython = (Resolve-Path -LiteralPath $PythonPath).Path
    } else {
        $modelPython = @('.venv-model\Scripts\python.exe', '.venv\Scripts\python.exe') |
            ForEach-Object { Join-Path $projectRoot $_ } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Select-Object -First 1
    }
    if (-not $modelPython) {
        throw 'Chay scripts/setup-model.ps1 truoc, hoac truyen -PythonPath toi Python da cai model.'
    }
    Write-Host "Python: $modelPython"
    # Verify and serve the same pinned bundle, regardless of a stale shell override.
    $env:KIDSAFE_MODEL_DIR = Join-Path $projectRoot 'models\checkpoints'
    & $modelPython -m ml.release_verify
    if ($LASTEXITCODE -ne 0) { throw 'Model thieu hoac sai checksum. Hay nhan lai goi model tu Huy.' }
    $env:KIDSAFE_BACKEND = 'model'
    & $modelPython -m scripts.run_app
    if ($LASTEXITCODE -ne 0) { throw 'App da dung do loi. Xem thong bao phia tren.' }
} finally {
    $env:KIDSAFE_BACKEND = $previousBackend
    $env:KIDSAFE_MODEL_DIR = $previousModelDir
    Pop-Location
}
