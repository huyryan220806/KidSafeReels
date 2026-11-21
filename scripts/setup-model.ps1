param([string]$ModelBundle)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $modelPython = Join-Path $projectRoot '.venv-model\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $modelPython)) {
        python -m venv .venv-model
        if ($LASTEXITCODE -ne 0) { throw 'Can Python 3.12 de tao moi truong model.' }
    }
    & $modelPython -m pip install -r requirements-dev.txt -r requirements-pilot.txt -c requirements-release.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Cai thu vien model that bai.' }
    & $modelPython -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'Cai PyTorch CPU that bai.' }
    if ($ModelBundle) {
        & $modelPython -m ml.release_verify --install-bundle $ModelBundle
        if ($LASTEXITCODE -ne 0) { throw 'Goi model khong hop le. Dung lai de kiem tra SHA-256.' }
    }
    & $modelPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Phu thuoc khong tuong thich.' }
    & $modelPython -m ml.release_verify
    if ($LASTEXITCODE -ne 0) { throw 'Chua co du artifact. Nhan ZIP tu Huy va chay lai voi -ModelBundle.' }
    Write-Host 'Da san sang. Chay: powershell -ExecutionPolicy Bypass -File scripts/run-model.ps1'
} finally {
    Pop-Location
}
