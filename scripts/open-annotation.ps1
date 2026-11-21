$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$annotationPage = Join-Path $projectRoot 'data\private\pilot-v1\annotation\review-app.html'
if (-not (Test-Path -LiteralPath $annotationPage)) {
    throw 'Chua giai nen goi annotation. Xem docs/APP_ACCEPTANCE_TIEN.md.'
}
Start-Process -FilePath $annotationPage
