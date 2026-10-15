param([string]$DownloadProxy = '')
# 独立OCR环境只用于文档导入，不改变后端.venv。
$ErrorActionPreference = 'Stop'
$ocrProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ocrProjectRoot
$ocrPython = Join-Path $ocrProjectRoot '.ocr-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ocrPython)) {
    & py -3.11 -m venv .ocr-venv
    if ($LASTEXITCODE -ne 0) { throw 'Please install Python 3.11 first.' }
}
$ocrRequirements = Join-Path $ocrProjectRoot 'requirements-ocr.lock.txt'
if (-not (Test-Path -LiteralPath $ocrRequirements)) { $ocrRequirements = Join-Path $ocrProjectRoot 'requirements-ocr.txt' }
$ocrInstallArgs = @('-m', 'pip', 'install', '-r', $ocrRequirements)
if ($DownloadProxy) { $ocrInstallArgs += @('--proxy', $DownloadProxy) }
& $ocrPython @ocrInstallArgs
if ($LASTEXITCODE -ne 0) { throw 'OCR dependency installation failed.' }
Write-Host 'OCR environment is ready. Models are prepared on first parse.'