# setup_env.ps1 - Automated Python 3.10 Venv Setup for LumaViewPro
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " LumaViewPro Python 3.10 Virtual Environment Setup " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$TargetDir = "H:\Trae\Projects\LumaViewPro-main"
if (Test-Path $TargetDir) {
    Set-Location $TargetDir
}

if (-not (Test-Path ".venv")) {
    Write-Host "--> Creating Python 3.10 virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "--> Activating virtual environment & updating pip..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host "--> Installing Kivy 2.3.0 and microscope dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Setup Complete! Launch with: python lumaviewpro.py " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
