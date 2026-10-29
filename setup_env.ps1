# setup_env.ps1 - Automated Python 3.10 Venv Setup for LumaViewPro
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " LumaViewPro Python 3.10 Virtual Environment Setup " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$TargetDir = "H:\Trae\Projects\LumaViewPro-main"
$StockDir  = "C:\Program Files (x86)\LumaViewPro-stock"
$Py310Exe  = "C:\Users\dkc25.BLUE\AppData\Local\Programs\Python\Python310\python.exe"

Set-Location $TargetDir

# 1. Check Python 3.10 Executable
if (-not (Test-Path $Py310Exe)) {
    Write-Host "[Error] Python 3.10 executable not found at: $Py310Exe" -ForegroundColor Red
    Write-Host "Please verify Python 3.10 install path." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/5] Removing old incompatible venv if present..." -ForegroundColor Yellow
if (Test-Path ".\venv") {
    Remove-Item -Recurse -Force ".\venv"
}

Write-Host "[2/5] Creating new Python 3.10 Virtual Environment..." -ForegroundColor Yellow
& $Py310Exe -m venv venv

Write-Host "[3/5] Copying stock hardware driver modules..." -ForegroundColor Yellow
if (Test-Path $StockDir) {
    Copy-Item "$StockDir\pyloncamera*" -Destination "$TargetDir\" -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item "$StockDir\motorboard*" -Destination "$TargetDir\" -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item "$StockDir\ledboard*" -Destination "$TargetDir\" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Stock driver modules copied to project directory." -ForegroundColor Green
} else {
    Write-Host "[Warning] Stock directory $StockDir not found. Skipping driver file copy." -ForegroundColor Yellow
}

Write-Host "[4/5] Setting JAVA_HOME and Maven Environment Variables..." -ForegroundColor Yellow
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-8"
$env:Path += ";C:\Maven\apache-maven-3.9.8\bin;C:\Program Files\Zulu\zulu-8\bin"

Write-Host "[5/5] Installing dependencies into Python 3.10 venv..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\venv\Scripts\pip.exe install kivy==2.3.0 opencv-python numpy Pillow jgo scyjava imglyb pyimagej

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " SETUP COMPLETE! " -ForegroundColor Green
Write-Host " To run LumaViewPro:" -ForegroundColor White
Write-Host "   .\\venv\\Scripts\\Activate.ps1" -ForegroundColor Yellow
Write-Host "   python hw_debug.py" -ForegroundColor Yellow
Write-Host "   python lumaviewpro.py" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
