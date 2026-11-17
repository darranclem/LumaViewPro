# Build Quark's stitching module (quark_tiles) and put it where quark_stitch.py looks for it.
#
# Needs the Quark source (a private repository) and Rust. The module is built once and works
# with every Python from 3.9 on, so the same file can be copied to another PC instead:
#   <LumaViewPro>\lib\quark\quark_tiles.pyd
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\build_quark_tiles.ps1 [-Quark H:\Trae\Projects\Quark]
param([string]$Quark = "H:\Trae\Projects\Quark")
$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $Quark "quark-py\Cargo.toml"))) {
    throw "No quark-py crate in $Quark. Pass -Quark <path to the Quark source>."
}
Push-Location $Quark
try {
    cargo build --release -p quark-py
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Quark "target" }
    $built = Join-Path $target "release\quark_tiles.dll"
    $dest = Join-Path (Split-Path $PSScriptRoot -Parent) "lib\quark"
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item $built (Join-Path $dest "quark_tiles.pyd") -Force
    Write-Host "Installed $(Join-Path $dest 'quark_tiles.pyd')"
} finally {
    Pop-Location
}
