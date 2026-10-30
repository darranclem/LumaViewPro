# LumaViewPro Hardware Deployment Guide

## Quick Deployment Steps
1. Download the `-v13` files from your Studio panel to your `Downloads` folder.
2. Open PowerShell and run:
   ```powershell
   Set-Location "H:\Trae\Projects\LumaViewPro-main"
   Get-ChildItem "$env:USERPROFILE\Downloads\*-v13.*" | ForEach-Object {
       $targetName = $_.Name -replace "-v13", ""
       Copy-Item $_.FullName -Destination ".\$targetName" -Force
   }
   .\.venv\Scripts\Activate.ps1
   python hw_debug.py
   python lumaviewpro.py
   ```
