# PowerShell script to launch the Integrated Data Processor
Write-Host "Starting Integrated Data Processor..." -ForegroundColor Green
Write-Host ""
Write-Host "This version includes:" -ForegroundColor Yellow
Write-Host "- Original CSV processing functionality" -ForegroundColor White
Write-Host "- Format converter with support for multiple file formats" -ForegroundColor White
Write-Host "- Parquet file analyzer" -ForegroundColor White
Write-Host "- All existing plotting and analysis features" -ForegroundColor White
Write-Host ""

# Change to the script directory
Set-Location $PSScriptRoot

try {
    python launch_integrated.py
}
catch {
    Write-Host ""
    Write-Host "Error running the application." -ForegroundColor Red
    Write-Host "Please ensure Python and all dependencies are installed." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
}
