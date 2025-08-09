Set-Location $PSScriptRoot
Write-Host "Starting Data Processor..." -ForegroundColor Green
try {
    python Data_Processor_r0.py
} catch {
    Write-Host "Error occurred: $_" -ForegroundColor Red
    Read-Host "Press Enter to close"
}