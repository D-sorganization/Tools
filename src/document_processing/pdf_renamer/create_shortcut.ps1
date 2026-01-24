# PowerShell script to create desktop shortcut for PDF Renamer

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "PDF Renamer.lnk"
$TargetPath = Join-Path $PSScriptRoot "PDF_Renamer.bat"
$WorkingDir = $PSScriptRoot

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.WorkingDirectory = $WorkingDir
$Shortcut.Description = "PDF Renamer - AI-Powered PDF Title Extraction and Renaming"
$Shortcut.IconLocation = "%SystemRoot%\System32\shell32.dll,71"
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "Location: $ShortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now double-click 'PDF Renamer' on your desktop to launch the application." -ForegroundColor Yellow

# Wait for user
Read-Host "Press Enter to exit"
