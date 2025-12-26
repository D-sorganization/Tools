$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Folder Tool.lnk"
$PythonPath = (Get-Command python).Source
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PythonPath
$Shortcut.Arguments = """$PSScriptRoot\Folders_Tool_r0.py"""
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.IconLocation = "$PSScriptRoot\paper_plane_icon.ico"
$Shortcut.Save()
Write-Host "Shortcut created at $ShortcutPath with Python at $PythonPath"
