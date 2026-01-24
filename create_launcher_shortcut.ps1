
$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = "$DesktopPath\Tools Launcher.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutFile)

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetDir = $PSScriptRoot

# Check if pythonw is available
if (Get-Command pythonw -ErrorAction SilentlyContinue) {
    $PythonExe = "pythonw"
}
else {
    $PythonExe = "python"
}

$Shortcut.TargetPath = $PythonExe
$Shortcut.Arguments = """$TargetDir\UnifiedToolsLauncher.py"""
$Shortcut.WorkingDirectory = $TargetDir
$Shortcut.Description = "Launch Professional Tools Launcher"

# Look for icon
$IconPath = Join-Path $TargetDir "tools_icon.ico"
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = $IconPath
} else {
   # Try alt icon
   $IconPath = Join-Path $TargetDir "tools_icon_alt.ico"
   if (Test-Path $IconPath) {
       $Shortcut.IconLocation = $IconPath
   }
}

$Shortcut.Save()

Write-Host "Created shortcut at: $ShortcutFile pointing to $TargetDir"
