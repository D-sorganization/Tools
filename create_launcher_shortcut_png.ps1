$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = "$DesktopPath\Tools Launcher (PNG Icon).lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutFile)

# Assuming python is in PATH. If not, this might need adjustment.
# Using pythonw to run without console window if available, else python.
if (Get-Command pythonw -ErrorAction SilentlyContinue) {
    $Shortcut.TargetPath = "pythonw"
}
else {
    $Shortcut.TargetPath = "python"
}

$Shortcut.Arguments = "tools_launcher.py"
$Shortcut.WorkingDirectory = "C:\Users\diete\Repositories\Tools"
$Shortcut.Description = "Launch Professional Tools Launcher (PNG Icon)"
$Shortcut.IconLocation = "C:\Users\diete\Repositories\Tools\tools_icon.png"
$Shortcut.Save()

Write-Host "Created shortcut with PNG icon at: $ShortcutFile"