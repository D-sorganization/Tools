# Creates a desktop shortcut for the Tools launcher using the PNG icon.
# Paths are resolved from this script's own location so the script works from
# any checkout (no machine-specific absolute paths).

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = Join-Path $DesktopPath "Tools Launcher (PNG Icon).lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutFile)

# Assuming python is in PATH. If not, this might need adjustment.
# Using pythonw to run without console window if available, else python.
if (Get-Command pythonw -ErrorAction SilentlyContinue) {
    $Shortcut.TargetPath = "pythonw"
}
else {
    $Shortcut.TargetPath = "python"
}

$Shortcut.Arguments = "UnifiedToolsLauncher.py"
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.Description = "Launch Professional Tools Launcher (PNG Icon)"

# Only set an icon when the tracked asset is present; fall back gracefully.
$IconPath = Join-Path $RepoRoot "assets\tools_icon.png"
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = $IconPath
}
else {
    Write-Host "Icon not found at $IconPath - using default icon."
}

$Shortcut.Save()

Write-Host "Created shortcut with PNG icon at: $ShortcutFile"
