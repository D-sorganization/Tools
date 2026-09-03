# Creates desktop shortcuts for a handful of frequently used tools.
# Paths are resolved from this script's own location so the script works from
# any checkout (no machine-specific absolute paths).

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")

function Create-Shortcut {
    param (
        [string]$Name,
        [string]$Target,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$IconPath
    )

    $ShortcutFile = Join-Path $DesktopPath "$Name.lnk"
    $Shortcut = $WshShell.CreateShortcut($ShortcutFile)

    $Shortcut.TargetPath = $Target
    if ($Arguments) {
        $Shortcut.Arguments = $Arguments
    }
    if ($WorkingDirectory) {
        $Shortcut.WorkingDirectory = $WorkingDirectory
    }
    # Only set an icon when the file is present; fall back gracefully.
    if ($IconPath -and (Test-Path $IconPath)) {
        $Shortcut.IconLocation = $IconPath
    }

    $Shortcut.Save()
    Write-Host "Created shortcut: $ShortcutFile"
}

# Using pythonw to suppress console.
$PythonW = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $PythonW) { $PythonW = "python" }

$ToolsIcon = Join-Path $RepoRoot "assets\tools_icon_alt.ico"

# 1. Data Processor
$DataProcessorDir = Join-Path $RepoRoot "src\data_processing\data_processor\python\data_processor"
Create-Shortcut `
    -Name "Data Processor" `
    -Target $PythonW `
    -Arguments (Join-Path $DataProcessorDir "Data_Processor_Integrated.py") `
    -WorkingDirectory $DataProcessorDir `
    -IconPath $ToolsIcon

# 2. RRT Path Planner
$RrtDir = Join-Path $RepoRoot "src\rrt_path_planner\python\src"
Create-Shortcut `
    -Name "RRT Path Planner" `
    -Target $PythonW `
    -Arguments (Join-Path $RrtDir "star_wars_rrt.py") `
    -WorkingDirectory $RrtDir `
    -IconPath $ToolsIcon

# 3. Calculator
$CalculatorDir = Join-Path $RepoRoot "src\web_applications\calculator"
Create-Shortcut `
    -Name "Calculator" `
    -Target $PythonW `
    -Arguments (Join-Path $CalculatorDir "calculator.py") `
    -WorkingDirectory $CalculatorDir `
    -IconPath $ToolsIcon

# 4. Unit Converter (Web App)
# Targeting the HTML file directly lets Windows choose the browser.
$UnitConverterDir = Join-Path $RepoRoot "src\web_applications\unit_converter\unit-converter-app"
Create-Shortcut `
    -Name "Unit Converter" `
    -Target (Join-Path $UnitConverterDir "index.html") `
    -WorkingDirectory $UnitConverterDir `
    -IconPath (Join-Path $UnitConverterDir "icon.svg")

Write-Host "Done."
