
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

    $ShortcutFile = "$DesktopPath\$Name.lnk"
    $Shortcut = $WshShell.CreateShortcut($ShortcutFile)
    
    $Shortcut.TargetPath = $Target
    if ($Arguments) {
        $Shortcut.Arguments = $Arguments
    }
    if ($WorkingDirectory) {
        $Shortcut.WorkingDirectory = $WorkingDirectory
    }
    if ($IconPath) {
        $Shortcut.IconLocation = $IconPath
    }
    
    $Shortcut.Save()
    Write-Host "Created shortcut: $ShortcutFile"
}

# 1. Data Processor
# Using pythonw to suppress console.
$PythonW = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $PythonW) { $PythonW = "python" }

Create-Shortcut `
    -Name "Data Processor" `
    -Target $PythonW `
    -Arguments "C:\Users\diete\Repositories\Tools\data_processing\data_processor\python\data_processor\Data_Processor_Integrated.py" `
    -WorkingDirectory "C:\Users\diete\Repositories\Tools\data_processing\data_processor\python\data_processor" `
    -IconPath "C:\Users\diete\Repositories\Tools\tools_icon_alt.ico"

# 2. RRT Path Planner
# Using pythonw.
Create-Shortcut `
    -Name "RRT Path Planner" `
    -Target $PythonW `
    -Arguments "C:\Users\diete\Repositories\Tools\scientific_modeling\rrt_path_planner\python\src\star_wars_rrt.py" `
    -WorkingDirectory "C:\Users\diete\Repositories\Tools\scientific_modeling\rrt_path_planner\python\src" `
    -IconPath "C:\Users\diete\Repositories\Tools\tools_icon_alt.ico"

# 3. Calculator
Create-Shortcut `
    -Name "Calculator" `
    -Target $PythonW `
    -Arguments "C:\Users\diete\Repositories\Tools\web_applications\calculator\calculator.py" `
    -WorkingDirectory "C:\Users\diete\Repositories\Tools\web_applications\calculator" `
    -IconPath "C:\Users\diete\Repositories\Tools\tools_icon_alt.ico"

# 4. Unit Converter (Web App)
# Targetting the HTML file directly let's Windows choose the browser.
Create-Shortcut `
    -Name "Unit Converter" `
    -Target "C:\Users\diete\Repositories\Tools\web_applications\unit_converter\unit-converter-app\index.html" `
    -WorkingDirectory "C:\Users\diete\Repositories\Tools\web_applications\unit_converter\unit-converter-app" `
    -IconPath "C:\Users\diete\Repositories\Tools\web_applications\unit_converter\unit-converter-app\icon.svg" 

Write-Host "Done."
