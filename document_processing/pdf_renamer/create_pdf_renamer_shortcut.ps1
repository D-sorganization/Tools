# Enhanced PowerShell script to create desktop shortcut for PDF Renamer
# This script creates a professional desktop shortcut with proper icon and taskbar support

param(
    [switch]$Verbose
)

function Write-Status {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

try {
    Write-Status "🚀 Creating PDF Renamer Desktop Shortcut..." "Cyan"
    Write-Status ""

    # Get paths
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RootDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
    $DesktopPath = [Environment]::GetFolderPath('Desktop')
    $ShortcutPath = Join-Path $DesktopPath "PDF Renamer.lnk"
    
    # Target files
    $BatchFile = Join-Path $ScriptDir "PDF_Renamer.bat"
    $IconFile = Join-Path $RootDir "tools_icon.ico"
    $FallbackIconFile = Join-Path $RootDir "tools_icon_hq.ico"
    
    if ($Verbose) {
        Write-Status "Script Directory: $ScriptDir" "Gray"
        Write-Status "Root Directory: $RootDir" "Gray"
        Write-Status "Desktop Path: $DesktopPath" "Gray"
        Write-Status "Shortcut Path: $ShortcutPath" "Gray"
        Write-Status "Batch File: $BatchFile" "Gray"
        Write-Status "Icon File: $IconFile" "Gray"
    }

    # Verify batch file exists
    if (-not (Test-Path $BatchFile)) {
        throw "PDF_Renamer.bat not found at: $BatchFile"
    }

    # Check for icon file
    $UseIcon = $null
    if (Test-Path $IconFile) {
        $UseIcon = $IconFile
        Write-Status "✓ Found primary icon: tools_icon.ico" "Green"
    } elseif (Test-Path $FallbackIconFile) {
        $UseIcon = $FallbackIconFile
        Write-Status "✓ Found fallback icon: tools_icon_hq.ico" "Green"
    } else {
        Write-Status "⚠ No custom icon found, using system default" "Yellow"
    }

    # Create shortcut
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    
    # Configure shortcut properties
    $Shortcut.TargetPath = $BatchFile
    $Shortcut.WorkingDirectory = $ScriptDir
    $Shortcut.Description = "PDF Renamer - AI-Powered PDF Title Extraction and Renaming Tool"
    $Shortcut.WindowStyle = 1  # Normal window
    
    # Set icon
    if ($UseIcon) {
        $Shortcut.IconLocation = $UseIcon
        Write-Status "✓ Custom icon applied" "Green"
    } else {
        # Use a nice system icon for PDF/document tools
        $Shortcut.IconLocation = "%SystemRoot%\System32\shell32.dll,71"
        Write-Status "✓ System document icon applied" "Green"
    }
    
    # Save the shortcut
    $Shortcut.Save()
    
    # Verify shortcut was created
    if (Test-Path $ShortcutPath) {
        Write-Status ""
        Write-Status "🎉 Desktop shortcut created successfully!" "Green"
        Write-Status "📍 Location: $ShortcutPath" "Cyan"
        Write-Status ""
        Write-Status "📋 Shortcut Details:" "White"
        Write-Status "   • Name: PDF Renamer" "Gray"
        Write-Status "   • Target: PDF_Renamer.bat" "Gray"
        Write-Status "   • Working Directory: $ScriptDir" "Gray"
        if ($UseIcon) {
            Write-Status "   • Icon: Custom tools icon" "Gray"
        } else {
            Write-Status "   • Icon: System document icon" "Gray"
        }
        Write-Status ""
        Write-Status "🚀 Usage Instructions:" "Yellow"
        Write-Status "   1. Double-click 'PDF Renamer' on your desktop to launch" "White"
        Write-Status "   2. While running, right-click taskbar icon → 'Pin to taskbar'" "White"
        Write-Status "   3. The icon will appear in your taskbar for quick access" "White"
        Write-Status ""
        
        # Additional taskbar pinning instructions
        if (-not (Test-Administrator)) {
            Write-Status "💡 Pro Tip: For best taskbar experience:" "Cyan"
            Write-Status "   • Launch the app once using the desktop shortcut" "White"
            Write-Status "   • Right-click the taskbar icon while it's running" "White"
            Write-Status "   • Select 'Pin to taskbar' for permanent quick access" "White"
            Write-Status ""
        }
        
    } else {
        throw "Failed to create shortcut at: $ShortcutPath"
    }

} catch {
    Write-Status ""
    Write-Status "❌ Error creating shortcut: $($_.Exception.Message)" "Red"
    Write-Status ""
    Write-Status "🔧 Troubleshooting:" "Yellow"
    Write-Status "   • Make sure you have write permissions to the desktop" "White"
    Write-Status "   • Try running as administrator if the issue persists" "White"
    Write-Status "   • Check that PDF_Renamer.bat exists in the pdf_renamer folder" "White"
    Write-Status ""
    exit 1
}

Write-Status "✨ Setup complete! Enjoy using PDF Renamer!" "Green"
Write-Status ""

# Wait for user input before closing
Read-Host "Press Enter to exit"