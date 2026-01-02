' VBScript to create a desktop shortcut for PDF Renamer
' Run this script to create a desktop shortcut

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get the current script directory
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Get desktop path
strDesktop = objShell.SpecialFolders("Desktop")

' Create shortcut
Set objShortcut = objShell.CreateShortcut(strDesktop & "\PDF Renamer.lnk")

' Set shortcut properties
objShortcut.TargetPath = strScriptPath & "\PDF_Renamer.bat"
objShortcut.WorkingDirectory = strScriptPath
objShortcut.Description = "PDF Renamer - AI-Powered PDF Title Extraction and Renaming"
objShortcut.IconLocation = "shell32.dll,71"  ' PDF-like icon from Windows

' Save the shortcut
objShortcut.Save

' Notify user
MsgBox "Desktop shortcut created successfully!" & vbCrLf & vbCrLf & _
       "Shortcut location: " & strDesktop & "\PDF Renamer.lnk", _
       vbInformation, "PDF Renamer Shortcut Creator"
