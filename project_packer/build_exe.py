#!/usr/bin/env python3
"""
Build script for Folder Packer GUI executable
Creates a standalone Windows executable using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """Build the executable using PyInstaller"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Check if required files exist
    main_script = current_dir / "folder_packer_gui.py"
    icon_file = current_dir / "folder_icon.jpg"
    
    if not main_script.exists():
        print("❌ Error: folder_packer_gui.py not found!")
        return False
    
    # Convert JPG icon to ICO if needed
    ico_file = current_dir / "folder_icon.ico"
    if icon_file.exists() and not ico_file.exists():
        try:
            from PIL import Image
            # Load and convert icon
            img = Image.open(icon_file)
            # Resize to common icon sizes
            sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            img.save(ico_file, format='ICO', sizes=sizes)
            print(f"✓ Converted {icon_file.name} to {ico_file.name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not convert icon: {e}")
            ico_file = None
    
    # Build PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # No console window
        "--name", "FolderPacker",
        "--distpath", str(current_dir / "dist"),
        "--workpath", str(current_dir / "build"),
        "--specpath", str(current_dir),
    ]
    
    # Add icon if available
    if ico_file and ico_file.exists():
        cmd.extend(["--icon", str(ico_file)])
    
    # Add the main script
    cmd.append(str(main_script))
    
    print("🔨 Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            exe_path = current_dir / "dist" / "FolderPacker.exe"
            if exe_path.exists():
                print(f"✅ Successfully built executable: {exe_path}")
                print(f"📁 File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
                return True
            else:
                print("❌ Build completed but executable not found!")
                return False
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ PyInstaller not found! Please install it with: pip install pyinstaller")
        return False
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def clean_build():
    """Clean build artifacts"""
    current_dir = Path(__file__).parent
    
    # Remove build directories
    for folder in ["build", "dist", "__pycache__"]:
        folder_path = current_dir / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"🧹 Cleaned {folder}")
    
    # Remove spec file
    spec_file = current_dir / "FolderPacker.spec"
    if spec_file.exists():
        spec_file.unlink()
        print(f"🧹 Cleaned {spec_file.name}")

def main():
    """Main function"""
    print("📦 Folder Packer - Executable Builder")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_build()
        return
    
    # Check if PyInstaller is installed
    try:
        subprocess.run(["pyinstaller", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ PyInstaller not found!")
        print("📥 Installing PyInstaller...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            print("✅ PyInstaller installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyInstaller: {e}")
            return
    
    # Check if Pillow is installed (for icon conversion)
    try:
        import PIL
    except ImportError:
        print("📥 Installing Pillow for icon support...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"], check=True)
            print("✅ Pillow installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Pillow: {e}")
    
    # Build the executable
    if build_executable():
        print("\n🎉 Build completed successfully!")
        print("📁 The executable is located in the 'dist' folder")
    else:
        print("\n💥 Build failed!")

if __name__ == "__main__":
    main()
