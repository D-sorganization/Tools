#!/usr/bin/env python3
import subprocess
import sys
import shutil
from pathlib import Path

def print_step(message):
    print(f"\n\033[1;34m[SETUP] {message}\033[0m")

def check_python():
    print_step("Checking Python environment...")
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("\033[1;31mError: Python 3.10+ is required.\033[0m")
        sys.exit(1)

def install_python_deps():
    print_step("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_node_deps():
    print_step("Installing Node.js dependencies...")

    # Check for pnpm
    pnpm_path = shutil.which("pnpm")
    if not pnpm_path:
        print("\033[1;33mWarning: 'pnpm' not found. Attempting to install via npm...\033[0m")
        npm_path = shutil.which("npm")
        if not npm_path:
             print("\033[1;31mError: neither 'pnpm' nor 'npm' found. Node.js dependencies skipped.\033[0m")
             return
        try:
            subprocess.check_call(["npm", "install", "-g", "pnpm"])
        except subprocess.CalledProcessError:
             print("\033[1;31mError: Failed to install pnpm globally. Please install it manually.\033[0m")
             return

    unit_converter_path = Path("web_applications/unit_converter")
    if unit_converter_path.exists():
        print(f"Installing dependencies in {unit_converter_path}...")
        try:
            subprocess.check_call(["pnpm", "install"], cwd=unit_converter_path)
        except subprocess.CalledProcessError:
             print("\033[1;31mError: Failed to install dependencies in unit_converter.\033[0m")
    else:
        print(f"Path not found: {unit_converter_path}")

def main():
    try:
        check_python()
        install_python_deps()
        install_node_deps()
        print_step("Setup complete! You are ready to go.")
    except Exception as e:
        print(f"\n\033[1;31mSetup failed: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
