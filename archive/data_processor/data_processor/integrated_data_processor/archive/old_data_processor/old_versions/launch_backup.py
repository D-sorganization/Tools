#!/usr/bin/env python3
"""
Launch script for the backup version of the data processor.
This launches the version from BEFORE the 2025-01-27 changes.
"""

import subprocess
import sys
import os

def main():
    print("Starting Advanced CSV Processor (BACKUP VERSION)...")
    print("⚠️  This is the backup version from BEFORE 2025-01-27 changes")
    print("🚀 Launching backup application...")
    
    try:
        # Launch the backup version
        subprocess.run([sys.executable, "Data_Processor_BACKUP_BEFORE_2025-01-27_CHANGES.py"], check=True)
        print("✅ Backup application launched successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching backup application: {e}")
    except FileNotFoundError:
        print("❌ Backup file not found: Data_Processor_BACKUP_BEFORE_2025-01-27_CHANGES.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 