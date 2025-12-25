#!/usr/bin/env python3
"""
Fix B023 lambda binding issues specifically.
"""

import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_lambda_binding_in_file(file_path: str) -> bool:
    """Fix B023 lambda binding issues in a specific file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Fix lambda that captures processed_files and total_files from loop
        # Replace: lambda: self.folder_status_var.set(f"Processed {processed_files}/{total_files} files")
        # With: lambda p=processed_files, t=total_files: self.folder_status_var.set(f"Processed {p}/{t} files")
        
        pattern1 = r'lambda:\s*self\.folder_status_var\.set\(\s*f"Processed\s*\{processed_files\}/\{total_files\}\s*files"\s*\)'
        replacement1 = r'lambda p=processed_files, t=total_files: self.folder_status_var.set(f"Processed {p}/{t} files")'
        content = re.sub(pattern1, replacement1, content)
        
        # Pattern 2: Fix lambda that captures processed_files and total_files for analysis
        pattern2 = r'lambda:\s*self\.folder_status_var\.set\(\s*f"Analyzed\s*\{processed_files\}/\{total_files\}\s*files"\s*\)'
        replacement2 = r'lambda p=processed_files, t=total_files: self.folder_status_var.set(f"Analyzed {p}/{t} files")'
        content = re.sub(pattern2, replacement2, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed lambda binding issues in {file_path}")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error fixing lambda binding in {file_path}: {e}")
        return False

def main():
    """Main function."""
    file_path = "data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py"
    
    logger.info(f"Fixing lambda binding issues in {file_path}")
    
    if fix_lambda_binding_in_file(file_path):
        logger.info("Lambda binding fixes applied successfully")
    else:
        logger.info("No lambda binding fixes needed")

if __name__ == "__main__":
    main()