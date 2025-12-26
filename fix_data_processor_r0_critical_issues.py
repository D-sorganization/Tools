#!/usr/bin/env python3
"""
Fix critical issues in Data_Processor_r0.py following AGENTS.md guidelines.

This script addresses:
1. Bare except clauses (E722) - Replace with specific exceptions
2. Critical line length issues (E501) - Break long lines strategically
3. Add logging instead of print statements where appropriate
4. Maintain functionality while improving code quality

Following AGENTS.md:
- Use specific exception handling instead of bare except
- Replace print statements with logging where appropriate
- Maintain code quality and security standards
"""

import logging
import re
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_bare_except_clauses(content: str) -> str:
    """Replace bare except clauses with specific exception handling."""
    logger.info("Fixing bare except clauses...")
    
    # Pattern 1: except: followed by pass
    content = re.sub(
        r'(\s+)except:\s*\n(\s+)pass',
        r'\1except Exception:\n\2pass',
        content
    )
    
    # Pattern 2: except: with other content
    content = re.sub(
        r'(\s+)except:\s*\n',
        r'\1except Exception:\n',
        content
    )
    
    return content

def fix_critical_line_lengths(content: str) -> str:
    """Fix the most critical line length issues."""
    logger.info("Fixing critical line length issues...")
    
    # Fix long docstrings
    content = re.sub(
        r'"""Fixed version with proper splitter implementation and all advanced features\."""',
        '"""Fixed version with proper splitter implementation and all advanced features."""',
        content
    )
    
    # Fix long differentiation docstring
    content = re.sub(
        r'"""Apply differentiation to selected signals with support for up to 5th order\."""',
        '"""Apply differentiation to selected signals with support for up to 5th order."""',
        content
    )
    
    # Fix long error messages - break them into multiple lines
    content = re.sub(
        r'f"Error in spline differentiation for \{signal\}, order \{order\}: \{e\}"',
        r'f"Error in spline differentiation for {signal}, "\n                                 f"order {order}: {e}"',
        content
    )
    
    content = re.sub(
        r'f"Error in polynomial differentiation for \{signal\}, order \{order\}: \{e\}"',
        r'f"Error in polynomial differentiation for {signal}, "\n                                 f"order {order}: {e}"',
        content
    )
    
    # Fix long status messages
    content = re.sub(
        r'f"Reading file \{i\+1\}/\{total_files\}: \{os\.path\.basename\(file_path\)\}"',
        r'f"Reading file {i+1}/{total_files}: "\n                        f"{os.path.basename(file_path)}"',
        content
    )
    
    content = re.sub(
        r'f"Ready - \{len\(self\.input_file_paths\)\} files loaded\. Go to Plotting tab to visualize\."',
        r'f"Ready - {len(self.input_file_paths)} files loaded. "\n                        f"Go to Plotting tab to visualize."',
        content
    )
    
    # Fix long processing messages
    content = re.sub(
        r'f"\\n--- Processing file \{i\+1\}/\{len\(self\.input_file_paths\)\}: \{os\.path\.basename\(file_path\)\} ---"',
        r'f"\\n--- Processing file {i+1}/{len(self.input_file_paths)}: "\n                f"{os.path.basename(file_path)} ---"',
        content
    )
    
    content = re.sub(
        r'f"Processing file \{i\+1\}/\{len\(self\.input_file_paths\)\}: \{os\.path\.basename\(file_path\)\}"',
        r'f"Processing file {i+1}/{len(self.input_file_paths)}: "\n                    f"{os.path.basename(file_path)}"',
        content
    )
    
    return content

def add_logging_imports(content: str) -> str:
    """Add logging import if not present."""
    if 'import logging' not in content:
        # Add logging import after other imports
        import_section = content.find('import pandas as pd')
        if import_section != -1:
            # Find the end of the import section
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import pandas as pd'):
                    # Insert logging import after pandas
                    lines.insert(i + 1, 'import logging')
                    content = '\n'.join(lines)
                    break
    
    return content

def add_logger_setup(content: str) -> str:
    """Add logger setup after imports."""
    if 'logger = logging.getLogger(__name__)' not in content:
        # Find the end of imports and add logger setup
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('# ============================================================================='):
                if 'WORKER FUNCTION' in lines[i + 1]:
                    # Insert logger setup before the worker function section
                    lines.insert(i, '')
                    lines.insert(i + 1, '# Set up logging')
                    lines.insert(i + 2, 'logger = logging.getLogger(__name__)')
                    lines.insert(i + 3, '')
                    content = '\n'.join(lines)
                    break
    
    return content

def replace_critical_print_statements(content: str) -> str:
    """Replace critical print statements with logging where appropriate."""
    logger.info("Replacing critical print statements with logging...")
    
    # Replace error print statements with logger.error
    content = re.sub(
        r'print\(f"Error processing \{file_path\}: \{e!\s\}"\)',
        r'logger.error(f"Error processing {file_path}: {e!s}")',
        content
    )
    
    # Replace debug print statements with logger.debug
    content = re.sub(
        r'print\("DEBUG: ([^"]+)"\)',
        r'logger.debug("DEBUG: \\1")',
        content
    )
    
    content = re.sub(
        r'print\(f"DEBUG: ([^"]+)"\)',
        r'logger.debug(f"DEBUG: \\1")',
        content
    )
    
    return content

def main():
    """Main function to fix critical issues in Data_Processor_r0.py."""
    file_path = Path("data_processing/data_processor/archive/Data_Processor_r0.py")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Starting fixes for {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    logger.info(f"Original file has {original_lines} lines")
    
    # Apply fixes
    content = add_logging_imports(content)
    content = add_logger_setup(content)
    content = fix_bare_except_clauses(content)
    content = fix_critical_line_lengths(content)
    content = replace_critical_print_statements(content)
    
    # Write the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    final_lines = len(content.split('\n'))
    logger.info(f"Fixed file has {final_lines} lines")
    logger.info(f"Applied critical fixes to {file_path}")
    
    # Verify the file still compiles
    try:
        import py_compile
        py_compile.compile(str(file_path), doraise=True)
        logger.info("✅ File compiles successfully after fixes")
    except py_compile.PyCompileError as e:
        logger.error(f"❌ Compilation error after fixes: {e}")
        return False
    
    logger.info("🎉 Critical fixes completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)