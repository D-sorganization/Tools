#!/usr/bin/env python3
"""
Fix the final 6 line length issues to complete CI/CD compliance.
"""

from pathlib import Path


def fix_all_remaining_line_lengths():
    """Fix all remaining line length issues."""

    # Fix Data_Processor_Integrated.py issues
    file_path = Path("data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py")
    if file_path.exists():
        content = file_path.read_text(encoding='utf-8')

        # Fix line 1612 - lambda expression
        content = content.replace(
            'self.folder_status_var.set(f"Processed {p}/{t} files")',
            'self.folder_status_var.set(\n                                        f"Processed {p}/{t} files"\n                                    )'
        )

        # Fix line 2215 - long comment
        content = content.replace(
            'Transform raw CSV time series data into processed, analyzed, and visualized datasets with',
            'Transform raw CSV time series data into processed, analyzed,\n    and visualized datasets with'
        )

        # Fix line 2261 - bullet point
        content = content.replace(
            '- **🔢 Finite Difference**: Direct numerical differentiation (forward, backward, central)',
            '- **🔢 Finite Difference**: Direct numerical differentiation\n  (forward, backward, central)'
        )

        # Fix line 2714 - long sentence
        content = content.replace(
            'work together seamlessly while maintaining the full functionality of the original standalone applications.',
            'work together seamlessly while maintaining the full functionality\n    of the original standalone applications.'
        )

        file_path.write_text(content, encoding='utf-8')
        print(f"Fixed 4 line length issues in {file_path}")

    # Fix Data_Processor_r0.py issues
    file_path = Path("data_processing/data_processor/python/data_processor/Data_Processor_r0.py")
    if file_path.exists():
        content = file_path.read_text(encoding='utf-8')

        # Fix line 7345 - long comment
        content = content.replace(
            'This application provides comprehensive tools for processing, analyzing, and visualizing time series data from CSV files',
            'This application provides comprehensive tools for processing,\n# analyzing, and visualizing time series data from CSV files'
        )

        # Fix line 7525 - long sentence
        content = content.replace(
            'For additional support or feature requests, please refer to the application documentation or contact the development team.',
            'For additional support or feature requests, please refer to the\n# application documentation or contact the development team.'
        )

        file_path.write_text(content, encoding='utf-8')
        print(f"Fixed 2 line length issues in {file_path}")


if __name__ == "__main__":
    fix_all_remaining_line_lengths()
    print("✅ All remaining line length issues fixed!")
