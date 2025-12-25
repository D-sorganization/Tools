#!/usr/bin/env python3
"""
Comprehensive fix for remaining linting issues in Data_Processor_r0.py
"""

import re
from pathlib import Path


def fix_comprehensive_issues(file_path: str) -> bool:
    """Fix comprehensive linting issues"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    changes_made = False

    # Fix duplicate plot_signal_vars definitions
    # Remove the duplicate line that redefines plot_signal_vars
    duplicate_patterns = [
        (
            r"\s+# Initialize plot signal variables.*\n\s+self\.plot_signal_vars = {}",
            "",
        ),
        (
            r"\s+# Custom legend entries for plots\n\s+self\.custom_legend_entries = {}",
            "",
        ),
        (r"\s+self\.plots_list = \[\]$", ""),  # Remove duplicate plots_list at end
    ]

    for pattern, replacement in duplicate_patterns:
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content != content:
            content = new_content
            changes_made = True

    # Fix the remaining plot_signal_vars type annotation issue
    # Find all instances and fix the type
    if "self.plot_signal_vars: dict[str, tk.BooleanVar] = {}" in content:
        content = content.replace(
            "self.plot_signal_vars: dict[str, tk.BooleanVar] = {}",
            "self.plot_signal_vars: dict[str, dict[str, Any]] = {}",
        )
        changes_made = True

    # Fix missing function type annotations
    function_fixes = [
        (
            "def _filter_signals(self, event):",
            "def _filter_signals(self, event: Any) -> None:",
        ),
        ("def _clear_search(self):", "def _clear_search(self) -> None:"),
        ("def _on_bulk_mode_change(self):", "def _on_bulk_mode_change(self) -> None:"),
        (
            "def _on_dataset_naming_change(self):",
            "def _on_dataset_naming_change(self) -> None:",
        ),
        (
            "def _on_window_configure(self, event):",
            "def _on_window_configure(self, event: Any) -> None:",
        ),
        (
            "def _on_canvas_configure(self, event):",
            "def _on_canvas_configure(self, event: Any) -> None:",
        ),
        (
            "def _on_mousewheel(self, event):",
            "def _on_mousewheel(self, event: Any) -> None:",
        ),
        (
            "def _on_plot_canvas_configure(self, event):",
            "def _on_plot_canvas_configure(self, event: Any) -> None:",
        ),
        (
            "def _on_plot_mousewheel(self, event):",
            "def _on_plot_mousewheel(self, event: Any) -> None:",
        ),
        (
            "def _on_plots_list_select(self, event):",
            "def _on_plots_list_select(self, event: Any) -> None:",
        ),
    ]

    for old, new in function_fixes:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes_made = True

    # Fix some specific type annotation issues
    type_fixes = [
        ("all_signals = set()", "all_signals: set[str] = set()"),
        ("last_plotted_signals = set()", "last_plotted_signals: set[str] = set()"),
        ("plots_signal_vars = {}", "plots_signal_vars: dict[str, dict[str, Any]] = {}"),
    ]

    for old, new in type_fixes:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes_made = True

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed comprehensive issues in {file_path}")
        return True
    return False


def fix_remaining_long_lines_manual(file_path: str) -> bool:
    """Manually fix the most problematic long lines"""
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    changes_made = False
    new_lines = []

    for _i, line in enumerate(lines):
        if len(line.rstrip()) > 88:
            # Fix specific long lines that are commonly flagged in PRs
            if 'f"DEBUG:' in line and len(line.rstrip()) > 88:
                # Break debug print statements
                indent = len(line) - len(line.lstrip())
                if "print(" in line and line.strip().endswith(","):
                    # Find the f-string content
                    match = re.search(r'f"([^"]*)"', line)
                    if match:
                        content_text = match.group(1)
                        if len(content_text) > 60:
                            # Break at colon or equals
                            for break_point in [": ", " = ", " - ", ", "]:
                                if break_point in content_text:
                                    parts = content_text.split(break_point, 1)
                                    if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                        new_line1 = line.replace(
                                            f'f"{content_text}"',
                                            f'f"{parts[0]}{break_point.rstrip()}"',
                                        )
                                        new_line2 = (
                                            " " * (indent + 4) + f'f"{parts[1]}",'
                                        )
                                        new_lines.append(new_line1)
                                        new_lines.append(new_line2)
                                        changes_made = True
                                        break
                            else:
                                new_lines.append(line)
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            elif (
                "text=" in line
                and ('f"' in line or '"' in line)
                and len(line.rstrip()) > 88
            ):
                # Fix long text assignments
                indent = len(line) - len(line.lstrip())
                match = re.search(r'text=f?"([^"]*)"', line)
                if match:
                    content_text = match.group(1)
                    if len(content_text) > 60:
                        for break_point in [
                            ": ",
                            " - ",
                            ", ",
                            " from ",
                            " to ",
                            " with ",
                        ]:
                            if break_point in content_text:
                                parts = content_text.split(break_point, 1)
                                if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                    is_fstring = 'text=f"' in line
                                    prefix = 'f"' if is_fstring else '"'
                                    new_line1 = line.replace(
                                        f'{prefix}{content_text}"',
                                        f'{prefix}{parts[0]}{break_point.rstrip()}"',
                                    )
                                    new_line2 = (
                                        " " * (indent + 4) + f'{prefix}{parts[1]}"'
                                    )
                                    if line.rstrip().endswith(","):
                                        new_line2 += ","
                                    elif line.rstrip().endswith(")"):
                                        new_line2 += ")"
                                    new_lines.append(new_line1)
                                    new_lines.append(new_line2)
                                    changes_made = True
                                    break
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Fixed remaining long lines in {file_path}")
        return True
    return False


def main() -> None:
    """Fix comprehensive issues in Data_Processor_r0.py"""
    file_path = (
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )
    path = Path(file_path)
    if path.exists():
        print("Fixing comprehensive issues...")
        fix_comprehensive_issues(str(path))
        print("Fixing remaining long lines...")
        fix_remaining_long_lines_manual(str(path))
    else:
        print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
