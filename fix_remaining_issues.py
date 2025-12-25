#!/usr/bin/env python3
"""
Fix remaining linting issues in Data_Processor_r0.py
"""

import re
from pathlib import Path


def fix_remaining_long_lines(file_path: str) -> bool:
    """Fix remaining long lines that black couldn't handle"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    changes_made = False

    for _i, line in enumerate(lines):
        if len(line.rstrip()) <= 88:
            new_lines.append(line)
            continue

        # Fix long print statements with f-strings
        if line.strip().startswith("print(") and 'f"' in line:"
            indent = len(line) - len(line.lstrip())
            # Break long print statements
            if "DEBUG:" in line or "Error" in line or "Warning" in line:
                # Extract the f-string content
                match = re.search(r'f"([^"]*)"', line)"
                if match:
                    content_text = match.group(1)
                    # Break at logical points
                    for break_point in [
                        ": ",
                        " - ",
                        ", ",
                        " from ",
                        " to ",
                        " with ",
                        " for ",
                        " in ",
                        " of ",
                        " = ",
                    ]:
                        if break_point in content_text and len(content_text) > 60:
                            parts = content_text.split(break_point, 1)
                            if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                new_line1 = line.replace(
                                    f'f"{content_text}"',
                                    f'f"{parts[0]}{break_point.rstrip()}"',
                                )
                                new_line2 = " " * (indent + 4) + f'f"{parts[1]}",'
                                if line.endswith(","):
                                    new_line1 = new_line1.replace('",', '"')
                                    new_line2 = new_line2.replace(",", ",")
                                elif line.endswith(")"):
                                    new_line1 = new_line1.replace('")', '"')
                                    new_line2 = new_line2.replace(",", ")")
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

        # Fix long text= assignments
        elif "text=" in line and ('f"' in line or '"' in line):
            indent = len(line) - len(line.lstrip())
            # Look for text=f"..." or text="..." patterns
            match = re.search(r'text=f?"([^"]*)"', line)
            if match:
                content_text = match.group(1)
                if len(content_text) > 60:
                    # Break at logical points
                    for break_point in [
                        ": ",
                        " - ",
                        ", ",
                        " from ",
                        " to ",
                        " with ",
                        " for ",
                        " in ",
                        " of ",
                    ]:
                        if break_point in content_text:
                            parts = content_text.split(break_point, 1)
                            if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                is_fstring = 'text=f"' in line"
                                prefix = 'f"' if is_fstring else '"'
                                new_line1 = line.replace(
                                    f'{prefix}{content_text}"',
                                    f'{prefix}{parts[0]}{break_point.rstrip()}"',
                                )
                                new_line2 = " " * (indent + 4) + f'{prefix}{parts[1]}",'
                                if line.endswith(","):
                                    new_line1 = new_line1.replace('",', '"')
                                elif line.endswith(")"):
                                    new_line2 = new_line2.replace(",", ")")
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

        # Fix long comments
        elif line.strip().startswith("#") and len(line.rstrip()) > 88:
            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()

            # Break at logical points
            for break_point in [
                " - ",
                ": ",
                ", ",
                " and ",
                " or ",
                " but ",
                " with ",
                " for ",
                " to ",
                " from ",
            ]:
                if break_point in comment_text and len(comment_text) > 60:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        new_lines.append(
                            " " * indent + "# " + parts[0] + break_point.rstrip()
                        )
                        new_lines.append(" " * indent + "# " + parts[1])
                        changes_made = True
                        break
            else:
                new_lines.append(line)

        # Fix long string literals in other contexts
        elif '"' in line and len(line.rstrip()) > 88:
            # Try to break long strings at logical points
            if any(
                keyword in line
                for keyword in ["messagebox.", "success_message", "debug_text"]
            ):
                indent = len(line) - len(line.lstrip())
                # Find string content
                quote_matches = list(re.finditer(r'"([^"]*)"', line))
                if quote_matches:
                    for match in quote_matches:
                        content_text = match.group(1)
                        if len(content_text) > 60:
                            # Break at logical points
                            for break_point in [
                                ". ",
                                ": ",
                                ", ",
                                " - ",
                                " and ",
                                " or ",
                                " with ",
                                " for ",
                                " to ",
                                " from ",
                            ]:
                                if break_point in content_text:
                                    parts = content_text.split(break_point, 1)
                                    if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                        new_line1 = line.replace(
                                            f'"{content_text}"',
                                            f'"{parts[0]}{break_point.rstrip()}"',
                                        )
                                        new_line2 = " " * (indent + 4) + f'"{parts[1]}"'
                                        if line.endswith(","):
                                            new_line2 += ","
                                        elif line.endswith(")"):
                                            new_line2 += ")"
                                        new_lines.append(new_line1)
                                        new_lines.append(new_line2)
                                        changes_made = True
                                        break
                            else:
                                new_lines.append(line)
                            break
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
            f.write("\n".join(new_lines))
        print(f"Fixed remaining long lines in {file_path}")
        return True
    return False


def add_basic_type_annotations(file_path: str) -> bool:
    """Add basic type annotations to fix some of the simpler type issues"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    changes_made = False

    # Fix some basic type annotations
    replacements = [
        # Fix class variable type annotations
        ("self.splitters = {}", "self.splitters: dict[str, Any] = {}"),
        ("self.input_file_paths = []", "self.input_file_paths: list[str] = []"),
        (
            "self.loaded_data_cache = {}",
            "self.loaded_data_cache: dict[str, pd.DataFrame] = {}",
        ),
        (
            "self.processed_files = {}",
            "self.processed_files: dict[str, pd.DataFrame] = {}",
        ),
        ("self.signal_vars = {}", "self.signal_vars: dict[str, tk.BooleanVar] = {}"),
        (
            "self.plot_signal_vars = {}",
            "self.plot_signal_vars: dict[str, tk.BooleanVar] = {}",
        ),
        (
            "self.custom_vars_list = []",
            "self.custom_vars_list: list[dict[str, Any]] = []",
        ),
        (
            "self.reference_signal_widgets = {}",
            "self.reference_signal_widgets: dict[str, Any] = {}",
        ),
        ("self.dat_tag_vars = {}", "self.dat_tag_vars: dict[str, tk.BooleanVar] = {}"),
        ("self.plots_list = []", "self.plots_list: list[dict[str, Any]] = []"),
        ("self.saved_signal_list = []", "self.saved_signal_list: list[str] = []"),
        (
            "self.integrator_signal_vars = {}",
            "self.integrator_signal_vars: dict[str, tk.BooleanVar] = {}",
        ),
        (
            "self.deriv_signal_vars = {}",
            "self.deriv_signal_vars: dict[str, tk.BooleanVar] = {}",
        ),
        (
            "self.custom_legend_entries = {}",
            "self.custom_legend_entries: dict[str, str] = {}",
        ),
    ]

    for old, new in replacements:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes_made = True

    # Fix some function signatures
    function_fixes = [
        ("def get_deriv(w):", "def get_deriv(w: pd.Series) -> float:"),
        (
            "def process_single_csv_file(",
            "def process_single_csv_file(",
        ),  # Already has types
    ]

    for old, new in function_fixes:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes_made = True

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Added basic type annotations to {file_path}")
        return True
    return False


def main() -> None:
    """Fix remaining issues in Data_Processor_r0.py"""
    file_path = (
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )
    path = Path(file_path)
    if path.exists():
        print("Fixing remaining long lines...")
        fix_remaining_long_lines(str(path))
        print("Adding basic type annotations...")
        add_basic_type_annotations(str(path))
    else:
        print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
