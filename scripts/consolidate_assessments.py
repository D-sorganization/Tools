import re
from pathlib import Path

ASSESSMENTS_DIR = Path("docs/assessments")
OUTPUT_MANUAL = ASSESSMENTS_DIR / "Manual_Assessment_Consolidated.md"
OUTPUT_AUTO_PREFIX = "Automated_Assessment_Log"


def parse_manual_file(filepath):
    content = filepath.read_text(encoding="utf-8")
    category_match = re.search(r"# Assessment: (.*?) \(Category ([A-Z])\)", content)
    if not category_match:
        return None

    title = category_match.group(1)
    category = category_match.group(2)

    grade_match = re.search(r"## Grade: ([\d\.]+) / 10", content)
    grade = grade_match.group(1) if grade_match else "N/A"

    # Extract sections
    sections = {}
    current_section = None
    lines = content.split("\n")
    section_content = []

    for line in lines:
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(section_content).strip()
            current_section = line.strip().replace("#", "").strip()
            section_content = []
        elif current_section:
            section_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(section_content).strip()

    return {"category": category, "title": title, "grade": grade, "sections": sections}


def consolidate_manual_assessments():
    manual_files = list(ASSESSMENTS_DIR.glob("Assessment_?_*.md"))
    # Filter out "Results" files
    manual_files = [f for f in manual_files if "Results" not in f.name]

    data = []
    for f in manual_files:
        parsed = parse_manual_file(f)
        if parsed:
            data.append(parsed)

    # Sort by category
    data.sort(key=lambda x: x["category"])

    with open(OUTPUT_MANUAL, "w", encoding="utf-8") as out:
        out.write("# Consolidated Manual Assessment Report\n\n")
        out.write("## Executive Summary\n\n")
        out.write("| Category | Topic | Grade | Status |\n")
        out.write("| :--- | :--- | :--- | :--- |\n")

        for item in data:
            out.write(
                f"| {item['category']} | {item['title']} | {item['grade']} | Manual |\n"
            )

        out.write("\n---\n\n")

        for item in data:
            out.write(f"## Category {item['category']}: {item['title']}\n\n")
            out.write(f"**Current Grade: {item['grade']} / 10**\n\n")

            for section_title, content in item["sections"].items():
                if section_title.startswith("Grade"):
                    continue
                out.write(f"### {section_title}\n\n")
                out.write(content + "\n\n")

            out.write("---\n\n")

    print(f"Generated {OUTPUT_MANUAL}")
    return manual_files


def consolidate_automated_results(date_str):
    files = list(ASSESSMENTS_DIR.glob(f"Assessment_*_Results_{date_str}.md"))

    entries = []

    for f in files:
        content = f.read_text(encoding="utf-8")
        # Extract Category and Score
        match = re.search(r"# Assessment ([A-Z]) Results", content)
        cat = match.group(1) if match else "?"

        score_match = re.search(r"Overall Score\*\*: \*\*([\d\.]+)/10", content)
        score = score_match.group(1) if score_match else "N/A"

        entries.append({"category": cat, "score": score})

    entries.sort(key=lambda x: x["category"])

    out_file = ASSESSMENTS_DIR / f"{OUTPUT_AUTO_PREFIX}_{date_str}.md"
    with open(out_file, "w", encoding="utf-8") as out:
        out.write(f"# Automated Assessment Log: {date_str}\n\n")
        out.write("| Category | Score |\n")
        out.write("| --- | --- |\n")
        for e in entries:
            out.write(f"| {e['category']} | {e['score']} |\n")

    print(f"Generated {out_file}")
    return files


if __name__ == "__main__":
    if not ASSESSMENTS_DIR.exists():
        print("Wrong directory")
    else:
        # Consolidate Manual
        files_to_delete = consolidate_manual_assessments()
        # Consolidate Auto 2026-01-23
        files_to_delete += consolidate_automated_results("2026-01-23")
        # Consolidate Auto 2026-01-22
        files_to_delete += consolidate_automated_results("2026-01-22")

        print("\nFiles to delete:")
        for f in files_to_delete:
            print(f)
