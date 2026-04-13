import os


def fix_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    new_lines = []
    for line in content.splitlines():
        if line.startswith("<<<<<<<"):
            continue
        elif line.startswith("======="):
            continue
        elif line.startswith(">>>>>>>"):
            continue
        else:
            new_lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


fix_file("SPEC.md")
fix_file(".jules/bolt.md")
