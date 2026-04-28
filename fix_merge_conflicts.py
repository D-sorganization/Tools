import os
import re

def process_file(filepath):
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r') as f:
        content = f.read()

    HEAD_MARKER = '<' * 7 + ' HEAD'
    SEP_MARKER = '=' * 7
    ORIGIN_MAIN_MARKER = '>' * 7 + ' origin/main'
    ORIGIN_SLASH_MARKER = '>' * 7 + ' origin/'
    GENERIC_END_MARKER = '>' * 7 + ' '

    if HEAD_MARKER not in content:
        return

    print(f"Fixing {filepath}")

    lines = content.split('\n')
    new_lines = []

    in_head = False
    in_theirs = False

    for line in lines:
        if line.startswith(HEAD_MARKER):
            in_head = True
            continue
        elif line.startswith(SEP_MARKER):
            in_head = False
            in_theirs = True
            continue
        elif line.startswith(ORIGIN_MAIN_MARKER) or line.startswith(ORIGIN_SLASH_MARKER):
            in_theirs = False
            continue
        elif line.startswith(GENERIC_END_MARKER):
            in_theirs = False
            continue

        if in_theirs:
            continue

        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))

for root, _, files in os.walk('.'):
    if '.git' in root:
        continue
    for file in files:
        filepath = os.path.join(root, file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if HEAD_MARKER in content:
                    process_file(filepath)
        except Exception:
            pass