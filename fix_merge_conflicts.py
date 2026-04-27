import os

def process_file(filepath):
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r') as f:
        content = f.read()

    if '<<<<<<< HEAD' not in content:
        return

    print(f"Fixing {filepath}")

    lines = content.split('\n')
    new_lines = []

    in_head = False
    in_theirs = False

    for line in lines:
        if line.startswith('<<<<<<< HEAD'):
            in_head = True
            continue
        elif line.startswith('======='):
            in_head = False
            in_theirs = True
            continue
        elif line.startswith('>>>>>>> origin/main') or line.startswith('>>>>>>> origin/') or line.startswith('>>>>>>> '):
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
                if '<<<<<<< HEAD' in f.read():
                    process_file(filepath)
        except Exception:
            pass
