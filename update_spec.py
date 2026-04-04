import re

with open('SPEC.md', 'r') as f:
    content = f.read()

# Update LAST UPDATED date to today's date (2026-04-04 from the memory)
updated_content = re.sub(
    r'LAST UPDATED: \d{4}-\d{2}-\d{2}',
    'LAST UPDATED: 2026-04-04',
    content
)

with open('SPEC.md', 'w') as f:
    f.write(updated_content)

print("Updated SPEC.md")
