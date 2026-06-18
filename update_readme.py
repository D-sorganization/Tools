import re

with open("docs/assessments/README.md", "r") as f:
    content = f.read()

# Append the new version history row
new_row = "| 6.2     | 2026-06 | Executed Completist Audit (Jun 18)                                                                      |\n"
content = content.replace(
    "| 6.1     | 2026-06 | Generated comprehensive assessment reports for 2026-06-11                                               |\n",
    "| 6.1     | 2026-06 | Generated comprehensive assessment reports for 2026-06-11                                               |\n" + new_row
)

with open("docs/assessments/README.md", "w") as f:
    f.write(content)
