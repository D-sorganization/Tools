---
issue: 6023
summary: "Yield stable absolute paths from relative directory contexts"
---

Resolve change_directory targets before changing cwd so the yielded Path continues to identify the entered directory. Add regressions for relative strings and Path inputs, parent navigation, nesting, exception restoration, and symlinks.
