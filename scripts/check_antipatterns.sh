#!/bin/bash
# Prevent 'if not (x is not None):' double-negation anti-pattern
COUNT=$(grep -rn "if not (.* is not None):" src/shared/python --include="*.py" | wc -l)
if [ "$COUNT" -gt 0 ]; then
  echo "ERROR: Found $COUNT occurrences of 'if not (x is not None):' — use 'if x is None:' instead"
  grep -rn "if not (.* is not None):" src/shared/python --include="*.py"
  exit 1
fi
echo "No double-negation None checks found ✓"
