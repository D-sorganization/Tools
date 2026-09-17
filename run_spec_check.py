import re
from shared_scripts.spec_changelog import _failures
with open("SPEC.md") as f:
    text = f.read()
failures = _failures(None, text) # wait, _failures takes checker, spec_text... wait I can just import parse_changelog
print(failures)
