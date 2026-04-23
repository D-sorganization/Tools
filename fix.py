import re
from pathlib import Path

path = Path("src/shared/python/signal_toolkit/widget_processing.py")
content = path.read_text(encoding="utf-8")

# Add "WidgetProtocol" typing to all methods inside ProcessingMixin
content = re.sub(
    r"def ([a-zA-Z0-9_]+)\(\s*self\s*\)\s*(->[^:]+):",
    r'def \1(self: "WidgetProtocol") \2:',
    content
)

# And for methods with more arguments like def foo(self, x: int)
content = re.sub(
    r"def ([a-zA-Z0-9_]+)\(\n?\s*self,\n?\s*([^\)]+)\)\s*(->[^:]+):",
    r'def \1(self: "WidgetProtocol", \2) \3:',
    content
)

path.write_text(content, encoding="utf-8")
print("Done fixing widget_processing.py")
