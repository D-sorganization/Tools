import re
from pathlib import Path

path = Path("src/shared/python/signal_toolkit/widget_processing.py")
content = path.read_text(encoding="utf-8")

# In the methods of ProcessingMixin, the author already wrote w = cast(WidgetProtocol, self)
# But forgot to use w instead of self.
# So we just replace self._update_plot, self._log, self._update_secondary_plot, self._update_frequency_response_plot with w.

replacements = [
    (r"self\._update_plot\(", r"w._update_plot("),
    (r"self\._log\(", r"w._log("),
    (r"self\._update_secondary_plot\(", r"w._update_secondary_plot("),
    (r"self\._update_frequency_response_plot\(", r"w._update_frequency_response_plot("),
    (r"self\.signal_generated\.emit\(", r"w.signal_generated.emit("),
    (r"self\.signal_updated\.emit\(", r"w.signal_updated.emit("),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# But wait, what if 'w' is not defined in some methods?
# Let's ensure 'w = cast(WidgetProtocol, self)' is there if 'w.' is used
# Or we can just use cast(WidgetProtocol, self)._update_plot

path.write_text(content, encoding="utf-8")
print("Done fixing self to w in widget_processing.py")
