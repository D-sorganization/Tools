---
issue: 5871
summary: "Run the scipy.signal eager-import boundary check in a fresh interpreter so it no longer pops scipy.signal from sys.modules and breaks 17 nightly tests with _signal_api NameError"
---
