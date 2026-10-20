---
issue: 5680
summary: "Initialize and retain the ORT environment required by local embeddings"
---

After the paired ORT rc.14 dependency update in #5648 reached main, the
local-embeddings wheel feature set still failed with E0061 because session
construction lacked an Environment. Initialize one retained, mutex-protected
environment, reuse its clones for sessions, and retain retry on initialization
failure. The exact feature compile reproduces the defect before this repair.
