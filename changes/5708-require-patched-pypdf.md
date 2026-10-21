---
issue: 5708
summary: "Require patched pypdf 6.21.1 in PDF dependency declarations"
---

Raise the root and standalone PDF Renamer minimums to 6.21.1 and update the reproducible root lock. A real pypdf writer/reader contract covers PDF Renamer metadata extraction. This tracks Tools issue #5708; generated manual artifacts and their review records remain unchanged.
