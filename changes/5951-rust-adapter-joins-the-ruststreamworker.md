---
issue: 5951
summary: "rust_adapter joins the _RustStreamWorker QThread after its event loop quits, so a still-running thread is never destroyed (merge_group abort)"
---
