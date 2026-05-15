# file_watcher

Cross-platform debounced file watcher with `.gitignore` filtering.

Built on [`notify-rs`](https://crates.io/crates/notify) v6 with the
[`ignore`](https://crates.io/crates/ignore) crate for gitignore matching.
Distributed as both a Rust crate (for in-tree consumers) and a Python wheel
(via [maturin](https://maturin.rs)).

## Why

The bare `notify` crate is great but leaves three jobs to the caller:

1. **Debouncing.** Editor saves emit a flurry of events (write, rename, temp
   delete). Without debouncing, downstream consumers (RAG indexer, project
   explorer) re-process the same file 5–10 times per save.
2. **Filtering.** Watching a project root naively floods you with events from
   `node_modules/`, `target/`, `__pycache__/`, etc.
3. **Cross-runtime ergonomics.** Python callers want a callback API, not a
   raw `std::sync::mpsc::Receiver`.

This crate handles all three.

## Usage (Rust)

```rust
use file_watcher::{FileWatcher, FileWatcherConfig};

let watcher = FileWatcher::new(FileWatcherConfig {
    root: "/path/to/project".into(),
    debounce_ms: 100,
    respect_gitignore: true,
});

watcher.on_change(|events| {
    for ev in events {
        println!("{:?} {}", ev.kind, ev.path.display());
    }
});

watcher.start().unwrap();
// ... do work ...
watcher.stop().unwrap();
```

## Usage (Python)

The Python wrapper at `src/shared/python/file_watcher/` prefers the Rust
extension and falls back to `watchdog` if the wheel is not built:

```python
from file_watcher import FileWatcher

watcher = FileWatcher(
    root="/path/to/project",
    debounce_ms=100,
    respect_gitignore=True,
)

@watcher.on_change
def handle(events):
    for ev in events:
        print(ev.path, ev.kind)  # "create" | "modify" | "delete" | "rename"

watcher.start()
# ... later
watcher.stop()
```

## Building the Python wheel

```bash
cd rust_core/file_watcher
maturin develop --features python   # editable install into the active venv
# or
maturin build --release --features python
```

If maturin is not available, the Python wrapper transparently falls back to
the `watchdog`-based implementation, so callers do not need the wheel built
to use the package.
