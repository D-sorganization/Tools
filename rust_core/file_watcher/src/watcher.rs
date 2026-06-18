//! Core debounced, gitignore-aware file watcher.
//!
//! Design:
//! - A `notify::RecommendedWatcher` runs on its own thread (managed by notify).
//! - Raw events flow into a crossbeam channel.
//! - A debounce thread drains the channel, accumulates events into a `HashMap`
//!   keyed by `(PathBuf, ChangeKind)`, and flushes them to the user-supplied
//!   callback after `debounce_ms` of quiet.
//! - A `Gitignore` matcher (built once at start) filters events before they
//!   reach the debouncer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

/// What happened to a path. Mirrors the four high-level operations the rest of
/// the system cares about; finer-grained `notify` event types collapse into
/// these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeKind {
    Create,
    Modify,
    Delete,
    Rename,
}

impl ChangeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeKind::Create => "create",
            ChangeKind::Modify => "modify",
            ChangeKind::Delete => "delete",
            ChangeKind::Rename => "rename",
        }
    }
}

/// A single coalesced filesystem event.
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub path: PathBuf,
    pub kind: ChangeKind,
}

/// Watcher configuration. Construct via `FileWatcher::new`.
#[derive(Debug, Clone)]
pub struct FileWatcherConfig {
    pub root: PathBuf,
    pub debounce_ms: u64,
    pub respect_gitignore: bool,
}

#[derive(Debug)]
pub enum WatcherError {
    Notify(notify::Error),
    AlreadyStarted,
    NotStarted,
    Io(std::io::Error),
}

impl std::fmt::Display for WatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatcherError::Notify(e) => write!(f, "notify error: {e}"),
            WatcherError::AlreadyStarted => write!(f, "watcher already started"),
            WatcherError::NotStarted => write!(f, "watcher not started"),
            WatcherError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for WatcherError {}

impl From<notify::Error> for WatcherError {
    fn from(e: notify::Error) -> Self {
        WatcherError::Notify(e)
    }
}

impl From<std::io::Error> for WatcherError {
    fn from(e: std::io::Error) -> Self {
        WatcherError::Io(e)
    }
}

type Callback = Arc<dyn Fn(Vec<ChangeEvent>) + Send + Sync + 'static>;

/// Acquire a mutex, recovering the guard if the lock was poisoned by a panic on
/// another thread.
///
/// The watcher's `callback`/`state` mutexes only guard `Option<…>` slots — a
/// panic while one is held leaves the inner value structurally valid. Recovering
/// via `into_inner()` keeps event delivery alive instead of letting a single
/// panic poison the mutex and cascade every later `start`/`stop`/`on_change`
/// into a panic (issue #3556).
fn lock_poison_tolerant<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Cross-platform debounced file watcher.
pub struct FileWatcher {
    config: FileWatcherConfig,
    callback: Mutex<Option<Callback>>,
    state: Mutex<Option<RunningState>>,
}

struct RunningState {
    _notify_watcher: RecommendedWatcher,
    stop_flag: Arc<AtomicBool>,
    debounce_thread: Option<JoinHandle<()>>,
    _event_tx: Sender<Event>,
}

impl FileWatcher {
    pub fn new(config: FileWatcherConfig) -> Self {
        Self {
            config,
            callback: Mutex::new(None),
            state: Mutex::new(None),
        }
    }

    /// Register a callback. Replaces any previous callback. May be called
    /// before or after `start()`.
    pub fn on_change<F>(&self, callback: F)
    where
        F: Fn(Vec<ChangeEvent>) + Send + Sync + 'static,
    {
        *lock_poison_tolerant(&self.callback) = Some(Arc::new(callback));
    }

    /// Start watching. Returns `AlreadyStarted` if already running.
    pub fn start(&self) -> Result<(), WatcherError> {
        let mut state_guard = lock_poison_tolerant(&self.state);
        if state_guard.is_some() {
            return Err(WatcherError::AlreadyStarted);
        }

        let (tx, rx) = bounded::<Event>(1024);
        let tx_for_watcher = tx.clone();

        let mut notify_watcher: RecommendedWatcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    // Drop on full channel rather than blocking the OS thread.
                    let _ = tx_for_watcher.try_send(event);
                }
            })?;

        notify_watcher.watch(&self.config.root, RecursiveMode::Recursive)?;

        let stop_flag = Arc::new(AtomicBool::new(false));
        let debounce_thread = spawn_debounce_thread(
            rx,
            stop_flag.clone(),
            self.config.clone(),
            lock_poison_tolerant(&self.callback).clone(),
        );

        *state_guard = Some(RunningState {
            _notify_watcher: notify_watcher,
            stop_flag,
            debounce_thread: Some(debounce_thread),
            _event_tx: tx,
        });
        Ok(())
    }

    /// Stop watching. Idempotent after first call returns Ok.
    pub fn stop(&self) -> Result<(), WatcherError> {
        let mut state_guard = lock_poison_tolerant(&self.state);
        let Some(mut state) = state_guard.take() else {
            return Err(WatcherError::NotStarted);
        };
        state.stop_flag.store(true, Ordering::SeqCst);
        // Dropping `_notify_watcher` and `_event_tx` closes the channel, which
        // wakes the debounce thread.
        drop(state._notify_watcher);
        drop(state._event_tx);
        if let Some(handle) = state.debounce_thread.take() {
            let _ = handle.join();
        }
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        lock_poison_tolerant(&self.state).is_some()
    }

    pub fn root(&self) -> &Path {
        &self.config.root
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn build_gitignore(root: &Path) -> Gitignore {
    let mut builder = GitignoreBuilder::new(root);
    let candidate = root.join(".gitignore");
    if candidate.exists() {
        let _ = builder.add(&candidate);
    }
    builder.build().unwrap_or_else(|_| Gitignore::empty())
}

fn classify(kind: &EventKind) -> Option<ChangeKind> {
    use notify::event::{ModifyKind, RenameMode};
    match kind {
        EventKind::Create(_) => Some(ChangeKind::Create),
        EventKind::Remove(_) => Some(ChangeKind::Delete),
        EventKind::Modify(ModifyKind::Name(RenameMode::Both))
        | EventKind::Modify(ModifyKind::Name(RenameMode::From))
        | EventKind::Modify(ModifyKind::Name(RenameMode::To)) => Some(ChangeKind::Rename),
        EventKind::Modify(_) => Some(ChangeKind::Modify),
        _ => None,
    }
}

fn spawn_debounce_thread(
    rx: Receiver<Event>,
    stop_flag: Arc<AtomicBool>,
    config: FileWatcherConfig,
    callback: Option<Callback>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let gitignore = if config.respect_gitignore {
            Some(build_gitignore(&config.root))
        } else {
            None
        };
        let debounce = Duration::from_millis(config.debounce_ms);
        let mut pending: HashMap<(PathBuf, ChangeKind), ChangeEvent> = HashMap::new();
        let mut last_event_at: Option<Instant> = None;
        // Use a short poll interval so the loop wakes promptly to flush pending
        // events even if no new events arrive on the channel.
        let poll = Duration::from_millis(20);

        loop {
            if stop_flag.load(Ordering::SeqCst) {
                break;
            }

            match rx.recv_timeout(poll) {
                Ok(event) => {
                    let Some(kind) = classify(&event.kind) else {
                        continue;
                    };
                    for path in event.paths {
                        if should_ignore(&path, &config.root, gitignore.as_ref()) {
                            continue;
                        }
                        pending.insert((path.clone(), kind), ChangeEvent { path, kind });
                    }
                    last_event_at = Some(Instant::now());
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Fall through to flush check.
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }

            if let Some(t) = last_event_at {
                if !pending.is_empty() && t.elapsed() >= debounce {
                    let batch: Vec<ChangeEvent> = pending.drain().map(|(_, v)| v).collect();
                    last_event_at = None;
                    if let Some(cb) = callback.as_ref() {
                        cb(batch);
                    }
                }
            }
        }

        // Final flush on shutdown.
        if !pending.is_empty() {
            let batch: Vec<ChangeEvent> = pending.drain().map(|(_, v)| v).collect();
            if let Some(cb) = callback.as_ref() {
                cb(batch);
            }
        }
    })
}

fn should_ignore(path: &Path, root: &Path, gitignore: Option<&Gitignore>) -> bool {
    // Always skip the .git directory itself.
    let rel = path.strip_prefix(root).unwrap_or(path);
    if rel.components().any(|c| c.as_os_str() == ".git") {
        return true;
    }
    let Some(gi) = gitignore else { return false };
    let is_dir = path.is_dir();
    gi.matched_path_or_any_parents(path, is_dir).is_ignore()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;
    use tempfile::tempdir;

    fn collect_events(watcher: &FileWatcher) -> Arc<Mutex<Vec<ChangeEvent>>> {
        let bucket: Arc<Mutex<Vec<ChangeEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let bucket_cb = bucket.clone();
        watcher.on_change(move |events| {
            bucket_cb.lock().unwrap().extend(events);
        });
        bucket
    }

    #[test]
    fn detects_create_event() {
        let dir = tempdir().unwrap();
        let watcher = FileWatcher::new(FileWatcherConfig {
            root: dir.path().to_path_buf(),
            debounce_ms: 50,
            respect_gitignore: false,
        });
        let bucket = collect_events(&watcher);
        watcher.start().unwrap();

        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(dir.path().join("hello.txt"), b"hi").unwrap();
        std::thread::sleep(Duration::from_millis(400));

        watcher.stop().unwrap();
        let events = bucket.lock().unwrap();
        assert!(
            events.iter().any(|e| e.path.ends_with("hello.txt")),
            "expected create event for hello.txt, got: {events:?}"
        );
    }

    #[test]
    fn debounces_rapid_changes() {
        let dir = tempdir().unwrap();
        // Use a 500 ms debounce window so that all writes land inside one
        // window even on a heavily-loaded CI runner where sleep(5ms) can
        // stretch to 100+ ms per iteration.
        let watcher = FileWatcher::new(FileWatcherConfig {
            root: dir.path().to_path_buf(),
            debounce_ms: 500,
            respect_gitignore: false,
        });
        let call_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let cc = call_count.clone();
        watcher.on_change(move |_| {
            *cc.lock().unwrap() += 1;
        });
        watcher.start().unwrap();
        std::thread::sleep(Duration::from_millis(50));

        let path = dir.path().join("rapid.txt");
        for i in 0..10 {
            std::fs::write(&path, format!("v{i}")).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        // Wait long enough for the debounce window to close and the callback
        // to fire, even after timing variation on the CI runner.
        std::thread::sleep(Duration::from_millis(2000));
        watcher.stop().unwrap();

        // Debounce should collapse the burst to a single (or at most a small
        // handful of) callback invocations.
        let count = *call_count.lock().unwrap();
        assert!(
            count <= 3,
            "expected debounce to coalesce, got {count} flushes"
        );
    }

    #[test]
    fn detects_create_modify_delete_burst() {
        // Exercise the full create → modify → delete lifecycle in one burst and
        // assert the coalesced batch carries the distinct change kinds (#3556).
        let dir = tempdir().unwrap();
        let watcher = FileWatcher::new(FileWatcherConfig {
            root: dir.path().to_path_buf(),
            debounce_ms: 300,
            respect_gitignore: false,
        });
        let bucket = collect_events(&watcher);
        watcher.start().unwrap();
        std::thread::sleep(Duration::from_millis(100));

        let path = dir.path().join("burst.txt");
        std::fs::write(&path, b"v1").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, b"v2-modified").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::remove_file(&path).unwrap();

        std::thread::sleep(Duration::from_millis(1500));
        watcher.stop().unwrap();

        let events = bucket.lock().unwrap();
        assert!(
            events.iter().any(|e| e.path.ends_with("burst.txt")),
            "expected events for burst.txt, got: {events:?}"
        );
        // The final state is a delete; the OS may or may not surface every
        // intermediate kind, but a delete must be observed.
        assert!(
            events
                .iter()
                .any(|e| e.path.ends_with("burst.txt") && e.kind == ChangeKind::Delete),
            "expected a delete event for burst.txt, got: {events:?}"
        );
    }

    #[test]
    fn gitignore_filters_ignored_paths() {
        // A path matched by .gitignore must NOT be delivered, while a
        // non-ignored sibling must be (#3556).
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), b"ignored.log\n").unwrap();

        let watcher = FileWatcher::new(FileWatcherConfig {
            root: dir.path().to_path_buf(),
            debounce_ms: 100,
            respect_gitignore: true,
        });
        let bucket = collect_events(&watcher);
        watcher.start().unwrap();
        std::thread::sleep(Duration::from_millis(100));

        std::fs::write(dir.path().join("ignored.log"), b"noise").unwrap();
        std::fs::write(dir.path().join("kept.txt"), b"signal").unwrap();
        std::thread::sleep(Duration::from_millis(600));
        watcher.stop().unwrap();

        let events = bucket.lock().unwrap();
        assert!(
            events.iter().any(|e| e.path.ends_with("kept.txt")),
            "expected non-ignored kept.txt to be delivered, got: {events:?}"
        );
        assert!(
            !events.iter().any(|e| e.path.ends_with("ignored.log")),
            "expected ignored.log to be filtered out, got: {events:?}"
        );
    }

    #[test]
    fn gitignore_filter_is_applied_only_when_enabled() {
        // With respect_gitignore = false, the same .gitignore entry must NOT
        // suppress the event — proves the toggle is wired through.
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), b"ignored.log\n").unwrap();

        let watcher = FileWatcher::new(FileWatcherConfig {
            root: dir.path().to_path_buf(),
            debounce_ms: 100,
            respect_gitignore: false,
        });
        let bucket = collect_events(&watcher);
        watcher.start().unwrap();
        std::thread::sleep(Duration::from_millis(100));

        std::fs::write(dir.path().join("ignored.log"), b"noise").unwrap();
        std::thread::sleep(Duration::from_millis(600));
        watcher.stop().unwrap();

        let events = bucket.lock().unwrap();
        assert!(
            events.iter().any(|e| e.path.ends_with("ignored.log")),
            "with gitignore disabled, ignored.log should be delivered, got: {events:?}"
        );
    }
}
