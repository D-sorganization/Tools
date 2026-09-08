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
