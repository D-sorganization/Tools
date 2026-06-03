from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from file_watcher import _fallback
from file_watcher._fallback import ChangeEvent, FileWatcher


class _BaseEventHandler:
    pass


class _RecordingObserver:
    instances: list[_RecordingObserver] = []

    def __init__(self) -> None:
        self.handler = None
        self.path = ""
        self.recursive = False
        self.started = False
        self.stopped = False
        self.join_timeout = None
        self.__class__.instances.append(self)

    def schedule(self, handler, path: str, *, recursive: bool) -> None:
        self.handler = handler
        self.path = path
        self.recursive = recursive

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float) -> None:
        self.join_timeout = timeout


class _NoopThread:
    def __init__(self, *, target, daemon: bool) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False
        self.join_timeout = None

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float) -> None:
        self.join_timeout = timeout


def _install_fake_watchdog(
    monkeypatch: pytest.MonkeyPatch,
    observer_cls: type[_RecordingObserver] = _RecordingObserver,
) -> None:
    watchdog_pkg = types.ModuleType("watchdog")
    events_mod = types.ModuleType("watchdog.events")
    events_mod.FileSystemEventHandler = _BaseEventHandler
    observers_mod = types.ModuleType("watchdog.observers")
    observers_mod.Observer = observer_cls
    monkeypatch.setitem(sys.modules, "watchdog", watchdog_pkg)
    monkeypatch.setitem(sys.modules, "watchdog.events", events_mod)
    monkeypatch.setitem(sys.modules, "watchdog.observers", observers_mod)


def _install_fake_pathspec(monkeypatch: pytest.MonkeyPatch) -> None:
    pathspec_mod = types.ModuleType("pathspec")

    class _FakeSpec:
        def __init__(self, patterns: list[str]) -> None:
            self._patterns = {pattern.rstrip("/") for pattern in patterns if pattern}

        def match_file(self, path: str) -> bool:
            return path in self._patterns or any(
                path.startswith(f"{pattern}/") for pattern in self._patterns
            )

    class _PathSpec:
        @staticmethod
        def from_lines(_style: str, patterns: list[str]) -> _FakeSpec:
            return _FakeSpec(patterns)

    pathspec_mod.PathSpec = _PathSpec
    monkeypatch.setitem(sys.modules, "pathspec", pathspec_mod)


def test_constructor_validates_root_and_debounce(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="root path does not exist"):
        FileWatcher(tmp_path / "missing")

    with pytest.raises(ValueError, match="debounce_ms must be >= 0"):
        FileWatcher(tmp_path, debounce_ms=-1)

    watcher = FileWatcher(tmp_path, debounce_ms=25, respect_gitignore=False)

    assert watcher.root == str(tmp_path.resolve())
    assert watcher.is_running is False
    assert watcher._debounce == 0.025
    assert watcher._gitignore_matcher is None


def test_on_change_registers_callback_and_dispatch_logs_failures(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    watcher = FileWatcher(tmp_path)
    calls: list[list[ChangeEvent]] = []

    @watcher.on_change
    def _callback(batch: list[ChangeEvent]) -> None:
        calls.append(batch)

    batch = [ChangeEvent(path=str(tmp_path / "tool.py"), kind="modify")]
    watcher._dispatch(batch)

    assert calls == [batch]

    def _raising_callback(_batch: list[ChangeEvent]) -> None:
        raise RuntimeError("boom")

    watcher.on_change(_raising_callback)
    watcher._dispatch(batch)

    assert "file_watcher callback raised" in caplog.text


def test_enqueue_filters_default_skip_dirs_and_gitignore_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\nignored_dir/\n", encoding="utf-8")
    _install_fake_pathspec(monkeypatch)
    watcher = FileWatcher(tmp_path)
    observed: list[list[ChangeEvent]] = []
    watcher.on_change(observed.append)

    watcher._enqueue(str(tmp_path / ".git" / "config"), "modify")
    watcher._enqueue(str(tmp_path / "__pycache__" / "tool.pyc"), "modify")
    watcher._enqueue(str(tmp_path / "ignored.py"), "modify")
    watcher._enqueue(str(tmp_path / "ignored_dir" / "nested.py"), "modify")
    watcher._enqueue(str(tmp_path / "src" / "tool.py"), "modify")
    watcher._enqueue(str(tmp_path / "outside.py"), "modify")
    watcher._flush_now()

    assert observed == [
        [
            ChangeEvent(path=str(tmp_path / "src" / "tool.py"), kind="modify"),
            ChangeEvent(path=str(tmp_path / "outside.py"), kind="modify"),
        ]
    ]


def test_enqueue_deduplicates_by_path_and_kind_then_maybe_flushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = FileWatcher(tmp_path, debounce_ms=100, respect_gitignore=False)
    observed: list[list[ChangeEvent]] = []
    watcher.on_change(observed.append)
    now = 10.0
    monkeypatch.setattr(_fallback.time, "monotonic", lambda: now)

    source = tmp_path / "src" / "tool.py"
    watcher._enqueue(str(source), "modify")
    watcher._enqueue(str(source), "modify")
    watcher._enqueue(str(source), "delete")
    watcher._maybe_flush()

    assert observed == []

    now = 10.2
    watcher._maybe_flush()

    assert observed == [
        [
            ChangeEvent(path=str(source), kind="modify"),
            ChangeEvent(path=str(source), kind="delete"),
        ]
    ]
    assert watcher._pending == {}
    assert watcher._last_event_at is None


def test_start_stop_context_manager_and_handler_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingObserver.instances.clear()
    _install_fake_watchdog(monkeypatch)
    monkeypatch.setattr(_fallback.threading, "Thread", _NoopThread)
    watcher = FileWatcher(tmp_path, respect_gitignore=False)
    observed: list[list[ChangeEvent]] = []
    watcher.on_change(observed.append)

    with watcher as running:
        assert running is watcher
        assert watcher.is_running is True
        observer = _RecordingObserver.instances[-1]
        assert observer.started is True
        assert observer.path == str(tmp_path.resolve())
        assert observer.recursive is True

        with pytest.raises(RuntimeError, match="watcher already started"):
            watcher.start()

        handler = observer.handler
        handler.on_created(SimpleNamespace(src_path=str(tmp_path / "created.py")))
        handler.on_modified(
            SimpleNamespace(src_path=str(tmp_path / "directory"), is_directory=True)
        )
        handler.on_modified(
            SimpleNamespace(src_path=str(tmp_path / "modified.py"), is_directory=False)
        )
        handler.on_deleted(SimpleNamespace(src_path=str(tmp_path / "deleted.py")))
        handler.on_moved(
            SimpleNamespace(
                src_path=str(tmp_path / "old.py"),
                dest_path=str(tmp_path / "new.py"),
            )
        )

    assert watcher.is_running is False
    assert observer.stopped is True
    assert observer.join_timeout == 2.0
    assert observed == [
        [
            ChangeEvent(path=str(tmp_path / "created.py"), kind="create"),
            ChangeEvent(path=str(tmp_path / "modified.py"), kind="modify"),
            ChangeEvent(path=str(tmp_path / "deleted.py"), kind="delete"),
            ChangeEvent(path=str(tmp_path / "old.py"), kind="rename"),
            ChangeEvent(path=str(tmp_path / "new.py"), kind="rename"),
        ]
    ]


def test_handler_moved_event_without_destination_enqueues_source_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingObserver.instances.clear()
    _install_fake_watchdog(monkeypatch)
    monkeypatch.setattr(_fallback.threading, "Thread", _NoopThread)
    watcher = FileWatcher(tmp_path, respect_gitignore=False)
    observed: list[list[ChangeEvent]] = []
    watcher.on_change(observed.append)
    watcher.start()

    observer = _RecordingObserver.instances[-1]
    observer.handler.on_moved(SimpleNamespace(src_path=str(tmp_path / "old.py")))
    watcher.stop()

    assert observed == [[ChangeEvent(path=str(tmp_path / "old.py"), kind="rename")]]


def test_noop_branches_do_not_dispatch_or_require_running_watcher(
    tmp_path: Path,
) -> None:
    watcher = FileWatcher(tmp_path, respect_gitignore=False)

    watcher.__exit__(None, None, None)
    watcher._maybe_flush()
    watcher._flush_now()
    watcher._dispatch([ChangeEvent(path=str(tmp_path / "ignored.py"), kind="modify")])

    assert watcher._should_ignore(str(tmp_path.parent / "outside.py")) is False
    assert watcher._pending == {}
    assert watcher.is_running is False


def test_flush_loop_polls_until_stop_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = FileWatcher(tmp_path, debounce_ms=0, respect_gitignore=False)
    sleeps: list[float] = []
    maybe_flush_calls = 0

    def _sleep(delay: float) -> None:
        sleeps.append(delay)

    def _maybe_flush() -> None:
        nonlocal maybe_flush_calls
        maybe_flush_calls += 1
        watcher._stop_flag.set()

    monkeypatch.setattr(_fallback.time, "sleep", _sleep)
    monkeypatch.setattr(watcher, "_maybe_flush", _maybe_flush)

    watcher._flush_loop()

    assert sleeps == [0.005]
    assert maybe_flush_calls == 1


def test_stop_requires_started_watcher(tmp_path: Path) -> None:
    watcher = FileWatcher(tmp_path)

    with pytest.raises(RuntimeError, match="watcher not started"):
        watcher.stop()


def test_start_reports_missing_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _blocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "watchdog.events":
            raise ImportError("not installed")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(RuntimeError, match="watchdog is required") as excinfo:
        FileWatcher(tmp_path).start()

    assert isinstance(excinfo.value.__cause__, ImportError)


def test_gitignore_build_handles_missing_pathspec_and_unreadable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("ignored.py\n", encoding="utf-8")
    original_import = builtins.__import__

    def _blocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "pathspec":
            raise ImportError("missing pathspec")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert FileWatcher(tmp_path)._gitignore_matcher is None
    assert "pathspec not installed" in caplog.text

    _install_fake_pathspec(monkeypatch)
    monkeypatch.setattr(builtins, "__import__", original_import)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError),
    )

    assert FileWatcher(tmp_path)._gitignore_matcher is None
