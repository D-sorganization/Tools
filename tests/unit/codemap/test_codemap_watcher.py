from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from codemap import watcher


class _BaseEventHandler:
    pass


def _install_fake_watchdog(
    monkeypatch: pytest.MonkeyPatch,
    observer_cls: type | None = None,
) -> None:
    watchdog_pkg = types.ModuleType("watchdog")
    events_mod = types.ModuleType("watchdog.events")
    events_mod.FileSystemEventHandler = _BaseEventHandler
    observers_mod = types.ModuleType("watchdog.observers")
    observers_mod.Observer = observer_cls or _RecordingObserver
    monkeypatch.setitem(sys.modules, "watchdog", watchdog_pkg)
    monkeypatch.setitem(sys.modules, "watchdog.events", events_mod)
    monkeypatch.setitem(sys.modules, "watchdog.observers", observers_mod)


class _RecordingObserver:
    instances: list[_RecordingObserver] = []

    def __init__(self) -> None:
        self.handler = None
        self.path = ""
        self.recursive = False
        self.started = False
        self.stopped = False
        self.join_timeout = None
        self.events: list[SimpleNamespace] = []
        self.__class__.instances.append(self)

    def schedule(self, handler, path: str, *, recursive: bool) -> None:
        self.handler = handler
        self.path = path
        self.recursive = recursive

    def start(self) -> None:
        self.started = True
        for event in self.events:
            event.emit(self.handler)

    def stop(self) -> None:
        self.stopped = True

    def join(self, *, timeout: int) -> None:
        self.join_timeout = timeout


class _ImmediateTimer:
    def __init__(self, _delay: float, target) -> None:
        self.target = target
        self.daemon = False
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def start(self) -> None:
        self.target()


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[str, ...]]] = []
        self.commits = 0
        self.closed = False

    def execute(self, sql: str, params: tuple[str, ...]) -> None:
        self.executed.append((sql, params))

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def test_handler_enqueues_supported_paths_and_filters_unsupported_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_watchdog(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "tool.py"
    source.parent.mkdir()
    source.write_text("def run(): pass\n", encoding="utf-8")
    notes = repo / "src" / "notes.txt"
    notes.write_text("not source\n", encoding="utf-8")
    codemap_file = repo / ".codemap" / "generated.py"
    codemap_file.parent.mkdir()
    codemap_file.write_text("def hidden(): pass\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("def outside(): pass\n", encoding="utf-8")
    pending: dict[str, Path] = {}
    schedule_calls = 0

    def _schedule() -> None:
        nonlocal schedule_calls
        schedule_calls += 1

    handler = watcher._make_handler(
        repo,
        _Conn(),
        watcher.threading.Lock(),
        pending,
        _schedule,
    )

    handler.on_created(SimpleNamespace(src_path=str(source)))
    handler.on_modified(SimpleNamespace(src_path=str(notes)))
    handler.on_created(SimpleNamespace(src_path=str(codemap_file)))
    handler.on_created(SimpleNamespace(src_path=str(outside)))
    handler.on_created(SimpleNamespace(src_path=str(source.parent)))

    assert pending == {"src/tool.py": source}
    assert schedule_calls == 1


def test_handler_uses_moved_destination_and_enqueues_deleted_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_watchdog(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    moved = repo / "pkg" / "new_name.py"
    moved.parent.mkdir()
    moved.write_text("def moved(): pass\n", encoding="utf-8")
    deleted = repo / "pkg" / "old_name.py"
    pending: dict[str, Path] = {}
    scheduled: list[None] = []
    handler = watcher._make_handler(
        repo,
        _Conn(),
        watcher.threading.Lock(),
        pending,
        lambda: scheduled.append(None),
    )

    handler.on_moved(
        SimpleNamespace(src_path=str(repo / "pkg" / "old.txt"), dest_path=str(moved))
    )
    handler.on_deleted(SimpleNamespace(src_path=str(deleted)))

    assert pending == {
        "pkg/new_name.py": moved,
        "pkg/old_name.py": deleted,
    }
    assert len(scheduled) == 2


def test_run_reports_missing_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def _blocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "watchdog.observers":
            raise RuntimeError("not installed")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert watcher.run(str(tmp_path)) == 2
    assert "codemap-watch requires watchdog: not installed" in capsys.readouterr().err


def test_run_flushes_created_files_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed: list[tuple[Path, str, Path]] = []
    conn = _Conn()
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "pkg" / "tool.py"
    source.parent.mkdir()
    source.write_text("def run(): pass\n", encoding="utf-8")

    class CreatedObserver(_RecordingObserver):
        def start(self) -> None:
            self.events = [
                SimpleNamespace(
                    emit=lambda handler: handler.on_created(
                        SimpleNamespace(src_path=str(source))
                    )
                )
            ]
            super().start()

    _install_fake_watchdog(monkeypatch, CreatedObserver)
    monkeypatch.setattr(watcher.db_mod, "open_db", lambda _repo: conn)
    monkeypatch.setattr(watcher.threading, "Timer", _ImmediateTimer)
    monkeypatch.setattr(
        watcher.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(
        watcher.indexer_mod,
        "_process_file",
        lambda abs_p, rel, repo_root, _conn, _stats: processed.append(
            (abs_p, rel, repo_root)
        ),
    )

    assert watcher.run(str(repo), debounce=0.01) == 0

    observer = CreatedObserver.instances[-1]
    assert observer.started is True
    assert observer.stopped is True
    assert observer.join_timeout == 5
    assert observer.path == str(repo.resolve())
    assert observer.recursive is True
    assert processed == [(source, "pkg/tool.py", repo.resolve())]
    assert conn.commits == 1
    assert conn.closed is True


def test_run_flushes_deleted_files_from_the_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn()
    repo = tmp_path / "repo"
    repo.mkdir()
    deleted = repo / "pkg" / "removed.py"

    class DeletedObserver(_RecordingObserver):
        def start(self) -> None:
            self.events = [
                SimpleNamespace(
                    emit=lambda handler: handler.on_deleted(
                        SimpleNamespace(src_path=str(deleted))
                    )
                )
            ]
            super().start()

    _install_fake_watchdog(monkeypatch, DeletedObserver)
    monkeypatch.setattr(watcher.db_mod, "open_db", lambda _repo: conn)
    monkeypatch.setattr(watcher.threading, "Timer", _ImmediateTimer)
    monkeypatch.setattr(
        watcher.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert watcher.run(str(repo), debounce=0.01) == 0

    assert conn.executed == [
        ("DELETE FROM symbols WHERE path = ?", ("pkg/removed.py",)),
        ("DELETE FROM files WHERE path = ?", ("pkg/removed.py",)),
    ]
    assert conn.commits == 1
    assert conn.closed is True


def test_main_passes_cli_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, float]] = []
    monkeypatch.setattr(
        watcher,
        "run",
        lambda repo, *, debounce: calls.append((repo, debounce)) or 7,
    )

    assert watcher.main(["--repo", "repo-root", "--debounce", "0.25"]) == 7
    assert calls == [("repo-root", 0.25)]
