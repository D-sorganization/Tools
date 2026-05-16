"""Test bootstrap for tests/unit/chat (Tools issue #2871).

The shared :mod:`chat` package is shipped via an editable install in the
developer's main worktree. When tests run from a parallel worktree (e.g.
issue branches built in side-by-side repos), the editable install path
still wins, so this conftest forces the in-tree source to take priority
on ``sys.path`` and evicts any pre-cached chat modules so the live tree
is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_TREE_SRC = _REPO_ROOT / "src" / "shared" / "python"
_WORKTREE_CHAT_DIR = str(_TREE_SRC / "chat")

# Prepend the in-tree shared/python so it wins over any editable install.
_path_str = str(_TREE_SRC)
if _path_str in sys.path:
    sys.path.remove(_path_str)
sys.path.insert(0, _path_str)

# Prepend the worktree chat directory to the chat package's __path__ so that
# reimports of chat.* submodules resolve from the in-tree source first, before
# the editable-install path. We do NOT evict ``chat`` itself to avoid losing
# pytest's conftest-tracking entries.
_chat_mod = sys.modules.get("chat")
if _chat_mod is not None:
    _existing_path = list(getattr(_chat_mod, "__path__", []))
    if _WORKTREE_CHAT_DIR not in _existing_path:
        try:
            _chat_mod.__path__ = [_WORKTREE_CHAT_DIR] + _existing_path  # type: ignore[assignment]
        except (AttributeError, TypeError):
            pass
    elif _existing_path[0] != _WORKTREE_CHAT_DIR:
        # Already present but not first — move it to front.
        _existing_path.remove(_WORKTREE_CHAT_DIR)
        try:
            _chat_mod.__path__ = [_WORKTREE_CHAT_DIR] + _existing_path  # type: ignore[assignment]
        except (AttributeError, TypeError):
            pass

# Evict cached chat.* submodules (not chat itself, not test-dir entries)
# so they are reimported from the updated __path__ on next access.
_test_dir = str(_HERE.parent)
for _name in list(sys.modules):
    if not _name.startswith("chat."):
        continue
    _mod = sys.modules.get(_name)
    _file = getattr(_mod, "__file__", "") or ""
    if _test_dir in _file:
        continue
    del sys.modules[_name]
