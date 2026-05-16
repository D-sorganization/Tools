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

# Prepend the in-tree shared/python so it wins over any editable install.
_path_str = str(_TREE_SRC)
if _path_str in sys.path:
    sys.path.remove(_path_str)
sys.path.insert(0, _path_str)

# Evict the chat package from sys.modules so the next ``import chat``
# resolves through the prepended path. Only purge entries whose origin
# is *not* this test directory; otherwise we lose pytest's discovery of
# its own conftest under ``chat.conftest``.
_test_dir = str(_HERE.parent)
for _name in list(sys.modules):
    if not (_name == "chat" or _name.startswith("chat.")):
        continue
    _mod = sys.modules.get(_name)
    _file = getattr(_mod, "__file__", "") or ""
    if _test_dir in _file:
        continue
    del sys.modules[_name]
