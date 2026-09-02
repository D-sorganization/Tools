#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install pre-commit black isort ruff mypy nbstripout
pre-commit install
echo "Pre-commit hooks installed."

# pre-merge-commit is a distinct hook type from pre-commit -- plain
# pre-commit does NOT fire for merge commits (confirmed empirically while
# building #4818) -- needed by the module-inventory merge fixup hook (see
# scripts/git/regenerate_module_inventory_during_merge.py).
pre-commit install --hook-type pre-merge-commit
echo "Pre-merge-commit hooks installed."

# Register local-only git merge drivers (e.g. module-inventory-regen).
# .gitattributes can only *name* a driver; the command it runs is local
# git config that .gitattributes cannot embed (a deliberate security
# boundary -- see scripts/git/install_merge_drivers.py), so it has to be
# set up per clone/worktree here.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${REPO_ROOT}/scripts/git/install_merge_drivers.py"
echo "Git merge drivers registered."
