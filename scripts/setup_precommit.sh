#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install pre-commit black isort ruff mypy nbstripout
pre-commit install
echo "Pre-commit hooks installed."

# Register local-only git merge drivers (e.g. module-inventory-regen).
# .gitattributes can only *name* a driver; the command it runs is local
# git config that .gitattributes cannot embed (a deliberate security
# boundary -- see scripts/git/install_merge_drivers.py), so it has to be
# set up per clone/worktree here.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${REPO_ROOT}/scripts/git/install_merge_drivers.py"
echo "Git merge drivers registered."
