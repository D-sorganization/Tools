"""Bounded apt lock wait and package install helpers for CI workflows (#5717)."""

from __future__ import annotations

import shlex
from collections.abc import Sequence

LOCK_WAIT_SECONDS = 120
APT_NETWORK_TIMEOUT_SECONDS = 600
FLOCK_PATH = "/tmp/d-sorg-apt-install.lock"


def render_wait_for_apt_locks() -> str:
    """Return bash that waits for dpkg/apt locks with a finite deadline."""
    return f"""wait_for_apt_locks() {{
  local deadline=$((SECONDS + {LOCK_WAIT_SECONDS}))
  while fuser /var/lib/dpkg/lock >/dev/null 2>&1 \\
    || fuser /var/lib/apt/lists/lock >/dev/null 2>&1 \\
    || fuser /var/cache/apt/archives/lock >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "::error::Timed out after {LOCK_WAIT_SECONDS}s waiting for apt/dpkg locks"
      return 1
    fi
    sleep 1
  done
}}"""


def render_apt_install_body(packages: Sequence[str]) -> str:
    """Return bash function body that installs *packages* with bounded apt I/O."""
    pkg_array = " ".join(shlex.quote(p) for p in packages)
    return f"""install_system_deps() {{
  set -euo pipefail
  wait_for_apt_locks
  rm -f /etc/apt/sources.list.d/*chrome* /etc/apt/sources.list.d/*google*
  sed -i '/dl\\.google\\.com/d' /etc/apt/sources.list 2>/dev/null || true
  timeout {APT_NETWORK_TIMEOUT_SECONDS} env DEBIAN_FRONTEND=noninteractive \\
    apt-get -o DPkg::Lock::Timeout=60 -o Acquire::Retries=3 update --fix-missing
  timeout {APT_NETWORK_TIMEOUT_SECONDS} env DEBIAN_FRONTEND=noninteractive \\
    apt-get -o DPkg::Lock::Timeout=60 -o Acquire::Retries=3 install -y --fix-missing {pkg_array}
}}"""


def render_install_system_dependencies_step(packages: Sequence[str]) -> str:
    """Full ``run:`` block shared by quality-gate and test shard jobs."""
    wait_fn = render_wait_for_apt_locks()
    install_fn = render_apt_install_body(packages)
    return f"""set -o pipefail
{wait_fn}
{install_fn}
if [ "$(id -u)" -eq 0 ]; then
  flock -w {LOCK_WAIT_SECONDS} {FLOCK_PATH} bash -c "$(declare -f wait_for_apt_locks); $(declare -f install_system_deps); install_system_deps"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo -n flock -w {LOCK_WAIT_SECONDS} {FLOCK_PATH} bash -c "$(declare -f wait_for_apt_locks); $(declare -f install_system_deps); install_system_deps"
else
  echo "::warning::Passwordless sudo is unavailable; skipping apt install and using pre-provisioned system dependencies."
fi
"""
