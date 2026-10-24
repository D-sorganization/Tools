#!/usr/bin/env bash
# Bounded system dependency install for ci-standard (#5717).
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "::error::usage: install-ci-system-deps.sh <apt-package> [...]" >&2
  exit 2
fi

LOCK_WAIT_SECONDS="${CI_APT_LOCK_WAIT_SECONDS:-120}"
APT_NETWORK_TIMEOUT_SECONDS="${CI_APT_NETWORK_TIMEOUT_SECONDS:-600}"
FLOCK_PATH="/tmp/d-sorg-apt-install.lock"
APT_PACKAGES=("$@")

wait_for_apt_locks() {
  local deadline=$((SECONDS + LOCK_WAIT_SECONDS))
  while fuser /var/lib/dpkg/lock >/dev/null 2>&1 \
    || fuser /var/lib/apt/lists/lock >/dev/null 2>&1 \
    || fuser /var/cache/apt/archives/lock >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "::error::Timed out after ${LOCK_WAIT_SECONDS}s waiting for apt/dpkg locks"
      return 1
    fi
    sleep 1
  done
}

install_system_deps() {
  wait_for_apt_locks
  rm -f /etc/apt/sources.list.d/*chrome* /etc/apt/sources.list.d/*google*
  sed -i '/dl\.google\.com/d' /etc/apt/sources.list 2>/dev/null || true
  timeout "$APT_NETWORK_TIMEOUT_SECONDS" env DEBIAN_FRONTEND=noninteractive \
    apt-get -o DPkg::Lock::Timeout=60 -o Acquire::Retries=3 update --fix-missing
  timeout "$APT_NETWORK_TIMEOUT_SECONDS" env DEBIAN_FRONTEND=noninteractive \
    apt-get -o DPkg::Lock::Timeout=60 -o Acquire::Retries=3 install -y --fix-missing "${APT_PACKAGES[@]}"
}

if [ "$(id -u)" -eq 0 ]; then
  flock -w "$LOCK_WAIT_SECONDS" "$FLOCK_PATH" bash -c \
    "$(declare -f wait_for_apt_locks); $(declare -f install_system_deps); $(declare -p APT_PACKAGES); install_system_deps"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo -n flock -w "$LOCK_WAIT_SECONDS" "$FLOCK_PATH" bash -c \
    "$(declare -f wait_for_apt_locks); $(declare -f install_system_deps); $(declare -p APT_PACKAGES); install_system_deps"
else
  echo "::warning::Passwordless sudo is unavailable; skipping apt install and using pre-provisioned system dependencies."
fi
