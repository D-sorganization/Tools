#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "::error::install-system-deps.sh requires at least one package name"
  exit 2
fi

APT_PACKAGES=("$@")

if [ "$(id -u)" -eq 0 ]; then
  APT=(env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300)
  LOCK_PREFIX=()
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  APT=(sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300)
  LOCK_PREFIX=(sudo)
else
  echo "::notice::Skipping apt install because this self-hosted runner does not provide noninteractive sudo."
  echo "::notice::Skipped packages: ${APT_PACKAGES[*]}"
  exit 0
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>/tmp/d-sorg-apt-install.lock
  flock 9
fi

while "${LOCK_PREFIX[@]}" fuser /var/lib/dpkg/lock >/dev/null 2>&1 ||
  "${LOCK_PREFIX[@]}" fuser /var/lib/apt/lists/lock >/dev/null 2>&1 ||
  "${LOCK_PREFIX[@]}" fuser /var/cache/apt/archives/lock >/dev/null 2>&1; do
  sleep 1
done

"${LOCK_PREFIX[@]}" rm -f \
  /var/lib/apt/lists/lock \
  /var/cache/apt/archives/lock \
  /var/lib/dpkg/lock \
  /var/cache/apt/pkgcache.bin \
  /var/cache/apt/srcpkgcache.bin

"${APT[@]}" update --fix-missing
"${APT[@]}" install -y --fix-missing "${APT_PACKAGES[@]}"
