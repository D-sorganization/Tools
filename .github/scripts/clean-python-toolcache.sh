#!/usr/bin/env bash
set -euo pipefail

requested_minor="${1:-}"
if [[ ! "$requested_minor" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "::error::Expected a Python minor version such as 3.12"
  exit 2
fi

cache_roots=(
  "${AGENT_TOOLSDIRECTORY:-}"
  "/opt/runner-tool-cache"
  "/home/dieterolson/actions-runners/.shared-tool-cache"
  "/home/dieterolson/actions-runners-nvme/.shared-tool-cache"
)

clean_cache_entry() {
  local arch_dir="$1"
  local complete_marker="${arch_dir}.complete"

  echo "Cleaning semantically invalid Python cache entry $arch_dir..."
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo -n rm -rf -- "$arch_dir" "$complete_marker"
  else
    rm -rf -- "$arch_dir" "$complete_marker"
  fi
}

shopt -s nullglob
for cache_root in "${cache_roots[@]}"; do
  [ -n "$cache_root" ] || continue
  python_cache="$cache_root/Python"
  [ -d "$python_cache" ] || continue

  echo "Checking $python_cache for stale $requested_minor installations..."
  for version_dir in "$python_cache"/"$requested_minor".*; do
    [ -d "$version_dir" ] || continue
    expected_version="$(basename "$version_dir")"

    for arch_dir in "$version_dir"/*; do
      [ -d "$arch_dir" ] || continue
      py_bin="$arch_dir/bin/python"
      if [ ! -x "$py_bin" ]; then
        py_bin="$arch_dir/bin/python3"
      fi

      if [ ! -x "$py_bin" ] || [ ! -s "$py_bin" ]; then
        clean_cache_entry "$arch_dir"
        continue
      fi

      echo "Verifying Python $expected_version and zlib on $py_bin..."
      python_version="$(
        LD_LIBRARY_PATH="$arch_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$py_bin" -c \
          'import sys, zlib; print(".".join(str(part) for part in sys.version_info[:3]))' \
          2>/dev/null
      )" || python_version=""
      pip_version="$(
        LD_LIBRARY_PATH="$arch_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$py_bin" -m pip --version 2>/dev/null
      )" || pip_version=""

      if [ "$python_version" != "$expected_version" ] ||
        [[ "$pip_version" != pip\ *" from "* ]]; then
        clean_cache_entry "$arch_dir"
      else
        echo "Directory $arch_dir is healthy."
      fi
    done
  done
done
