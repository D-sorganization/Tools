#!/usr/bin/env bash
set -euo pipefail

tool_dir="${AGENT_TOOLSDIRECTORY:-}"
if [ -z "$tool_dir" ]; then
  echo "AGENT_TOOLSDIRECTORY is not set; skipping local Python toolcache restore."
  exit 0
fi

python_cache="$tool_dir/Python"
mkdir -p "$python_cache"

restore_python() {
  local version="$1"
  local venv_dir="$2"
  local interpreter="$venv_dir/bin/python"
  local cache_dir="$python_cache/$version/x64"

  if [ ! -x "$interpreter" ]; then
    echo "Local Python $version not found at $interpreter; setup-python may download it if supported."
    return
  fi

  mkdir -p "$cache_dir/bin"
  ln -sfn "$interpreter" "$cache_dir/bin/python"
  ln -sfn "$interpreter" "$cache_dir/bin/python3"
  ln -sfn "$interpreter" "$cache_dir/bin/python${version%.*}"
  if [ -f "$venv_dir/pyvenv.cfg" ]; then
    ln -sfn "$venv_dir/pyvenv.cfg" "$cache_dir/pyvenv.cfg"
  fi
  if [ -d "$venv_dir/lib" ]; then
    ln -sfn "$venv_dir/lib" "$cache_dir/lib"
  fi
  if [ -d "$venv_dir/include" ]; then
    ln -sfn "$venv_dir/include" "$cache_dir/include"
  fi
  pip_bin="$venv_dir/bin/pip"
  if [ -x "$pip_bin" ]; then
    ln -sfn "$pip_bin" "$cache_dir/bin/pip"
    ln -sfn "$pip_bin" "$cache_dir/bin/pip3"
    ln -sfn "$pip_bin" "$cache_dir/bin/pip${version%.*}"
  fi
  touch "$python_cache/$version/x64.complete"
  echo "Restored Python $version in $cache_dir"
}

shopt -s nullglob
for venv_dir in /home/dieterolson/actions-runners/python-venvs/[0-9]*.[0-9]*.[0-9]*; do
  version="$(basename "$venv_dir")"
  case "$version" in
    3.10.*|3.11.*|3.12.*|3.13.*)
      restore_python "$version" "$venv_dir"
      ;;
  esac
done
