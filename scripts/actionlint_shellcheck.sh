#!/usr/bin/env bash
set -euo pipefail

exec shellcheck -S warning "$@"
