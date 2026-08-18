#!/usr/bin/env bash
#
# Install + enable systemd services for the P1AM control system so the backend
# (FastAPI/uvicorn, the single Modbus master) and the HMI (Vite, :3002) start on
# boot and restart on failure — no more manual relaunches after a reboot, power
# loss, or terminal/session teardown.
#
# Idempotent: re-run after pulling changes to refresh the unit files. Paths are
# detected from this script's location, so it is not pinned to one machine/user.
# Generated credentials are preserved across re-runs.
#
# SECURITY (issue #4007): this script used to write
# `Environment=P1AM_DEV_NO_AUTH=1` into the backend unit, which short-circuits
# require_api_key / require_admin_key / verify_operator_key — every production
# install shipped with authentication disabled. It now generates real
# credentials into a root-owned EnvironmentFile and REFUSES to write the unit
# without one. The old behaviour is still available, but only behind an
# explicit --bench flag that says out loud what it does.
#
# Usage:
#   ./deploy/install-services.sh                     # secure install
#   PLC_IP=192.168.1.100 ./deploy/install-services.sh
#   ./deploy/install-services.sh --bench             # NO AUTH, bench only
#
set -euo pipefail

BENCH_MODE=0
for arg in "$@"; do
  case "$arg" in
    --bench)
      BENCH_MODE=1
      ;;
    -h|--help)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$arg' (expected --bench)" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYS="$(cd "$SCRIPT_DIR/.." && pwd)"      # .../src/p1am_control_system
REPO="$(cd "$SYS/../.." && pwd)"         # .../Tools
VENV="$REPO/.venv"
PY="$VENV/bin/python"
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
NPM="$(command -v npm)"
NODE_BIN_DIR="$(dirname "$(command -v node)")"
PLC_IP="${PLC_IP:-192.168.1.100}"
PLC_PORT="${PLC_PORT:-502}"
ENV_DIR="/etc/p1am"
ENV_FILE="$ENV_DIR/backend.env"
HMI_ORIGINS="${P1AM_CORS_ORIGINS:-http://localhost:3002,http://127.0.0.1:3002}"

[ -x "$PY" ] || { echo "ERROR: venv python not found at $PY" >&2; exit 1; }
[ -n "$NPM" ] || { echo "ERROR: npm not found on PATH" >&2; exit 1; }
[ -d "$SYS/backend" ] || { echo "ERROR: backend dir missing: $SYS/backend" >&2; exit 1; }
[ -d "$SYS/frontend" ] || { echo "ERROR: frontend dir missing: $SYS/frontend" >&2; exit 1; }

echo "repo:    $REPO"
echo "venv py: $PY"
echo "npm:     $NPM   (node dir: $NODE_BIN_DIR)"
echo "user:    $USER_NAME:$GROUP_NAME"
echo "PLC:     $PLC_IP:$PLC_PORT"
echo

# --------------------------------------------------------------------------- #
# 1) Backend runtime dependencies                                              #
# --------------------------------------------------------------------------- #
# Checking that the interpreter exists proved nothing: no manifest installed
# fastapi/uvicorn/pymodbus/sqlmodel/pydantic-settings, so the unit crash-looped
# on ModuleNotFoundError (issue #4014). The `p1am` extra in pyproject.toml is
# the single source of truth for what the backend needs.
echo "==> Installing backend runtime dependencies (.[p1am])"
"$PY" -m pip install --disable-pip-version-check -e "$REPO[p1am]"

echo "==> Verifying the backend can import"
if ! (cd "$SYS/backend" && PYTHONPATH="$REPO/src" "$PY" -c "import main" >/dev/null); then
  echo "ERROR: the backend failed to import; refusing to install a crash loop." >&2
  exit 1
fi

# tools_core is the Rust-accelerated SCADA kernel main.py prefers. It is shipped
# by no artifact, so a deployment silently runs the pure-Python fallback behind
# one boot WARNING nobody reads on a headless Pi (issue #4036). Say it here,
# where a human is watching, and build it when a toolchain is available.
if "$PY" -c "import tools_core" >/dev/null 2>&1; then
  echo "==> SCADA kernel: tools_core (Rust-accelerated)"
elif command -v cargo >/dev/null 2>&1 && "$PY" -m maturin --version >/dev/null 2>&1; then
  echo "==> Building the tools_core aarch64 wheel (maturin)"
  (cd "$REPO" && "$PY" -m maturin build --release --features python \
      -m rust_core/tools-core/Cargo.toml) &&
    "$PY" -m pip install --disable-pip-version-check --force-reinstall \
      "$REPO"/target/wheels/tools_core-*.whl ||
    echo "WARNING: tools_core build failed; continuing on the Python fallback."
else
  echo "WARNING: tools_core is not installed and no Rust toolchain is present."
  echo "         The backend will run the pure-Python scada_fallback kernel."
fi

# --------------------------------------------------------------------------- #
# 2) Credentials                                                               #
# --------------------------------------------------------------------------- #
if [ "$BENCH_MODE" = "1" ]; then
  cat >&2 <<'BENCH'
################################################################################
#  --bench: INSTALLING WITH AUTHENTICATION DISABLED (P1AM_DEV_NO_AUTH=1).      #
#                                                                              #
#  Every control endpoint — E-stop clear, tag force, setpoints, project import  #
#  (which wipes the plant DB) — will accept ANY caller. Use this only on an     #
#  isolated bench with no hardware energized. Re-run without --bench to         #
#  install the secure configuration.                                           #
################################################################################
BENCH
  AUTH_ENV_LINES="Environment=P1AM_DEV_NO_AUTH=1"
else
  echo "==> Provisioning credentials in $ENV_FILE"
  sudo install -d -m 0750 -o root -g "$GROUP_NAME" "$ENV_DIR"

  # Preserve existing keys so a re-run does not invalidate the operator's
  # browser or any script already holding a credential.
  existing_api_key="$(sudo sh -c "grep -m1 '^P1AM_API_KEY=' '$ENV_FILE' 2>/dev/null || true")"
  existing_admin_key="$(sudo sh -c "grep -m1 '^P1AM_ADMIN_API_KEY=' '$ENV_FILE' 2>/dev/null || true")"

  if [ -z "$existing_api_key" ]; then
    command -v openssl >/dev/null 2>&1 || {
      echo "ERROR: openssl is required to generate credentials." >&2; exit 1; }
    # The value is generated at install time; no credential is stored in source.
    existing_api_key="P1AM_API_KEY=$(openssl rand -hex 32)" # pragma: allowlist secret
    echo "    generated a new operator credential"
  fi
  if [ -z "$existing_admin_key" ]; then
    existing_admin_key="P1AM_ADMIN_API_KEY=$(openssl rand -hex 32)"
    echo "    generated a new admin credential"
  fi

  # Root-owned, group-readable: the systemd unit and the kiosk launcher (which
  # seeds the HMI's stored key) can read it; nothing else on the box can.
  sudo sh -c "umask 027 && cat > '$ENV_FILE'" <<ENVFILE
# Generated by deploy/install-services.sh — do not commit, do not world-read.
$existing_api_key
$existing_admin_key
ENVFILE
  sudo chown "root:$GROUP_NAME" "$ENV_FILE"
  sudo chmod 640 "$ENV_FILE"

  # Fail closed: never write a unit that would come up unauthenticated.
  if ! sudo grep -qE '^P1AM_(ADMIN_)?API_KEY=.+' "$ENV_FILE"; then
    echo "ERROR: no credential present in $ENV_FILE — refusing to install an" >&2
    echo "       unauthenticated control backend. Use --bench if that is truly" >&2
    echo "       what you want." >&2
    exit 1
  fi
  AUTH_ENV_LINES="EnvironmentFile=$ENV_FILE"
fi

# --------------------------------------------------------------------------- #
# 3) Build the HMI ONCE, here — not inside ExecStart                           #
# --------------------------------------------------------------------------- #
# A full tsc + vite build inside ExecStart on a Restart=always unit saturates
# all four Pi cores for 1-3 minutes on every start, and loops forever if preview
# then fails to bind (issue #4036). Build at install time; ExecStart just serves.
echo "==> Building the HMI production bundle"
(cd "$SYS/frontend" && "$NPM" ci && "$NPM" run build)

# --------------------------------------------------------------------------- #
# 4) Unit files                                                                #
# --------------------------------------------------------------------------- #
backend_unit="[Unit]
Description=P1AM control backend (FastAPI/uvicorn, Modbus master)
Documentation=https://github.com/D-sorganization/Tools
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$SYS/backend
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$REPO/src
$AUTH_ENV_LINES
Environment=P1AM_REQUIRE_READ_AUTH=1
Environment=P1AM_ENV=production
Environment=P1AM_CORS_ORIGINS=$HMI_ORIGINS
Environment=P1AM_PLC_DRIVER=modbus
Environment=P1AM_PLC_IP=$PLC_IP
Environment=P1AM_PLC_PORT=$PLC_PORT
Environment=P1AM_BIND_HOST=127.0.0.1
Environment=P1AM_BIND_PORT=8000
ExecStart=$PY -m uvicorn main:app --host 127.0.0.1 --port 8000
# The Modbus master is the real-time path: give it the CPU ahead of the HMI.
Nice=-5
CPUWeight=800
Restart=always
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
"

frontend_unit="[Unit]
Description=P1AM HMI (production build served via vite preview, 127.0.0.1:3002)
Documentation=https://github.com/D-sorganization/Tools
After=network-online.target p1am-backend.service
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$SYS/frontend
Environment=PATH=$NODE_BIN_DIR:/usr/local/bin:/usr/bin:/bin
Environment=NODE_ENV=production
# Serve the pre-built dist/ only. The bundle is built by install-services.sh,
# so a restart is instant instead of a multi-minute four-core build storm.
# vite.config.ts pins preview.host to 127.0.0.1 — the kiosk browser runs on this
# Pi, and binding wider would re-expose the loopback-bound backend via the
# preview server's /api + WebSocket proxy (#4007).
ExecStart=$NPM run preview
# The HMI yields to the control loop under contention.
Nice=10
CPUWeight=100
Restart=always
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
"

echo "$backend_unit" | sudo tee /etc/systemd/system/p1am-backend.service >/dev/null
echo "$frontend_unit" | sudo tee /etc/systemd/system/p1am-frontend.service >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable p1am-backend.service p1am-frontend.service
sudo systemctl restart p1am-backend.service p1am-frontend.service

echo
echo "Installed, enabled (start on boot), and (re)started."
if [ "$BENCH_MODE" != "1" ]; then
  echo
  echo "The HMI needs the operator credential once per browser profile."
  echo "deploy/launch-hmi.sh seeds it automatically from $ENV_FILE."
  echo "To enter it by hand (or from another machine over an SSH tunnel):"
  echo "  sudo grep '^P1AM_API_KEY=' $ENV_FILE"
  echo "then paste the value when the HMI prompts for it."
fi
echo "  status:  systemctl status p1am-backend p1am-frontend"
echo "  logs:    journalctl -u p1am-backend -f"
echo "  audit:   journalctl -u p1am-backend -f | grep AUDIT"
echo "  restart: sudo systemctl restart p1am-backend"
