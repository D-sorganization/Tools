#!/usr/bin/env bash
#
# Install + enable systemd services for the P1AM control system so the backend
# (FastAPI/uvicorn, the single Modbus master) and the HMI (Vite, :3002) start on
# boot and restart on failure — no more manual relaunches after a reboot, power
# loss, or terminal/session teardown.
#
# Idempotent: re-run after pulling changes to refresh the unit files. Paths are
# detected from this script's location, so it is not pinned to one machine/user.
#
# Usage:
#   ./deploy/install-services.sh            # install, enable, (re)start both
#   PLC_IP=192.168.1.100 ./deploy/install-services.sh
#
set -euo pipefail

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
# Bench/dev mode: disable API auth (mirrors run_pi.sh). Set P1AM_API_KEY /
# P1AM_ADMIN_API_KEY and drop this if the HMI is ever exposed beyond the bench.
Environment=P1AM_DEV_NO_AUTH=1
Environment=PLC_DRIVER=modbus
Environment=PLC_IP=$PLC_IP
Environment=PLC_PORT=$PLC_PORT
Environment=P1AM_BIND_HOST=127.0.0.1
Environment=P1AM_BIND_PORT=8000
ExecStart=$PY -m uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
"

frontend_unit="[Unit]
Description=P1AM HMI (Vite dev server, :3002)
Documentation=https://github.com/D-sorganization/Tools
After=network-online.target p1am-backend.service
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$SYS/frontend
Environment=PATH=$NODE_BIN_DIR:/usr/local/bin:/usr/bin:/bin
Environment=NODE_ENV=development
ExecStart=$NPM run dev
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
echo "  status:  systemctl status p1am-backend p1am-frontend"
echo "  logs:    journalctl -u p1am-backend -f"
echo "  restart: sudo systemctl restart p1am-backend"
