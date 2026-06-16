#!/usr/bin/env bash
# Turnkey launcher for the P1AM control / data-capture system on this Raspberry Pi.
#
#   ./run_pi.sh backend      # FastAPI backend + Modbus + historian (port 8000)
#   ./run_pi.sh frontend     # React HMI dev server (port 3002)
#   ./run_pi.sh desktop      # PyQt6 desktop HMI (needs a display)
#   ./run_pi.sh sim          # backend in SIMULATOR mode (no PLC needed)
#   ./run_pi.sh check        # ping + Modbus read against the PLC
#   ./run_pi.sh fw-build     # compile firmware (aarch64 toolchain)
#   ./run_pi.sh fw-flash     # compile + upload firmware to /dev/ttyACM0
#
# Real-PLC settings (from BENCH_HANDOFF.md): PLC at 192.168.1.100:502, host at
# 192.168.1.50/24 on eth0. Override with PLC_IP / PLC_PORT env vars if needed.
set -euo pipefail

REPO=~/Repositories/Tools
SYS="$REPO/src/p1am_control_system"
VENV="$REPO/.venv"
PY="$VENV/bin/python"

PLC_IP="${PLC_IP:-192.168.1.100}"
PLC_PORT="${PLC_PORT:-502}"

# Bench/dev auth opt-out. For a networked deployment, unset this and set
# P1AM_API_KEY / P1AM_ADMIN_API_KEY instead (see backend/auth_config.py).
export P1AM_DEV_NO_AUTH="${P1AM_DEV_NO_AUTH:-1}"

cmd="${1:-help}"
case "$cmd" in
  backend)
    cd "$SYS/backend"
    echo "Backend -> Modbus PLC at $PLC_IP:$PLC_PORT, listening on 127.0.0.1:8000"
    PLC_DRIVER=modbus PLC_IP="$PLC_IP" PLC_PORT="$PLC_PORT" \
      P1AM_BIND_HOST=127.0.0.1 P1AM_BIND_PORT=8000 \
      "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8000
    ;;
  sim)
    cd "$SYS/backend"
    echo "Backend -> SIMULATOR mode (no PLC), listening on 127.0.0.1:8000"
    PLC_DRIVER=simulator P1AM_BIND_HOST=127.0.0.1 P1AM_BIND_PORT=8000 \
      "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8000
    ;;
  frontend)
    cd "$SYS/frontend"
    echo "React HMI -> http://localhost:3002 (proxies backend on :8000)"
    npm run dev
    ;;
  desktop)
    cd "$SYS"
    echo "Launching PyQt6 desktop HMI..."
    "$PY" launch_desktop.py
    ;;
  check)
    echo "== ping $PLC_IP =="
    ping -c 2 -W 2 "$PLC_IP" || echo "PLC not reachable (check cable / power / eth0 link)"
    echo "== Modbus read regs 0-3 =="
    "$PY" - <<PYEOF
from pymodbus.client import ModbusTcpClient
c = ModbusTcpClient("$PLC_IP", port=$PLC_PORT, timeout=3)
if not c.connect():
    print("Modbus connect FAILED"); raise SystemExit(1)
r = c.read_holding_registers(address=0, count=4)
print("tag value regs 0-3:", getattr(r, "registers", r))
c.close()
PYEOF
    ;;
  fw-build)
    export PATH="$HOME/.local/bin:$PATH"
    echo "Compiling firmware (FQBN P1AM-100:samd:P1AM-100_native, aarch64 toolchain)..."
    arduino-cli compile --fqbn P1AM-100:samd:P1AM-100_native \
      "$SYS/firmware" --output-dir /tmp/p1am-build
    ;;
  fw-flash)
    export PATH="$HOME/.local/bin:$PATH"
    PORT="${PORT:-/dev/ttyACM0}"
    echo "Compiling + uploading firmware to $PORT ..."
    arduino-cli compile --fqbn P1AM-100:samd:P1AM-100_native \
      "$SYS/firmware" --output-dir /tmp/p1am-build
    # CORRECT FQBN ONLY. arduino:samd:arduino_zero_native bricks the unit.
    arduino-cli upload --fqbn P1AM-100:samd:P1AM-100_native \
      --port "$PORT" --input-dir /tmp/p1am-build "$SYS/firmware"
    echo "Verify: lsusb | grep -i facts  -> 1354:4000 = app running (good)"
    ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
