# Raspberry Pi Bring-up — P1AM Control System

State of this Raspberry Pi's setup for the P1AM-100 control / data-capture
system, prepared 2026-06-11. This is the Pi counterpart to `BENCH_HANDOFF.md`
(which documents the original x86 host "Brick").

## What's already installed and verified (no PLC was present)

- **Repo**: `D-sorganization/Tools` cloned to `~/Repositories/Tools`. The system
  lives at `src/p1am_control_system/`.
- **Python venv**: `~/Repositories/Tools/.venv` (Python 3.13.5) with the focused
  dependency set — fastapi, uvicorn[standard], websockets, sqlmodel, sqlalchemy,
  pymodbus, pydantic, numpy, requests, python-dotenv, python-multipart, keyring,
  defusedxml, pyyaml, PyQt6 6.11, pyqtgraph, httpx, pytest, maturin.
  (The full monorepo `requirements.txt` — torch/mujoco/playwright — was NOT
  installed; it isn't needed to run this system.)
- **tools_core** Rust wheel built locally (maturin, Rust 1.96) and installed into
  the venv — the backend hard-requires `from tools_core import scada`.
- **React HMI** frontend: `npm install` done, production `npm run build` passes.
- **Ethernet**: NetworkManager profile "Wired connection 1" set static to
  `192.168.1.50/24`, no gateway, never-default, IPv6 off. Wi-Fi keeps internet.
- **Smoke test (simulator mode)** passed: backend serves REST (`/api/routing`
  returns the 4-limit interlock contract) and the `/api/stream` WebSocket emits
  full per-scan frames (tags[32], tags_dict, alicats, active_alarms,
  e_stop_active, power_supply). SQLite historian (`backend/dcs_scada.db`) logs.
  Desktop GUI imports cleanly under PyQt6.

## Fixed during setup

- **Stale `dcs_scada.db`** (handoff landmine #4): the db committed in the repo had
  an old `taglog` schema (`no column named tag_name`). Moved aside to
  `backend/dcs_scada.db.stale-bak`; SQLModel recreated a fresh, correct schema.

## Known issue (simulator mode only — does NOT affect real PLC)

- `SimulatedPLCClient` has no `_get_client`, so `write_pid_setpoint` throws every
  scan when `PLC_DRIVER=simulator`. The real `PLC_DRIVER=modbus` path
  (AsyncModbusManager) is unaffected. Harmless for bench/no-hardware use.

## When the P1AM arrives — hookup steps

1. **Cable**: connect the Pi's `eth0` port to the P1AM-ETH shield (direct, or both
   on the same isolated switch — NOT through the shop router/Wi-Fi). Power on the
   P1AM-100.
2. **Verify link**: `cat /sys/class/net/eth0/carrier` should read `1`. The static
   profile activates automatically; confirm with
   `nmcli -t -f NAME,DEVICE,STATE c show --active` (look for "Wired connection 1").
3. **Verify the PLC**: `./run_pi.sh check` (ping + Modbus read of regs 0-3).
4. **Run it**:
   - `./run_pi.sh backend`   — FastAPI + Modbus + historian on :8000
   - `./run_pi.sh frontend`  — React HMI on :3002 (separate terminal)
   - `./run_pi.sh desktop`   — PyQt6 desktop HMI (needs the Pi's display)
5. The backend connects to the PLC in a background task, so it starts even if the
   PLC isn't up yet and connects when it appears.

## Live hardware bring-up results (2026-06-12, AO↔AI loopback)

With the P1AM connected (Ethernet) and AOs jumpered back to AIs, verified from
this Pi:

- `eth0` carrier up, static `192.168.1.50/24` active; PLC pings (~0.16 ms) and
  Modbus TCP :502 open.
- Direct Modbus read of the register map OK. Routing: AO0←TAG_10, AO1←TAG_11,
  AI0→TAG_12, AI1→TAG_13 (TC0-3→TAG_0-3).
- **AO→AI loopback sweep (0→100→0%) passed both channels**, AI tracking AO
  within ~0.14 percentage points (≈0.02 mA on the 4-20 mA scale — ADC noise
  floor). Ran via `calibration/calibrate.py setup` then `sweep --channel 0/1`.
- **Backend confirmed reading the real PLC** (not the sim fallback): with AO1
  held at 40%, the `/api/stream` WebSocket reported TAG_11=40.000 / TAG_13≈40.07.
  Log shows "Connected to PLC successfully in background." (The first connect
  attempt logs one `Failed to connect` warning before the link settles — benign.)

### Operational gotcha — the backend OWNS AO0 / PID 0

`backend/power_supply_integration.py` (`PowerSupplyService.poll`) writes **PID
setpoint 0 every scan** with the power-supply controller's command. When the
PS controller is idle that command is 0%, so **the backend continuously forces
AO0 to 0** while it runs. Observed live: a manually-held AO0=60% was driven back
to 0 the moment the backend connected, while AO1 was untouched.

Implications:
- Don't use AO0 / PID0 / TAG_10 for manual hold or calibration while the backend
  is running — it will fight you. Use AO1 / PID1, or stop the backend first.
- Note the asymmetry is by design (AO0 is the power-supply actuator), and it's
  also why simulator mode throws a harmless `_get_client` error: the real
  `AsyncModbusManager` implements `_get_client` (used by that write path); the
  `SimulatedPLCClient` does not.
- Run only one Modbus master at a time against the P1AM (backend OR the
  calibration CLI), not both — the firmware's Modbus server is single-client.

## Security note

`run_pi.sh` sets `P1AM_DEV_NO_AUTH=1` (bench/dev). The backend binds to
`127.0.0.1` and gates mutating endpoints/E-stop-clear/tag-writes behind an API
key. For any networked deployment, unset `P1AM_DEV_NO_AUTH` and set
`P1AM_API_KEY` (and optionally `P1AM_ADMIN_API_KEY`) — see `backend/auth_config.py`.

## Firmware toolchain (installed + verified on this Pi)

`arduino-cli` 1.5.1 (ARM64) is at `~/.local/bin`. Libraries installed: P1AM 1.0.9,
ArduinoModbus 1.0.9, ArduinoRS485 1.1.1, Ethernet 2.0.2, FlashStorage 1.0.0.
User is in `dialout`, so `/dev/ttyACM0` upload access is ready (no re-login).

**aarch64 workaround (important).** The `P1AM-100:samd` 1.6.21 core pins 2014-era
`arduino:` tools (`arm-none-eabi-gcc@4.8.3-2014q1`, `bossac@1.7.0`) that have NO
aarch64 build, so `arduino-cli core install P1AM-100:samd` fails with "platform is
not available for your OS". The newer `FACTS:samd` core IS aarch64-ready but only
ships the **P1AM-200**, not the -100. Workaround applied here:

1. `arduino-cli core install arduino:samd` (1.8.12) — provides aarch64 GCC
   7-2017q4 (the correct modern SAMD21 compiler), bossac 1.7.0-arduino3, openocd,
   CMSIS 4.5.0, CMSIS-Atmel 1.2.0.
2. P1AM-100 1.6.21 platform files installed manually to
   `~/.arduino15/packages/P1AM-100/hardware/samd/1.6.21/` (verified SHA-256).
3. `platform.local.txt` in that dir overrides `compiler.path`, the CMSIS flags,
   and `tools.bossac.path` / `tools.openocd.path` to the aarch64 tool absolute
   paths. (If you ever `git`-clean or reinstall the core, recreate that file.)

**Verified:** `arduino-cli compile --fqbn P1AM-100:samd:P1AM-100_native firmware`
builds clean at **21% flash** (matches README). Compiler and bossac are both
native aarch64 ELF binaries. Upload not yet tested (no hardware / no ttyACM0).

Build / flash via the launcher:

```bash
./run_pi.sh fw-build     # compile only -> /tmp/p1am-build
./run_pi.sh fw-flash     # compile + upload to /dev/ttyACM0 (PORT=... to override)
```

**Landmine #1 (bricking):** use `P1AM-100:samd:P1AM-100_native` ONLY. The
`arduino:samd:arduino_zero_native` FQBN compiles fine but links the app at 0x0,
overwriting the SAM-BA bootloader and bricking the unit until reflashed. The
launcher hard-codes the correct FQBN.
