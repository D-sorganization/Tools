# Bench Handoff — P1AM Bring-up

Updated **2026-08-29** from protected `main`. PR #3078 merged as `04289469b8a0`; PR #3081 merged as `0add84c54249`. Live hardware state was not requalified.

## Hardware in scope

Facts Engineering P1AM-100 with the P1AM-ETH shield, a P1-04THM thermocouple
module in slot 1, and a P1-4ADL2DAL-1 (2 AI / 2 AO, 4-20 mA) module in slot 2.
External per analog leg: signal conditioners that convert 4-20 mA <-> 0-5 V
(one pair upstream of the loads, one pair downstream feeding back to the AIs).

The PLC sits on an isolated subnet at `192.168.1.100`. The host machine's
Ethernet NIC `enp3s0` is statically configured at `192.168.1.50/24` with no
gateway (Wi-Fi keeps internet on a separate interface).

## What works

- **USB programming/console** on `/dev/ttyACM0`. The unit enumerates as
  `1354:4000 Facts Engineering P1AM-100` when running the application;
  `2341:804d Arduino SA Arduino Zero` if it falls back to the SAM-BA
  bootloader (something has gone wrong — see [firmware/README.md](firmware/README.md)).
- **Modbus TCP** on `192.168.1.100:502`. The host driver
  (`backend/modbus_client.py`) talks to the four-limit interlock contract
  (PR #3081). Verified live: write distinct lolo/low/high/hihi to all 32
  tags, read back, 0 byte-level errors.
- **FastAPI backend** at `backend/main.py`, port 8000. Connects to the PLC on
  startup, polls tags at ~10 Hz, and:
  - Logs every tag value to SQLite (`backend/dcs_scada.db`, table `taglog`).
  - Broadcasts a per-scan JSON frame on the WebSocket at `/api/stream`
    containing `tags` (32-float array), `tags_dict` (TAG_i -> float),
    `alicats`, `active_alarms`, `e_stop_active`.
  - Serves REST endpoints for routing, alarms, PID tuning, MPC simulation,
    trend queries, CSV export. See `GET /docs` (Swagger) for the full surface.

The backend was last left **off** to free CPU during reboot. Re-launch with:

```bash
cd src/p1am_control_system/backend
PLC_DRIVER=modbus PLC_IP=192.168.1.100 PLC_PORT=502 \
  ../../../.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Open PRs (today's work)

| PR                                                                                                                                    | Branch                   | Status                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------- |
| [#3078](https://github.com/D-sorganization/Tools/pull/3078) — `feat(calibration): P1AM analog I/O calibration procedure + helper`     | `calibrate-analog-io`    | Open; needs review. Live-PLC verified.                            |
| [#3081](https://github.com/D-sorganization/Tools/pull/3081) — `fix(p1am): extend interlock contract to 4 limits (lolo/low/high/hihi)` | `fix-interlock-contract` | Open; needs review. Live-PLC verified with 0 errors on roundtrip. |

The calibration helper (PR #3078) cannot drive AOs directly because the
firmware's 10 Hz scan rewrites tag-value registers from `broker.GetTag(i)`
every tick. The helper works around this by repurposing PID 0 and PID 1 as
pass-through drivers (`pv` = unrouted tag, `cv` = AO-routed tag, `kp=1`,
`ki=kd=0`). Verified live; the sweep tracks setpoint exactly.

A first-class "manual override" path in firmware would be a useful follow-up.

## Machine state (host: Brick, user: dieterolson)

Installed and configured during today's session:

- `gh` CLI authenticated to D-sorganization (HTTPS via gh credential helper).
- Repos cloned under `~/Repositories/`: `Controls`, `Tools`, `UpstreamDrift`,
  `Gasification_Model`, `Tools_Private`.
- Python venv at `~/Repositories/Tools/.venv` (Python 3.12.3) with: pymodbus,
  fastapi, sqlmodel, uvicorn, websockets, pydantic, pyserial, ruff, pytest,
  pytest-asyncio, pytest-benchmark, pytest-xdist, httpx, python-multipart,
  maturin, pre-commit, **tools_core** (built locally as a wheel from `rust_core/tools-core`).
- Rust 1.95.0 toolchain at `~/.cargo/bin` (rustup minimal profile).
- arduino-cli 1.5.0 at `~/.local/bin` with the **P1AM-100:samd@1.6.21** board
  package and libraries: P1AM 1.0.9, ArduinoModbus 1.0.9, ArduinoRS485 1.1.1,
  Ethernet 2.0.2, FlashStorage 1.0.0.
- NetworkManager profile "Wired connection 1" set to static `192.168.1.50/24`,
  no gateway, never-default, IPv6 off. Wi-Fi (`wlp5s0`) carries internet.
- `dialout` group membership added for user (takes effect on next login;
  meanwhile `sg dialout -c '...'` works for one-shot commands).

Not yet installed (apt block) — left pending at end of session because the
host got bogged down and the user rebooted:

```bash
sudo apt install -y nodejs npm xvfb libgl1 libglx0 build-essential pkg-config libssl-dev
```

`libgl1-mesa-glx` was renamed to `libgl1` in Ubuntu 24.04 / Mint 22 (Zena) —
don't try the old name, it 404s and aborts the block. Without `npm` the React
HMI at `frontend/` (port 3002) can't start.

## How to verify everything still works after the reboot

1. Confirm the PLC is reachable:

   ```bash
   ping -c 2 192.168.1.100
   ```

   If this fails, check that the Ethernet cable is in the PLC (not the home
   router) and that `nmcli -t -f NAME,DEVICE,STATE c show --active` lists
   `Wired connection 1` as activated.

2. Read tags via Modbus:

   ```bash
   cd ~/Repositories/Tools
   .venv/bin/python -c "
   from pymodbus.client import ModbusTcpClient
   c = ModbusTcpClient('192.168.1.100', port=502, timeout=3)
   assert c.connect()
   r = c.read_holding_registers(address=0, count=4)
   print('tag value regs 0-3:', r.registers)
   c.close()
   "
   ```

3. (Optional) Restart the backend per the command above and curl
   `http://localhost:8000/api/routing` to confirm the four-limit contract
   round-trips.

## Known landmines, ordered by how much time they cost

1. **Wrong FQBN.** See [firmware/README.md](firmware/README.md) — using
   `arduino:samd:arduino_zero_native` instead of `P1AM-100:samd:P1AM-100_native`
   silently overwrites the bootloader and bricks the unit until reflashed.
2. **1200bps touch for serial capture.** The touch is also the upload trigger;
   doing it just to read Serial puts the SAMD in bootloader and requires a
   reflash to recover. Open the monitor _before_ triggering a reset, never
   _to_ trigger one.
3. **Direct tag-value writes don't stick.** The 10 Hz scan rewrites
   regs (i*2, i*2+1) from `broker.GetTag(i)` every tick. To drive an AO from
   the host, configure a pass-through PID (see PR #3078's calibration code)
   or route the AO from a CV tag a real PID is updating.
4. **Stale `dcs_scada.db`** from an earlier schema causes the historian to
   error on every insert (`table taglog has no column named tag_name`). Wipe
   the file and let SQLModel re-create.
5. **CPU pressure from running everything at once.** Uvicorn at 10 Hz plus
   several VS Code / Java language-server processes pushed load average past
   4.5 and made the user's terminal stall. If you start the backend, plan to
   close other heavy windows.

## Recommended next steps

1. **Requalify the bench state** before energizing hardware; the retained live
   verification is from 2026-05-27, not this turnover refresh.
2. **Run the bench calibration** from merged PR #3078 with a DMM end-to-end. The
   procedure document is at `calibration/CALIBRATION.md`.
3. **Wire input routing** for whichever physical sensors get connected
   (registers 100..105). Currently the unit has all routing unmapped, so
   AIs and thermocouples don't reach tags.
4. **Bring up the React HMI** on port 3002 once `npm` is installed.
5. **Consider a governed firmware manual-override** only with explicit safety, provenance, and operator-authorization gates.
