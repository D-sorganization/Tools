# P1AM Heater & Power-Supply Control System

A bench SCADA/HMI that runs a resistive crucible heater (to ~1200 °C) under
closed-loop **on/off temperature control**, plus a programmable power-supply
channel, on a **Facts Engineering P1AM-100 PLC** driven from a **Raspberry Pi**.

> **New here? Read [`docs/SYSTEM_ARCHITECTURE.tex`](docs/SYSTEM_ARCHITECTURE.tex)**
> (a full architecture report — compile with `pdflatex`) and the operator-facing
> [`USER_MANUAL.md`](USER_MANUAL.md). This README is the quick orientation and the
> map of the other docs.

---

## What it is (three tiers)

```
110 V heater ─ P1-08TD2 relay(coil 2) ─ P1AM-100 PLC ─┬─ P1-04THM  (K=Ch1→TAG_0, R=Ch2→TAG_1)
                                        10 Hz scan     └─ P1-4ADL2DAL (4 AI / 2 AO)
                                        Modbus TCP :502
        │ Modbus (10 Hz poll)
   Raspberry Pi ── backend (FastAPI/uvicorn :8000) ── SQLite historian
        │ WebSocket + REST
   React HMI (:3002)
```

1. **PLC firmware** (`firmware/`) — 10 Hz scan loop, a 32-channel tag broker + 4
   PID blocks + four-limit interlocks over **Modbus TCP**, thermocouple/analog I/O,
   and the heater relay coil.
2. **Backend** (`backend/`) — polls the PLC at 10 Hz, runs the supervisory
   **temperature** and **power-supply** controllers (state machine + control law +
   safety), applies a **thermocouple deglitch filter**, logs to a SQLite historian,
   and streams telemetry to the HMI.
3. **HMI** (`frontend/`) — React/Vite app: live trends, controls, alarms, a
   data-explorer analysis suite, and a link-diagnostics feed.

## Hardware contract (single source of truth: [`backend/hardware.py`](backend/hardware.py))

| Tag | Signal | | Coil | Function |
|-----|--------|-|------|----------|
| `TAG_0` | Type-K thermocouple (P1-04THM Ch 1) | | `0` | Save config to flash |
| `TAG_1` | Type-R thermocouple (P1-04THM Ch 2) | | `1` | E-stop reset |
| `TAG_10/11` | Analog outputs (P1-4ADL2DAL AO0/AO1) | | `2` | **Heater relay** (24 V DO → relay → 110 V) |
| `TAG_12/13` | Analog inputs AI0/AI1 (power-supply V/I) | | `3` | THM burnout direction (1 = high-side / fail-safe, default) |
| `TAG_14/15` | Analog inputs AI2/AI3 (4–20 mA conditioned K/R TCs) | | | |
| `TAG_20…25` | Raw 0–5 V card diagnostics | | | |

- Thermocouples: 0–1400 °C, reported in °C. PLC at `192.168.1.100:502`; Pi `eth0`
  static `192.168.1.50/24`.
- The firmware configures **Ch 1 = Type-K, Ch 2 = Type-R** (Ch 3–4 = Type-K); the
  operator selects the controlling channel from the HMI.

## Temperature control (on/off hysteresis)

The heater is a **relay**, not an analog output:

- Relay **ON** when measured ≤ setpoint − deadband; **OFF** when ≥ setpoint + deadband;
  **holds** inside the band. Optional anti-short-cycle min on/off dwell.
- **Safety:** IDLE → ARMED → RUNNING → TRIPPED state machine with a one-way E-stop;
  the relay is force-held **off** in every non-RUNNING state. Trips: **HH_TEMP**
  (high-high on either TC), **TC_DISAGREE** (debounced controlling-vs-other
  cross-check), **TC_FAULT** (non-finite feedback). A **Stop** de-energizes the
  relay immediately (and is unconditional — never gated by a confirm dialog).
- A **deglitch filter** rejects implausible jumps toward a burnout rail and holds
  last-good, escalating to a hard fault after 15 s.

## Run it

Deployed as two **systemd** services (see [`deploy/README.md`](deploy/README.md)):

```bash
sudo bash deploy/install-services.sh          # install/update both units
systemctl status p1am-backend p1am-frontend
journalctl -u p1am-backend -f
```

- `p1am-backend` — FastAPI/uvicorn on `127.0.0.1:8000`, single Modbus master.
- `p1am-frontend` — builds the HMI and serves it with `vite preview` on `:3002`.
- Bench/dev without hardware: `run_pi.sh sim|backend|frontend`; real PLC uses
  `PLC_DRIVER=modbus`. API auth is off on the bench (`P1AM_DEV_NO_AUTH=1`).

## Remote access

| Method | How |
|--------|-----|
| **Raspberry Pi Connect** | Screen sharing at [connect.raspberrypi.com](https://connect.raspberrypi.com) (needs the Wayland/labwc desktop). If it shows a red dot: `systemctl --user restart rpi-connect`. |
| **VNC over Tailscale** | Any VNC viewer → `100.108.70.33:5900` (wayvnc, bound to the Tailscale IP only). |
| **SSH** | `ssh dieterolson@100.108.70.33` |

## Documentation map

| Doc | What it covers |
|-----|----------------|
| [`docs/SYSTEM_ARCHITECTURE.tex`](docs/SYSTEM_ARCHITECTURE.tex) | **Full architecture report** (hardware, firmware, backend, control law, safety, HMI, data, findings). |
| [`USER_MANUAL.md`](USER_MANUAL.md) | Operator manual + in-app Help: tabs, procedures, troubleshooting. |
| [`deploy/README.md`](deploy/README.md) | systemd deployment of the two services. |
| [`firmware/README.md`](firmware/README.md) | Build/flash the PLC firmware; register map. |
| [`calibration/CALIBRATION.md`](calibration/CALIBRATION.md) | Analog-I/O + power-supply calibration procedure. |
| [`scripts/README.md`](scripts/README.md) | Windows NIC-toggle helpers for a dev PC (the Pi uses NetworkManager). |
| [`PI_BRINGUP.md`](PI_BRINGUP.md), [`BENCH_HANDOFF.md`](BENCH_HANDOFF.md) | Historical point-in-time bring-up notes (snapshots; see the architecture report for current state). |

## Known engineering note (thermocouple noise)

R-channel "dips to zero" are **real dropped reads, rare and high-temperature-gated**:
~6 % on R near 1180 °C but ~0 % at 800 °C. Above ~1000–1100 °C the refractory turns
mildly conductive and couples the AC heater's noise onto the low-output Type-R TC.
**Fix:** isolated 4–20 mA TC signal conditioners + single-point grounding. These are
wired to analog inputs AI2/AI3 (`TAG_14/15`) and are selectable as an alternate
control source in the HMI (**Analog Type K / Analog Type R**) alongside the direct
TC-card path. Software also mitigates in either path (deglitch holds last-good;
trends bridge the zeros for display). See the architecture report §*Engineering
findings* for the full analysis.
