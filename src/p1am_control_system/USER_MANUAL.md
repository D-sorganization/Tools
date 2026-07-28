# P1AM Crucible Heater Control System — User Manual

A supervisory control and data-acquisition (SCADA) system for a bench crucible
rig, built around a **P1AM-100 PLC**, a **Raspberry Pi** backend, and a browser
HMI. This manual documents the hardware, the control and safety logic, and every
screen of the operator interface.

> **Quick help:** every HMI tab has a **Help** button (📖) in the top-right of the
> header that opens the reference for that specific tab. This document is the full
> version of that help.

---

## 1. What this system controls

| Subsystem | Actuator | Feedback | Purpose |
| --- | --- | --- | --- |
| **Crucible heater** | 110 V AC resistive element via a 24 V DO → relay | Type-K + type-R thermocouples | Heat the crucible to a setpoint (0–1400 °C) |
| **Power supply** | Programmable supply via 0–5 V analog command | Current + voltage monitor | Deliver a commanded current/power |
| **Mass flow** | Alicat MFCs (serial) | Flow / pressure / temperature | Meter process gas |

The **heater is the primary controlled process**: a resistive element wraps the
crucible; the PLC switches it on and off through a relay, using thermocouple
temperature feedback to hold a setpoint with an on/off (bang-bang) control law.

---

## 2. System architecture

```
[Thermocouples / AI]         [Heater relay / AO]
        │                            ▲
        ▼                            │
   ┌──────────────── P1AM-100 PLC ────────────────┐
   │  firmware.ino @ 10 Hz scan · Modbus-TCP server │
   └──────────────────────┬─────────────────────────┘
                          │ Modbus TCP  192.168.1.100:502
                          ▼
   ┌──────────── Raspberry Pi (backend) ───────────┐
   │  FastAPI / uvicorn  :8000   (loopback)         │
   │  · polls PLC @ 10 Hz   · control loops          │
   │  · safety interlocks    · deglitch filter       │
   │  · SQLite historian     · WebSocket stream      │
   └──────────────────────┬─────────────────────────┘
                          │ HTTP + WebSocket
                          ▼
   ┌──────────── Browser HMI (this app) ───────────┐
   │  React / Vite   served on  :3002                │
   └────────────────────────────────────────────────┘
```

- **PLC firmware** scans every **100 ms (10 Hz)**: it reads the thermocouple and
  analog modules, publishes them to Modbus registers (`TAG_0`…`TAG_31`), and drives
  the outputs from the coils/registers the backend writes.
- **Backend** (Python, on the Pi) is the authority for control and safety. It polls
  the PLC over Modbus TCP at 10 Hz, runs the temperature and power-supply
  controllers, enforces interlocks, logs every scan to SQLite, and streams live
  data to the HMI over a WebSocket (`/api/stream`, with `/api/snapshot` as a
  fallback).
- **HMI** is a static production build served by `vite preview` on port **3002**.
  It only displays data and sends operator commands; all control decisions are made
  by the backend/PLC.

Both firmware and backend run at 10 Hz, so end-to-end latency from a temperature
change to a relay decision is on the order of a couple of scans.

---

## 3. The PLC and I/O modules

**CPU:** P1AM-100 (AutomationDirect), an Arduino-form-factor controller on a SAMD
micro, acting as a Modbus-TCP **server** at `192.168.1.100:502`. Firmware FQBN:
`P1AM-100:samd:P1AM-100_native`.

**Coils (discrete commands from the backend):**

| Coil | Function |
| --- | --- |
| 0 | Save-to-flash |
| 1 | E-stop reset |
| **2** | **Heater relay command** (temperature controller) |

**Modules on the backplane and their tag mapping:**

| Slot | Module | Channels → tags | Notes |
| --- | --- | --- | --- |
| THM | **P1-04THM** | Ch1 (type K) → `TAG_0`, Ch2 (type R) → `TAG_1`, Ch3–4 (type K) → `TAG_2/3` | Celsius, **low-side burnout**, on-module linearization |
| DO | **P1-08TD2** | Heater relay = **coil 2** | 24 V discrete out → relay → 110 V heater |
| ANA | **P1-4ADL2DAL** | AI0/AI1 → `TAG_12/13`, AO0/AO1 ← `TAG_10/11` | Power-supply monitor + command |

**Signal scaling.** Every analog channel is carried as **0–100 % of full scale**.
The P1-04THM does per-type linearization on-module and the firmware reads degrees C
directly; the broker scales °C → % (×100/1400) and the backend scales it back
(×1400/100). So a 1400 °C full-scale type-K channel reads 50 % → 700 °C. Because K
and R share the same 1400 °C full scale, they differ only by which tag they land
on — not by how percent is converted to °C.

---

## 4. The heater control loop

**Actuator chain:** temperature controller → Modbus coil 2 → 24 V DO → relay →
110 V AC resistive element around the crucible.

**Control law — on/off with hysteresis (bang-bang):**

- Relay turns **ON** when measured ≤ `setpoint − deadband`.
- Relay turns **OFF** when measured ≥ `setpoint + deadband`.
- Inside the band the relay **holds** its previous state, so it doesn't chatter.
- **Anti-short-cycle:** after a switch the relay is held for at least the configured
  minimum on/off time, capping how often the heater cycles.

**Operator model — Start / Stop:**

1. **Start** arms the heater (permissive on) and applies the shown target.
2. Enter a **setpoint** (°C) and press **Enter** (or use the ± steps) — it applies
   immediately when running, or is staged and applied on Start when stopped.
3. **Stop** opens the relay immediately.

**Persistence:** the setpoint, thermocouple selection, and safety limits are stored
durably and **recalled on restart** — but the controller always comes back **IDLE**
(stopped). Recalled settings never auto-energize the heater; the operator must press
Start.

---

## 5. Thermocouples

Two probes read the crucible: **type K** (Ch1 → `TAG_0`) and **type R** (Ch2 →
`TAG_1`). The operator selects which one **controls** the heater; both are always
read, displayed, and plotted, and the non-controlling one is used as an independent
safety reference.

### Selecting and switching
Switching the controlling probe (K ↔ R) is **smooth** — it does not stop the heater.
The live value of each probe is shown next to its selector so a dead or stuck sensor
is obvious at a glance.

### Failure modes and what they look like
- **Reads 0 °C:** the P1-04THM's **low-side burnout** response to an **open input**
  (loose/broken connection, or a high-resistance/degraded element).
- **Stuck near ambient while the vessel is hot:** the junction is not thermally
  coupled (probe not inserted/seated), or the leads are reversed.
- **Intermittent drops to 0 at high temperature:** an open that appears only when
  hot — thermal expansion opening a marginal joint, a degrading element near its
  temperature limit, or (with a grounded/​degrading probe) insulation-resistance
  breakdown inside the sheath. This is a **wiring/probe** fault, not a control bug.

### High-temperature notes
At ~1300 °C type K is near the top of its practical range; elements can develop
high-resistance or intermittent opens. For sustained high-temperature work, prefer
an **ungrounded (isolated) junction**, adequate wire gauge, and a probe rated for the
temperature (type N, or a properly sheathed type R/S).

---

## 6. Safety systems

Multiple independent layers force the heater off; **any one** of them wins.

- **E-stop.** Latches the heater relay off and disarms the controller. Re-asserted
  automatically on a PLC reconnect. Cleared only by an explicit operator reset.
- **High-high (HH) cutoff.** A hard temperature limit that latches the controller
  **TRIPPED** and forces the relay off. Evaluated on **both** thermocouples, so a
  dead control probe cannot mask a real over-temperature. The limit is
  inline-editable in the live strip.
- **TC_FAULT.** A non-finite (open/garbage) control reading while running trips the
  heater rather than treating it as "cold."
- **TC_DISAGREE (cross-check).** While running, if the controlling probe reads cold
  (< 100 °C) while the other reads clearly hot (≥ 200 °C) for several consecutive
  scans, the heater trips — the signature of a stuck/dead control sensor.
- **Deglitch filter.** Rejects an implausible drop to ~0 (burnout) from a hot
  last-good value and **holds** the last-good so the control law never acts on a
  glitch; if the fault **persists past ~15 s** it escalates to a TC_FAULT trip
  (fail-safe). An amber banner warns while the filter is holding.
- **De-energize on comms loss.** A failed relay-OFF write is retried and raises a
  comms alarm; a momentary read hiccup **holds the last-good** value instead of
  substituting fabricated data.

A trip **latches** and must be **acknowledged** before the heater can run again.

---

## 7. The HMI — tab-by-tab reference

The header hosts the connection status, theme toggle, **Help** (📖, per-tab),
settings, and the **E-STOP**. Tabs can be reordered, hidden, and shown from
settings.

- **Heater Controls (`temperature`).** Start/Stop the heater, set the target,
  choose the controlling thermocouple, watch both probes trend with the heat-up
  rate, and edit the HH cutoff inline. Primary operating screen for the heater.
- **Power Supply (`powerSupply`).** Command the supply setpoint (A/W), set the hard
  output-limit %, and monitor current/voltage. Permissive gates the setpoint.
- **Trends & Monitors (`trends`).** Live multi-signal trend chart (freeze, zoom,
  export, curve-fit) plus per-signal monitor tiles. The default "all nominal?"
  screen.
- **Data Explorer (`explorer`).** Offline analysis of historian data — line,
  scatter, histogram, heatmap, and spectrum plots; a transform pipeline;
  correlation; CSV/session export. Read-only.
- **PID & Mass Flow (`controllers`).** Supervise the PLC PID loops and the Alicat
  mass-flow controllers.
- **Signal Routing (`routing`).** View/edit the input→tag and tag→output routing
  matrix and per-tag interlock limits.
- **Tuning & MPC (`tuning`).** Trial PID tuning and model-predictive strategies.
- **Events & Alarms (`events`).** Chronological log of alarm limit crossings from
  the historian.
- **Ladder Explorer (`ladder`).** Read the PLC ladder logic — how inputs, interlocks,
  and the heater coil are wired.
- **Plant Hierarchy (`hierarchy`).** Navigate signals by asset (area → unit →
  equipment → tag).
- **Signal Diagnostics (`diagnostics`).** Low-level raw-vs-scaled signal health for
  troubleshooting wiring and thermocouples.

---

## 8. Operating procedures

### Start a heat run
1. Confirm the HMI header shows **CONNECTED** and the E-stop is clear.
2. Open **Heater Controls**. Check both thermocouple readings are live and sane.
3. Select the controlling thermocouple (default **type K**).
4. Enter the target °C and press **Enter**, then press **Start**.
5. Watch the trend; the heat-up-rate box shows °C/min and °C/hr over your chosen
   fit window.

### Recover from a trip
1. Read the banner / **Events & Alarms** to see which trip fired (HH, TC_FAULT,
   TC_DISAGREE).
2. Resolve the cause (let it cool below HH, fix the sensor, etc.).
3. **Acknowledge** the trip, then Start again.

### Redeploy after a code/config change
The services must restart to load new backend code or a new HMI build. A restart
stops the heater (it returns **IDLE** with the setpoint recalled). Coordinate it for
a moment the heater can pause, then:

```bash
sudo systemctl restart p1am-backend p1am-frontend
systemctl is-active p1am-backend p1am-frontend   # both -> active
```

The frontend rebuilds on start; give it ~30 s to bind port 3002.

---

## 9. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Reading drops to **0 °C** | Open input → module burnout | Check the probe/connections; the deglitch filter protects control meanwhile |
| Drops only at **high temp** | Connection/element opens with thermal expansion; insulation breakdown | Re-terminate hot-side joints; inspect/replace the element; use an isolated-junction probe |
| Probe stuck near **ambient** while hot | Junction not coupled / leads reversed | Re-seat/insert the probe; verify polarity and extension-wire type |
| Heater **won't start** | Not permissive, tripped, or E-stopped | Acknowledge trips, clear E-stop, press Start |
| HMI shows **OFFLINE** | Backend/PLC comms down | Check services (`systemctl`), the PLC network, and Modbus at `192.168.1.100:502` |
| **TC_DISAGREE** trip | Control probe reads cold while other reads hot | Don't control off a dead probe; fix the sensor |

### Is a thermocouple problem the PLC, the sampling rate, or the setup?
The burnout-zeros are the **module's open-circuit detection** reporting an open
input, so the answer is usually the **field side**, not the sampling rate:

- **Sampling rate is not the cause of the zeros.** Firmware reads at 10 Hz, faster
  than the P1-04THM's own conversion, so you *oversample* the module — this changes
  how many zeros you observe, not whether they occur.
- **With tight connections, suspect the probe's high-temperature electrical
  behavior:** rising loop resistance or insulation-resistance breakdown at high
  temperature can trip the module's low-side burnout even with good terminals.
- **Discriminating tests:** move the probe to THM Ch3/Ch4 (if the fault follows the
  probe it's the probe/wire; if it stays it's the channel/module); measure loop
  resistance cold vs hot; try a known-good, ungrounded-junction probe.

---

## 10. Deployment and maintenance

- **Services:** `p1am-backend` (FastAPI/uvicorn) and `p1am-frontend` (`vite
  preview`), both `Restart=always` under systemd. Install via
  `deploy/install-services.sh`.
- **Bench mode:** `P1AM_DEV_NO_AUTH=1` (admin endpoints unauthenticated),
  `PLC_DRIVER=modbus`. When the PLC is offline the backend runs a simulator so the
  HMI still animates.
- **Historian:** SQLite (`dcs_scada.db`, WAL, retention-capped). Tag values are
  logged to `taglog` — **decimated by the capture-throttle interval, not every
  10 Hz scan** — and alarm crossings to `eventlog`.
- **Remote access:** Raspberry Pi Connect (screen share), VNC over Tailscale
  (`100.108.70.33:5900`), and SSH (`ssh dieterolson@100.108.70.33`).
- **Architecture:** see the top-level `README.md` and the full report
  `docs/SYSTEM_ARCHITECTURE.tex` (compile with `pdflatex`).
- **Tuning knobs (env):** `P1AM_POLL_INTERVAL_S` (default 0.1 s), the lightweight
  poll interval, and the capture/log-throttle interval.

---

*This manual is the full version of the in-app Help. Open any tab and press the
Help button (📖) for that tab's quick reference.*
