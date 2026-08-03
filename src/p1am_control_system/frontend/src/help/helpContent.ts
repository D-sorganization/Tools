import type { TabId } from "../lib/tabs";

/**
 * In-app help content, one entry per HMI tab (keyed by {@link TabId}).
 *
 * Each `body` is a small Markdown subset rendered by `markdownLite`. This is the
 * single source for the tab help shown by the Help button; the full reference is
 * `USER_MANUAL.md` at the app root. A compile-time `Record<TabId, HelpDoc>` means
 * adding a tab without adding help fails the type check (see helpContent.test).
 */
export interface HelpDoc {
  /** Modal heading. */
  title: string;
  /** Markdown body. */
  body: string;
}

/** System-wide primer shown at the foot of every tab's help. */
export const SYSTEM_OVERVIEW = `### About this system

This HMI supervises a **P1AM-100 PLC** (AutomationDirect, SAMD micro) that runs the
crucible bench: a **110 V resistive heater**, a programmable **power supply**, and
**Alicat mass-flow controllers**. The PLC is a Modbus-TCP server at
\`192.168.1.100:502\`; a **FastAPI backend** on the Raspberry Pi polls it at 10 Hz,
runs the control loops and safety interlocks, logs to a SQLite historian, and streams
live data to this browser HMI over a WebSocket.

**I/O modules:** P1-04THM thermocouples (Ch1 = type K → \`TAG_0\`, Ch2 = type R →
\`TAG_1\`), P1-08TD2 discrete outputs (heater relay = coil 2), and P1-4ADL2DAL analog
in/out (power-supply monitor and command). Full details are in \`USER_MANUAL.md\`.`;

export const HELP: Record<TabId, HelpDoc> = {
  operator: {
    title: "Representative Operator Overview",
    body: `A **synthetic, non-live-control** workspace demonstrating professional
overview-to-detail navigation without plant names, parameters, or control logic.

### Navigation and state
- Select a generic asset to open its reusable faceplate with value, quality,
mode, alarm, interlock, and trend-drill-down context.
- Protection cards keep control, interlock, and independent-protection
categories distinct and show deterministic first-out consequences.
- Any managed bypass is displayed in a persistent banner with actor, reason,
and expiry. Items marked **Non-bypassable** cannot be bypassed through this UI.

### Reusable product demonstrations
- Simulator-only procedures expose bounded start, run, hold, stop, abort, and
recovery states with attributable transitions.
- Connector health identifies the responsible plugin; a failed connector
degrades only its own tags and all failed commands are rejected closed.
- Notification and recovery summaries show escalation recipients, delivery
controls, single command authority, clock reliability, and explicit RTO/RPO.
- The advisory workspace identifies the synthetic model and data, shows bounded
constraints and confidence, verifies replay checksums, and records attributable
operator review dispositions. It has no authoritative command or write path.
These are representative contracts, not claims of deployed redundant hardware,
validated plant models, or approved advanced control.`,
  },

  temperature: {
    title: "Heater Controls",
    body: `Controls the **110 V resistive crucible heater** through a single 24 V
discrete output → relay, using a thermocouple for feedback.

### How control works
- **On/off (bang-bang) with hysteresis.** The relay closes when the measured
temperature falls to \`setpoint − deadband\` and opens at \`setpoint + deadband\`,
so it doesn't chatter around the target.
- **Start / Stop.** *Start* arms the heater and applies the shown target; *Stop*
opens the relay immediately. A setpoint is committed by pressing **Enter** (or the
± steps) — there is no separate apply step.
- **Thermocouple select.** Choose whether **type K** or **type R** drives control.
Both live readings are shown; switching is smooth and does not stop the heater.

### Safety interlocks
- **High-high (HH) cutoff** latches the heater OFF at the limit and trips the
controller — checked on **both** thermocouples, so a dead control probe can't hide
a real over-temp. The HH value is inline-editable in the live strip.
- **TC_DISAGREE** trips if the controlling probe reads cold while the other reads
hot (a stuck/dead control sensor).
- **Deglitch filter** holds the last-good reading through a momentary sensor
dropout and fails safe (TC_FAULT trip) if a fault persists — an amber banner warns
while it is holding.
- **E-stop** and a trip always force the relay off; a trip must be acknowledged.

### Reading the trend
Both thermocouples plot together (active solid, other dimmed) with the setpoint and
HH reference lines. A linear-fit window (in minutes) gives the **heat-up rate** in
°C/min and °C/hr. Settings and the last setpoint persist across restarts.`,
  },

  powerSupply: {
    title: "Power Supply",
    body: `Commands and monitors the programmable **power supply** via the PLC's
analog I/O (command out on \`TAG_10\`, current/voltage feedback on \`TAG_12/13\`).

### Controls
- **Setpoint** in amps (or watts, converted via measured voltage). Committed on
**Enter** or the ± steps; the server clamps it to the configured band.
- **Output limit (%)** caps delivery regardless of setpoint — a hard ceiling. It is
inline-editable in the live strip and via the dedicated limit control.
- **Permissive** must be ON before any setpoint takes effect (server-enforced).

### Reading it
The live strip shows commanded output %, measured current and voltage, and the
active limit. The effective maximum current = *output-limit %* × full-scale, so the
setpoint band is capped by the limit — raise the limit to deliver more. Config
(limits, alarms, ramp rate) persists across restarts.`,
  },

  trends: {
    title: "Trends & Monitors",
    body: `The live overview: real-time trend chart plus at-a-glance monitors for
every signal the PLC reports (\`TAG_0\`…\`TAG_31\`), updated each scan (10 Hz).

### Trend chart
- Pick which signals to plot; the chart windows by real wall-clock time, so the
span and any curve-fit slope are correct regardless of poll rate.
- **Freeze** to inspect a moment without new data pushing it off-screen; zoom and
scroll back through the buffer; export the window to CSV or an image.
- Manual or auto Y-axis; a curve-fit overlay with the fitted equation and R².

### Monitors
Compact per-signal tiles show current value and alarm state. Thermocouples read in
°C; analog channels read in engineering units per their scaling. Use this tab as
the default "is everything nominal?" screen.`,
  },

  explorer: {
    title: "Data Explorer",
    body: `Offline analysis of **historian** data (the SQLite log of every scan).
Load a dataset by tag and time range, then explore it without touching live control.

### What you can do
- **Plots:** line, scatter, histogram, heatmap, and spectrum (FFT) views.
- **Pipeline:** chain transforms (filter, resample, derive) on the loaded signals.
- **Correlation** across signals, and **export** to CSV or a saved session.

This is the tab for post-run diagnostics — e.g. reconstructing a heat-up curve,
checking a thermocouple's behavior over a run, or comparing channels. It reads
history only; it never commands the plant.`,
  },

  controllers: {
    title: "PID & Mass Flow",
    body: `Supervises the **PID loops** running on the PLC and the **Alicat
mass-flow controllers** (MFCs).

### PID loops
View each loop's setpoint, process value, and output, and adjust tuning where
permitted. On this bench PID0 is a pass-through that routes a command to the
power-supply analog output (\`TAG_10\`).

### Mass-flow controllers
Each Alicat MFC reports flow, pressure, and temperature and accepts a flow
setpoint. Devices are polled independently of the PLC scan and surfaced here with
their live readings and setpoint entry.`,
  },

  routing: {
    title: "Signal Routing",
    body: `Shows and edits the PLC's **signal routing matrix** — which physical
inputs map to which tags, which tags drive which outputs, and the interlock limits
attached to each tag.

### Reading the matrix
Rows and columns are the \`TAG_0\`…\`TAG_31\` channels. Input routing maps module
channels onto tags (e.g. thermocouple Ch1 → \`TAG_0\`); output routing maps tags to
actuators (e.g. \`TAG_10\` → power-supply AO). The backend syncs this from the PLC
on connect and re-asserts safe state on reconnect.

Changing routing changes what the control loops read and drive — treat it as a
configuration action, not a runtime knob.`,
  },

  tuning: {
    title: "Tuning & MPC",
    body: `Advanced controller tuning and model-predictive-control (MPC)
experimentation for the PID loops.

Use this tab to trial tuning parameters and predictive strategies offline or in a
supervised setting. Applied changes flow to the PLC's PID configuration; the same
safety interlocks (E-stop, HH cutoff, trips) remain in force regardless of the
tuning in use.`,
  },

  events: {
    title: "Events & Alarms",
    body: `The chronological **event and alarm log** persisted in the historian
(\`eventlog\` table).

### What's recorded
- **Alarms** when a tag crosses a configured limit (Lo/LoLo/Hi/HiHi), with the
tag, state, value, and timestamp.
- Active alarms are also surfaced live in the header banner across all tabs.

Use this tab to answer "what happened, and when?" after an event — e.g. to see the
sequence of limit crossings around a trip. Entries are timestamped in the Pi's
local time.`,
  },

  ladder: {
    title: "Ladder Explorer",
    body: `A read-oriented view of the PLC's **ladder logic** — the rungs the P1AM
firmware evaluates each scan (contacts, coils, timers, and the safety interlock
rungs).

Use it to understand *why* the PLC is doing what it is: which conditions energize
the heater relay, how the E-stop and trip interlocks are wired, and how inputs map
to the coil outputs. The heater relay (coil 2) is only energized when the
temperature controller commands it **and** no interlock has tripped.`,
  },

  hierarchy: {
    title: "Plant Hierarchy",
    body: `A structural tree of the plant model: **areas → units → equipment →
tags**. It organizes the flat \`TAG_0\`…\`TAG_31\` list into the physical assets they
belong to (heater, power supply, thermocouples, MFCs).

Use it to navigate by asset rather than by raw tag number, and to see how a given
piece of equipment's signals are grouped. It is a navigation and reference view; it
does not command the plant.`,
  },

  diagnostics: {
    title: "Signal Diagnostics",
    body: `Low-level signal health for troubleshooting the field wiring and modules.

### What to look for
- **Raw vs scaled values** per channel, to separate a wiring/scaling problem from a
control problem.
- **Thermocouple health:** a reading that drops to **0 °C** is the P1-04THM's
low-side **burnout** response to an open input — an intermittent connection, a
high-resistance/degraded element, or (at very high temperature) insulation
breakdown in the probe. A reading stuck near ambient while the vessel is hot means
the junction isn't thermally coupled.
- **Analog inputs** in raw volts/counts for the power-supply monitor signals.

This tab is where you confirm a sensor is misbehaving before trusting — or
distrusting — a control reading.`,
  },
};
