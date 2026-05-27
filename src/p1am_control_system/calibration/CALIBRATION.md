# Analog I/O Calibration Procedure

End-to-end procedure for calibrating the P1AM-100 analog outputs and inputs
together with their external 4-20 mA ↔ 0-5 V signal conditioners, before
connecting the system to a real field load.

## Why this exists

Two-stage calibration is required because the loop has three independently
adjustable links: the PLC AO DAC, the output-side signal conditioner (4-20 mA
→ 0-5 V), and the input-side signal conditioner (0-5 V → 4-20 mA). Each
conditioner has a zero and span pot. Until all three are tuned and verified
end-to-end, a tag value of `50%` may not correspond to 12 mA on either side.
Calibrate the outputs first against a DMM, then use the calibrated outputs as
the reference source for input calibration in a loopback.

## Hardware in scope

| Slot | Module        | Channels   | Purpose                |
| ---- | ------------- | ---------- | ---------------------- |
| 2    | P1-4ADL2DAL-1 | AI 0, AI 1 | 4-20 mA analog inputs  |
| 2    | P1-4ADL2DAL-1 | AO 0, AO 1 | 4-20 mA analog outputs |

External per analog leg: one signal conditioner upstream (4-20 mA → 0-5 V) and
one downstream (0-5 V → 4-20 mA), each with zero and span pots.

## Tag assignments used in this procedure

| Tag            | Role                                                   |
| -------------- | ------------------------------------------------------ |
| TAG_10         | drives AO ch 0 (via PID 0 pass-through)                |
| TAG_11         | drives AO ch 1 (via PID 1 pass-through)                |
| TAG_12         | reads AI ch 0                                          |
| TAG_13         | reads AI ch 1                                          |
| TAG_30, TAG_31 | reserved as unrouted PV tags for the pass-through PIDs |

PID 0 and PID 1 are repurposed during calibration as pass-through drivers for
the two AOs. **Do not configure real process control on PID 0 or PID 1 while
calibration is in progress** — the calibration helper will overwrite their
configuration each time it's run.

## Scaling reference

PLC AO percent → loop mA (firmware contract, linear over 0-100%):

```
mA = 4 + 0.16 × percent
```

Properly calibrated signal conditioner output (4-20 mA → 0-5 V):

```
V = 0 + 0.05 × percent
```

| Percent |  AO mA | Cond Out V |
| ------: | -----: | ---------: |
|       0 |  4.000 |      0.000 |
|      25 |  8.000 |      1.250 |
|      50 | 12.000 |      2.500 |
|      75 | 16.000 |      3.750 |
|     100 | 20.000 |      5.000 |

## Tolerance

**±1% of full span**: ≈ ±0.16 mA on the current loop, ≈ ±0.05 V at the
conditioner output. Use a DMM with at least 4½-digit resolution.

## Prerequisites

1. PLC powered up, reachable at `192.168.1.100:502` from the calibration host:
   ```
   ping -c 2 192.168.1.100
   ```
2. The signal conditioners are powered and wired:
   - Output side: AO+ / AO- → conditioner input; conditioner output → field
     load (which should be **disconnected from any process power supply** for
     calibration).
   - Input side: field source → conditioner input; conditioner output →
     AI+ / AI-.
3. DMM available with mA (in-series) and V (across terminals) modes.
4. One jumper wire long enough to bridge an AO-side conditioner output to an
   AI-side conditioner input for the loopback steps.
5. The calibration venv is active:
   ```
   source ~/Repositories/Tools/.venv/bin/activate
   cd ~/Repositories/Tools/src/p1am_control_system/calibration
   ```

## Safety notes

- Disconnect AO loops from any downstream power supply, valve, or actuator
  before starting. Calibration drives outputs across the full 4-20 mA range
  in arbitrary order.
- The firmware's safety interlocks default to broad limits (±99999) on a
  fresh boot. If a saved configuration with tight limits exists, widen them
  before sweeping, or clear flash (see helper `--clear-flash`).
- The Inhibit GPIO (D6) drives LOW during normal operation. If it goes HIGH
  during calibration, an interlock has tripped and all AOs will be forced to
  0% — investigate before resuming.
- Never bypass interlocks in production. They are only widened for calibration
  because the field load is disconnected.

## Procedure

### Step 1 — Verify comms and set routing

```
python calibrate.py status
python calibrate.py setup
python calibrate.py status
```

`setup` configures:

- Output routing: AO 0 ← TAG_10, AO 1 ← TAG_11
- Input routing: AI 0 → TAG_12, AI 1 → TAG_13
- PID 0: pv=TAG_30 (unused), cv=TAG_10, kp=1, ki=kd=0, sp=0 → pass-through for AO 0
- PID 1: pv=TAG_31 (unused), cv=TAG_11, kp=1, ki=kd=0, sp=0 → pass-through for AO 1

After running, `status` should show the routing in place and TAG_10/TAG_11
at 0% (i.e., AOs at 4 mA).

### Step 2 — Calibrate AO ch 0 zero and span

Wiring for this step:

```
PLC AO0+ ─[DMM in series, mA mode]─┐
PLC AO0- ──────────────────────────┴─→ Sig Cond #0 In
                                       Sig Cond #0 Out ─[DMM2 across, V mode]─ (open)
```

2.1 Drive 0% (target 4 mA, 0 V):

```
python calibrate.py ao --channel 0 --percent 0
```

- DMM (current): expect 4.000 mA ± 0.16 mA. Record: `____.____ mA`
- DMM2 (voltage at conditioner output): expect 0.000 V ± 0.05 V. Record: `____.____ V`
- If V is off: turn the **zero** pot on Sig Cond #0 until V = 0.000 V.

  2.2 Drive 100% (target 20 mA, 5 V):

```
python calibrate.py ao --channel 0 --percent 100
```

- DMM: 20.000 mA ± 0.16 mA. Record.
- DMM2: 5.000 V ± 0.05 V.
- Turn the **span** pot until V = 5.000 V.

  2.3 Iterate. Zero and span interact — adjusting span shifts the zero point
  slightly and vice versa. Repeat 2.1 and 2.2 alternately until both endpoints
  are within tolerance in the same pass. Typically 2-3 iterations.

  2.4 Linearity spot-check:

```
python calibrate.py ao --channel 0 --percent 25
python calibrate.py ao --channel 0 --percent 50
python calibrate.py ao --channel 0 --percent 75
```

Verify against the reference table. Any point off by more than ±1% indicates
either a nonlinear DAC, a damaged conditioner, or wiring trouble.

### Step 3 — Calibrate AO ch 1 zero and span

Repeat Step 2 with `--channel 1`. Wire the DMMs to AO1 and Sig Cond #1.

### Step 4 — Calibrate AI ch 0 (loopback)

Wiring change: bridge **Sig Cond #0 Out** directly to **Sig Cond In #0 In**
with the jumper. The output side feeds the input side; no field source.

```
Sig Cond #0 Out ─[jumper]─→ Sig Cond In #0 In
                            Sig Cond In #0 Out ─[DMM in series, mA mode]─ PLC AI0
```

4.1 Drive 0% from the calibrated AO 0:

```
python calibrate.py ao --channel 0 --percent 0
python calibrate.py ai --channel 0
```

- DMM at AI loop: 4.000 mA ± 0.16 mA. (Should be correct from Step 2; if
  not, the input-side signal conditioner is the suspect.)
- `ai` command output: expect TAG_12 = 0.0% ± 1%.
- If TAG_12 reads outside tolerance: adjust the **zero** pot on Sig Cond In #0.

  4.2 Drive 100%:

```
python calibrate.py ao --channel 0 --percent 100
python calibrate.py ai --channel 0
```

- DMM: 20.000 mA. TAG_12: 100.0% ± 1%.
- Adjust the **span** pot on Sig Cond In #0.

  4.3 Iterate until both endpoints are in spec.

  4.4 Linearity check at 25/50/75.

### Step 5 — Calibrate AI ch 1 (loopback)

Repeat Step 4 with `--channel 1`, using the AO 1 → Sig Cond #1 → jumper →
Sig Cond In #1 → AI 1 path.

### Step 6 — End-to-end sweep verification

```
python calibrate.py sweep --channel 0
python calibrate.py sweep --channel 1
```

`sweep` walks each AO through 0 → 25 → 50 → 75 → 100 → 75 → 50 → 25 → 0
with a 1-second dwell at each step and prints the corresponding AI reading.
For each row, `|AI − %|` should be within tolerance. Save the output as the
calibration record.

### Step 7 — Persist and verify

7.1 Save the routing and PID configuration to PLC flash:

```
python calibrate.py save
```

7.2 Power-cycle the PLC. Reconnect and check that routing survived:

```
python calibrate.py status
```

7.3 (Optional but recommended) Re-run one point of Step 2 and Step 4 after
the power cycle to confirm the calibration is stable across reboot.

7.4 Record the calibration data in your engineering log:

- Date and operator
- Pot positions on all four signal conditioners (mark them on the
  conditioner enclosure if possible)
- DMM model and serial used
- Sweep output from Step 6
- Any out-of-spec points and what was done

## Returning to normal operation

Calibration leaves PID 0 and PID 1 configured as pass-through drivers. Before
running real process control:

```
python calibrate.py teardown
```

This unmaps PID 0 and PID 1 (sets pv/cv tags to `kUnmappedTag = 255` and zeros
their gains), leaving the output routing in place so other components (real
PIDs, the FastAPI backend) can drive the AOs.

## Troubleshooting

| Symptom                                         | Likely cause                                                                                   |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `ao` runs but DMM shows 0 mA                    | Loop power supply off, or AO+/AO- swapped, or the conditioner's loop input isn't powered       |
| AO stuck at 4 mA (0%) regardless of `--percent` | Output routing not set — re-run `setup`. Or interlock tripped — check `status`                 |
| V at conditioner output never changes           | Wrong pot being turned, or input mA is not actually changing — verify with DMM in series first |
| AI reads 0% with known good 4-20 mA input       | Input routing not set — re-run `setup`. Or wrong AI channel wired                              |
| Tag value oscillates                            | A PID besides 0/1 has pv or cv pointing at the same tag — check with `status`                  |
| `setup` fails to write a register               | Modbus connection dropped — run `ping 192.168.1.100`, then retry                               |

## Appendix — Raw Modbus reference

If `calibrate.py` is unavailable, the same operations via pymodbus:

```python
from pymodbus.client import ModbusTcpClient
import struct

c = ModbusTcpClient("192.168.1.100", port=502)
c.connect()

def f2regs(v):
    lo, hi = struct.unpack("<HH", struct.pack("<f", v))
    return [lo, hi]

# Output routing: AO0 <- TAG_10, AO1 <- TAG_11
c.write_register(address=110, value=10)
c.write_register(address=111, value=11)

# Input routing: AI0 -> TAG_12 (slot 4), AI1 -> TAG_13 (slot 5)
c.write_register(address=104, value=12)
c.write_register(address=105, value=13)

# PID 0 pass-through: pv=30, cv=10, sp=0, kp=1, ki=kd=0
c.write_register(address=200, value=30)              # pv_tag
c.write_register(address=201, value=10)              # cv_tag
c.write_registers(address=202, values=f2regs(0.0))   # setpoint
c.write_registers(address=204, values=f2regs(1.0))   # kp
c.write_registers(address=206, values=f2regs(0.0))   # ki
c.write_registers(address=208, values=f2regs(0.0))   # kd

# To drive AO 0 to X%: rewrite PID 0 setpoint
c.write_registers(address=202, values=f2regs(50.0))

# To read AI 0 percent: read TAG_12 value registers (12*2, 12*2+1)
rr = c.read_holding_registers(address=24, count=2)
ai0_percent = struct.unpack("<f", struct.pack("<HH", *rr.registers))[0]

# Save to flash
c.write_coil(0, True)
```
