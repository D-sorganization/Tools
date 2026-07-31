# P1AM-100 Firmware — Build & Flash

Firmware for the Facts Engineering P1AM-100 PLC (SAMD21G18A + W5500 Ethernet
shield + P1-04THM + P1-4ADL2DAL-1 stack). Implements a 10 Hz DCS scan loop
with Modbus TCP front-end, four PID controllers, four-limit safety interlocks
(lolo / low / high / hihi), and EEPROM-backed config persistence.

## Toolchain

```bash
# arduino-cli (single binary)
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
  | BINDIR=~/.local/bin sh

# Facts Engineering board package — provides the correct FQBN
arduino-cli config init
arduino-cli config add board_manager.additional_urls \
  https://raw.githubusercontent.com/facts-engineering/facts-engineering.github.io/master/package_productivity-P1AM-boardmanagermodule_index.json
arduino-cli core update-index
arduino-cli core install P1AM-100:samd

# Libraries
arduino-cli lib install P1AM ArduinoModbus ArduinoRS485 Ethernet FlashStorage
```

## Build

```bash
arduino-cli compile --fqbn P1AM-100:samd:P1AM-100_native \
  src/p1am_control_system/firmware --output-dir /tmp/p1am-build
```

Expected output: roughly 21% flash / 17% RAM (varies as code grows).

## Flash

```bash
arduino-cli upload --fqbn P1AM-100:samd:P1AM-100_native \
  --port /dev/ttyACM0 --input-dir /tmp/p1am-build \
  src/p1am_control_system/firmware
```

Linux users need read/write on `/dev/ttyACM0`. Either log in after adding
yourself to `dialout` (`sudo usermod -aG dialout $USER`) or wrap the upload in
`sg dialout -c '...'` for one-shot use.

Verify success by USB ID:

```bash
lsusb | grep -i facts          # 1354:4000  -> application running (good)
lsusb | grep -i 'arduino zero' # 2341:804d  -> SAM-BA bootloader (something crashed)
```

If you're stuck at `2341:804d`, see the FQBN warning below.

## FQBN warning — read before flashing

**Use `P1AM-100:samd:P1AM-100_native`. Do NOT use `arduino:samd:arduino_zero_native`.**

The Arduino Zero FQBN compiles cleanly and bossac reports `Verify successful`,
but the resulting binary is linked at flash 0x0 — overwriting the SAM-BA
bootloader at 0x0..0x1FFF. After the reset that follows upload, the SAMD jumps
straight into your application before the bootloader's startup code runs, and
the application crashes early enough that:

- Ping may briefly succeed (W5500 retains its IP from a partial init before
  the crash), then stop after a true power cycle.
- Modbus TCP never listens — port 502 returns `Connection refused` or
  times out.
- USB enumerates as `2341:804d Arduino SA Arduino Zero` (the SAM-BA bootloader
  USB descriptor), not `1354:4000 Facts Engineering P1AM-100`.

Recovery: install the P1AM board package above and reflash with the correct
FQBN. The P1AM linker script puts the app at 0x2000, where the bootloader
expects it. Discovered the hard way on 2026-05-27 — two hours of bench
debugging.

## Module slot assignments

These are hard-coded in `P1AMHardware.h` and assumed by `SignalBroker`:

| Slot | Module        | Channels                                         |
| ---- | ------------- | ------------------------------------------------ |
| 1    | P1-4ADL2DAL-1 | 2 analog inputs + 2 analog outputs (all 4-20 mA) |
| 2    | P1-04THM      | 4 thermocouple inputs                            |

The Inhibit GPIO is wired to D6. **D5 is reserved** — the P1AM-ETH shield
hardwires the W5500 chip-select there and driving D5 from the firmware
breaks Ethernet SPI.

## Modbus register map

| Range    | Width | Purpose                                                                     |
| -------- | ----- | --------------------------------------------------------------------------- |
| 0..63    | 64    | Tag values — TAG_i is at regs (i*2, i*2+1) as little-endian IEEE-754 float  |
| 100..105 | 6     | Input routing — channel -> tag id; slots 0-3 = TC0-3, 4-5 = AI0-1           |
| 110..111 | 2     | Output routing — channel -> tag id; slots 0-1 = AO0-1                       |
| 200..239 | 40    | PID config — 4 PIDs x 10 regs = (pv_tag, cv_tag, sp, kp, ki, kd)            |
| 300..555 | 256   | Interlock limits — 32 tags x 8 regs = (lolo, low, high, hihi) IEEE-754      |
| 560      | 1     | Host heartbeat — backend bumps this every scan; feeds the comms watchdog    |
| coil 0   | 1     | Save-to-flash trigger (firmware writes EEPROM on falling edge of write)     |
| coil 1   | 1     | E-stop reset — firmware clears the trip latch and writes the coil back to 0 |
| coil 2   | 1     | Heater relay command (interlock and comms watchdog both override it off)    |
| coil 3   | 1     | Thermocouple burnout direction (see below)                                  |

Tag values are clamped 0.0–100.0 by the broker. AO outputs scale linearly:
0.0% -> 4.000 mA, 100.0% -> 20.000 mA. AI readings are pre-scaled by the P1AM
library before reaching the broker.

### Interlock trip semantics

The firmware trips on the **hihi/lolo** band only. `low`/`high` is the warning
tier and is stored for host-side annunciation. Because broker tags are clamped
to `[0, 100]`, a limit outside that range can never be crossed — the firmware
treats such a limit as absent rather than comparing against an unreachable
threshold, and the host should reject it at the API boundary. Only tags that
are routed as an input or output are evaluated; an unrouted tag sits at 0.0 and
is not a measurement.

### Comms watchdog

If neither a Modbus TCP connection nor a change to the heartbeat register is
seen for 2 s, the firmware drives every analog output to 0 %, opens the heater
relay, asserts the Inhibit GPIO, and holds the PID loops with their integrators
cleared. This is the only protection that survives the host being absent
entirely — a powered-off Pi, a killed backend, or an unplugged cable. Control
resumes automatically when the link returns, starting from a clean integrator.

## Thermocouple module configuration

`P1AMHardware::Begin()` overrides the P1AM library's P1-04THM default after
`P1.init()` and before Ethernet setup. The firmware configures all four
thermocouple channels for type K, low-side burnout, and Celsius output:

```text
40 03 60 01 21 01 22 01 23 01 24 01 00 00 00 00 00 00 00 00
```

Boot diagnostics call `P1.readModuleConfig()` and print the active 20-byte
readback. Expected readback is the same byte sequence above. Temperature reads
are then consumed directly from `P1.readTemperature()` as degrees C; there is no
firmware Fahrenheit-to-Celsius conversion in this mode.

Hardware verification still requires a connected P1-04THM with type-K
thermocouple/reference hardware: confirm the boot readback matches, validate at
least one channel against a known-temperature source, and check that no burnout
or over-range status is asserted with a healthy junction connected.

## EEPROM / FlashStorage

`StorageManager::kMagic = 0xDC52`. Configs written by firmware with a
different magic are silently rejected on load and the unit boots with broker
/ interlock defaults instead of garbling the wider struct. Bump `kMagic` any
time `ConfigStruct` (or its members) change layout.

## Boot diagnostics

`setup()` emits per-step `Serial.println` traces at 115200 baud on `/dev/ttyACM0`.
To capture them across a reset, open the port _before_ triggering the reset:

```bash
# 1. Open the serial monitor in one terminal (and leave it running):
sg dialout -c 'python3 -c "
import serial, time, sys
s = serial.Serial(\"/dev/ttyACM0\", 115200, timeout=0.5)
while True:
    line = s.readline()
    if line: sys.stdout.write(line.decode(errors=\"replace\"))
"'

# 2. In another terminal, trigger a reset by reflashing or via the hardware
# reset button. The monitor will pick up the boot banner.
```

**Do not use a 1200bps touch to trigger reset just to capture serial.** That
puts the SAMD in bootloader mode (the touch is also the upload-mode trigger);
to recover you need to reflash. Use the hardware reset button or unplug USB
for >10s.

## Host driver

The Python driver `backend/modbus_client.py` chunks the 256-reg interlock
block into 4x64-reg reads/writes because pymodbus caps single requests at
125 / 123 registers respectively. If you extend the interlock block, keep the
chunk boundaries tag-aligned (multiples of 8 regs).
