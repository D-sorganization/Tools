#include "P1AMHardware.h"
#include <Arduino.h>
#include <P1AM.h>
#include <cmath>

// Note: facts-engineering/P1AM#31 documents an SPI bus-sharing issue between
// the P1AM library and Ethernet. The proven mitigation is to (a) call P1.init
// FIRST (before Ethernet.begin), and (b) apply the SPI bus reset workaround
// in setup() after Ethernet.begin. The probe sketch firmware_probe.ino
// validates this pattern works without per-call SPI transactions.

// Note: facts-engineering/P1AM#31 documents an SPI bus-sharing issue between
// the P1AM library and Ethernet. The proven mitigation is to (a) call P1.init
// FIRST (before Ethernet.begin), and (b) apply the SPI bus reset workaround
// in setup() after Ethernet.begin. The probe sketch firmware_probe.ino
// validates this pattern works without per-call SPI transactions.

P1AMHardware::P1AMHardware() {}

void P1AMHardware::Begin() {
  // P1.init() auto-configures the P1-04THM with the library default, which
  // reads a real, stable thermocouple value — but in Fahrenheit. We deliberately
  // do NOT call P1.configureModule() here: custom config arrays put the channel
  // into a bad state (dead-flat reading) on this module, so we keep the proven
  // default and convert F->C in software (see ReadThermocouple).
  P1.init();
  pinMode(kPinInhibit, OUTPUT);
  digitalWrite(kPinInhibit, LOW);
}

void P1AMHardware::Update() {}

float P1AMHardware::ReadThermocouple(int channel) {
  if (channel < 0 || channel >= 4) {
    return 0.0f;
  }
  // The P1-04THM reports in Fahrenheit under the library default config.
  // Convert to Celsius here so the broker/backend speak one unit (deg C).
  // P1AM library channels are 1-indexed; broker uses 0-indexed.
  const float fahrenheit = P1.readTemperature(kSlotThm, channel + 1);
  return (fahrenheit - 32.0f) * 5.0f / 9.0f;
}

float P1AMHardware::ReadAnalogInput(int channel) {
  if (channel < 0 || channel >= 2) {
    return 0.0f;
  }
  // P1.readAnalog returns raw ADC counts. For the P1-4ADL2DAL-1 the AI is
  // 13-bit over a 0-20 mA span: 0 counts -> 0 mA, 8191 counts -> 20 mA. The
  // power-supply monitor outputs are 0-5 V signals (0 V = zero output, 5 V =
  // full) which drive 0-20 mA through the current input's ~250 ohm burden, so
  // scale the full 0-20 mA span linearly to 0-100 % (0 mA -> 0 %, 20 mA ->
  // 100 %). If a channel is instead a 4-20 mA / 1-5 V signal (reads ~1 V / 4 mA
  // at zero output), use (mA - 4.0f) * (100.0f / 16.0f) instead.
  // Library channels are 1-indexed; broker uses 0-indexed.
  uint32_t counts = P1.readAnalog(kSlotAna, channel + 1);
  float mA = static_cast<float>(counts) * (20.0f / 8191.0f);
  float percent = mA * (100.0f / 20.0f);
  if (percent < 0.0f) {
    percent = 0.0f;
  } else if (percent > 100.0f) {
    percent = 100.0f;
  }
  return percent;
}

void P1AMHardware::WriteAnalogOutput(int channel, float value) {
  if (channel < 0 || channel >= 2) {
    return;
  }
  if (!std::isfinite(value) || value < 0.0f) {
    value = 0.0f;
  } else if (value > 100.0f) {
    value = 100.0f;
  }
  // P1.writeAnalog takes raw DAC counts (uint32_t). For the P1-4ADL2DAL-1
  // the AO is 12-bit over the 4-20 mA span: 0 counts -> 4 mA,
  // 4095 counts -> 20 mA. Scale the 0-100 % broker value to counts.
  // Library channels are 1-indexed; broker uses 0-indexed.
  uint32_t counts = static_cast<uint32_t>(value * (4095.0f / 100.0f));
  P1.writeAnalog(counts, kSlotAna, channel + 1);
}

void P1AMHardware::WriteInhibit(bool active) {
  digitalWrite(kPinInhibit, active ? HIGH : LOW);
}

void P1AMHardware::WriteHeaterRelay(bool on) {
  // Safe no-op until a discrete-output module slot is configured (see header).
  // This keeps the bench-verified build unchanged when no DO module is present.
  if (kSlotRelay < 0) {
    return;
  }
  // P1.writeDiscrete drives one channel of a discrete-output module.
  // Library channels are 1-indexed (kChanRelay).
  P1.writeDiscrete(on ? HIGH : LOW, kSlotRelay, kChanRelay);
}
