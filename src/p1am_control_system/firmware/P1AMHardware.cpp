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
  P1.init();
  // Override the P1-04THM's power-up default (type-J, Fahrenheit) with type-K
  // in degrees Celsius — otherwise a type-K probe reads in F (e.g. ~28 C indoor
  // air shows as ~83 F). Config bytes per the FACTS module reference:
  //   0x4003           enable channels 1-4
  //   0x6001           degrees C + low-side burnout (default 0x6005 = degrees F)
  //   0x21 11 .. 0x24 11  per-channel range; type nibble 1 = type-K
  // Verify after flashing with P1.readModuleConfig(); room air should read ~25 C.
  static const char kThmTypeKCelsius[20] = {
      0x40, 0x03, 0x60, 0x01, 0x21, 0x11, 0x22, 0x11, 0x23, 0x11,
      0x24, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  P1.configureModule(kThmTypeKCelsius, kSlotThm);
  pinMode(kPinInhibit, OUTPUT);
  digitalWrite(kPinInhibit, LOW);
}

void P1AMHardware::Update() {}

float P1AMHardware::ReadThermocouple(int channel) {
  if (channel < 0 || channel >= 4) {
    return 0.0f;
  }
  // P1AM library channels are 1-indexed; broker uses 0-indexed.
  return P1.readTemperature(kSlotThm, channel + 1);
}

float P1AMHardware::ReadAnalogInput(int channel) {
  if (channel < 0 || channel >= 2) {
    return 0.0f;
  }
  // P1.readAnalog returns raw ADC counts. For the P1-4ADL2DAL-1 the AI is
  // 13-bit over a 0-20 mA span: 0 counts -> 0 mA, 8191 counts -> 20 mA.
  // Convert to percent of the 4-20 mA process span: 4 mA -> 0 %, 20 mA -> 100 %.
  // Library channels are 1-indexed; broker uses 0-indexed.
  uint32_t counts = P1.readAnalog(kSlotAna, channel + 1);
  float mA = static_cast<float>(counts) * (20.0f / 8191.0f);
  float percent = (mA - 4.0f) * (100.0f / 16.0f);
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
