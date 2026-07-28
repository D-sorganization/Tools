#include "P1AMHardware.h"
#include <Arduino.h>
#include <P1AM.h>
#include <cmath>

// Note: facts-engineering/P1AM#31 documents an SPI bus-sharing issue between
// the P1AM library and Ethernet. The proven mitigation is to (a) call P1.init
// FIRST (before Ethernet.begin), and (b) apply the SPI bus reset workaround
// in setup() after Ethernet.begin. The probe sketch firmware_probe.ino
// validates this pattern works without per-call SPI transactions.

namespace {

const int kThmConfigBytes = 20;

// P1-04THM input-type codes — the low byte of each 0x2n channel config word.
// 0x01 = type K, 0x03 = type R (FACTS/AutomationDirect P1-04THM input-type
// table). VERIFY any new type against the datasheet, and confirm the channel
// against a known-temperature source before relying on it.
const char kTcTypeK = 0x01;
const char kTcTypeR = 0x03;

// P1-04THM: enable all 4 channels, low-side burnout, Celsius. Module channel 1
// is type K and channel 2 is type R, so the operator can drive the heater from
// either thermocouple via the HMI toggle. Channels 3-4 stay type K.
// Module channels are 1-indexed (1-4); the broker maps channel N to TAG_(N-1),
// so channel 1 (type K) reads back on TAG_0 and channel 2 (type R) on TAG_1.
// Layout from FACTS P1-04THM docs:
//   0x4003 = ch1-4 enabled
//   0x6001 = low-side burnout, degrees C
//   0x2n[type] = channel n input type (n = 1..4)
const char kP104ThmConfig[kThmConfigBytes] = {
    0x40, 0x03, 0x60, 0x01,
    0x21, kTcTypeK, 0x22, kTcTypeR, 0x23, kTcTypeK, 0x24, kTcTypeK,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

void PrintHexByte(char value) {
  uint8_t byte = static_cast<uint8_t>(value);
  if (byte < 0x10) {
    Serial.print('0');
  }
  Serial.print(byte, HEX);
}

void PrintThmConfig(const char config[]) {
  for (int i = 0; i < kThmConfigBytes; ++i) {
    if (i > 0) {
      Serial.print(' ');
    }
    PrintHexByte(config[i]);
  }
  Serial.println();
}

bool ConfigMatches(const char lhs[], const char rhs[]) {
  for (int i = 0; i < kThmConfigBytes; ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

P1AMHardware::P1AMHardware() {}

void P1AMHardware::Begin() {
  // P1.init() auto-configures the P1-04THM with the library default
  // (type-J/Fahrenheit). Override it before Ethernet init so thermocouple
  // channels report type-K values directly in degrees C. Boot default is
  // low-side burnout, which matches Modbus coil 3's power-on value of 0, so the
  // first control-loop scan sees no change and won't reconfigure the module
  // (see ConfigureThm / kThmBurnoutCoil).
  P1.init();
  ConfigureThm(false);

  pinMode(kPinInhibit, OUTPUT);
  digitalWrite(kPinInhibit, LOW);
  // Heater relay now drives the P1-08TD2 discrete-output module (see
  // WriteHeaterRelay). Force it OFF at boot so the heater is never energized
  // before the controller commands it.
  P1.writeDiscrete(LOW, kSlotHeaterDO, kChanHeaterDO);
}

bool P1AMHardware::ConfigureThm(bool highSideBurnout) {
  // Build the P1-04THM config identical to kP104ThmConfig except byte[3], the
  // burnout+units low byte, which selects the open-thermocouple fail direction:
  //   0x01 = low-side burnout  (open reads 0 C / cold)
  //   0x03 = high-side burnout (open reads full-scale / hot)
  // Channel types and the Celsius unit selection are unchanged. The P1-04THM
  // cannot disable burnout — only flip its direction.
  char thm_config[kThmConfigBytes];
  for (int i = 0; i < kThmConfigBytes; ++i) {
    thm_config[i] = kP104ThmConfig[i];
  }
  thm_config[3] = highSideBurnout ? 0x03 : 0x01;

  Serial.print(F("[hw] configuring P1-04THM: ch1=K ch2=R ch3-4=K, Celsius, burnout="));
  Serial.println(highSideBurnout ? F("high-side") : F("low-side"));
  bool thm_configured = P1.configureModule(thm_config, kSlotThm);
  Serial.print(F("[hw] P1-04THM configureModule="));
  Serial.println(thm_configured ? F("ok") : F("failed"));

  char thm_readback[kThmConfigBytes] = {};
  P1.readModuleConfig(thm_readback, kSlotThm);
  Serial.print(F("[hw] P1-04THM config readback: "));
  PrintThmConfig(thm_readback);
  if (ConfigMatches(thm_config, thm_readback)) {
    Serial.print(F("[hw] P1-04THM config verified: ch1=K ch2=R, Celsius, burnout="));
    Serial.println(highSideBurnout ? F("high-side") : F("low-side"));
  } else {
    Serial.println(F("[hw] WARNING: P1-04THM config readback mismatch"));
  }

  thm_high_side_ = highSideBurnout;
  return thm_configured;
}

void P1AMHardware::Update() {}

float P1AMHardware::ReadThermocouple(int channel) {
  if (channel < 0 || channel >= 4) {
    return 0.0f;
  }
  // The P1-04THM is configured in Begin() for Celsius. P1AM library channels
  // are 1-indexed; broker uses 0-indexed.
  return P1.readTemperature(kSlotThm, channel + 1);
}

float P1AMHardware::ReadAnalogInput(int channel) {
  if (channel < 0 || channel >= 4) {
    return 0.0f;
  }
  // P1.readAnalog returns raw ADC counts. For the P1-4ADL2DAL-1 the AI is
  // 13-bit over a 0-20 mA span: 0 counts -> 0 mA, 8191 counts -> 20 mA.
  // Library channels are 1-indexed; broker uses 0-indexed.
  uint32_t counts = P1.readAnalog(kSlotAna, channel + 1);
  float mA = static_cast<float>(counts) * (20.0f / 8191.0f);

  // Per-channel span -> 0-100 %:
  //   AI0/AI1 (channels 0,1): the DC-power-supply 0-5 V monitor outputs (0 V =
  //     zero output, 5 V = full) drive 0-20 mA through the ~250 ohm burden, so
  //     the whole 0-20 mA span maps linearly (0 mA -> 0 %, 20 mA -> 100 %).
  //   AI2/AI3 (channels 2,3): the thermocouple signal conditioners emit a
  //     4-20 mA loop (live zero), so map (mA - 4) over the 16 mA span
  //     (4 mA -> 0 %, 20 mA -> 100 %). A broken loop / unpowered conditioner
  //     reads < 4 mA and clamps to 0 % ("cold") — the backend deglitch filter +
  //     cross-check + HH-on-either then keep that fail-safe.
  float percent;
  if (channel >= 2) {
    percent = (mA - 4.0f) * (100.0f / 16.0f);
  } else {
    percent = mA * (100.0f / 20.0f);
  }
  if (percent < 0.0f) {
    percent = 0.0f;
  } else if (percent > 100.0f) {
    percent = 100.0f;
  }
  return percent;
}

float P1AMHardware::ReadAnalogInputRawVolts(int channel) {
  if (channel < 0 || channel >= 4) {
    return 0.0f;
  }
  // counts 0-8191 = 0-20 mA; the 0-5 V terminal voltage = mA * 250 ohm / 1000.
  // So volts = counts * 5 / 8191 (0 mA -> 0 V, 20 mA -> 5 V). No clamping or
  // process scaling: this is the raw signal for troubleshooting.
  // Library channels are 1-indexed; broker uses 0-indexed.
  uint32_t counts = P1.readAnalog(kSlotAna, channel + 1);
  return static_cast<float>(counts) * (5.0f / 8191.0f);
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
  // Drive channel 1 of the P1-08TD2 (slot 3): ON sources 24 V to the relay
  // coil, OFF drives ~0 V. If no DO module is in the slot, P1.writeDiscrete is
  // a harmless no-op.
  P1.writeDiscrete(on ? HIGH : LOW, kSlotHeaterDO, kChanHeaterDO);
}
