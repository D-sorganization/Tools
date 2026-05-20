#include "P1AMHardware.h"
#include <Arduino.h>
#include <P1AM.h>

P1AMHardware::P1AMHardware() {}

void P1AMHardware::Begin() {
  P1.init();
  pinMode(kPinInhibit, OUTPUT);
  digitalWrite(kPinInhibit, LOW);
}

void P1AMHardware::Update() {
  // Synchronize base controller with module updates.
  // In typical loops, we wait until the base is ready to scan.
  while (!P1.isReady()) {
    delay(1);
  }
}

float P1AMHardware::ReadThermocouple(int channel) {
  // Preconditions: channel must be between 0 and 3
  if (channel < 0 || channel >= 4) {
    return 0.0f;
  }
  return P1.readTemperature(kSlotThm, channel);
}

float P1AMHardware::ReadAnalogInput(int channel) {
  // Preconditions: channel must be between 0 and 1
  if (channel < 0 || channel >= 2) {
    return 0.0f;
  }
  // readAnalog on P1AM-100 returns floating point representations or raw counts
  // depending on scaling. We assume standard percentage scaling of 0.0f to 100.0f.
  return P1.readAnalog(kSlotAna, channel);
}

void P1AMHardware::WriteAnalogOutput(int channel, float value) {
  // Preconditions: channel between 0 and 1, value between 0.0 and 100.0
  if (channel < 0 || channel >= 2) {
    return;
  }
  if (value < 0.0f) {
    value = 0.0f;
  } else if (value > 100.0f) {
    value = 100.0f;
  }
  P1.writeAnalog(value, kSlotAna, channel);
}

void P1AMHardware::WriteInhibit(bool active) {
  digitalWrite(kPinInhibit, active ? HIGH : LOW);
}
