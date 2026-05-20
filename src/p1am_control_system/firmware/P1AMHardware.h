#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_

#include "HardwareInterface.h"

class P1AMHardware : public HardwareInterface {
 public:
  P1AMHardware();
  ~P1AMHardware() override = default;

  void Begin() override;
  void Update() override;
  float ReadThermocouple(int channel) override;
  float ReadAnalogInput(int channel) override;
  void WriteAnalogOutput(int channel, float value) override;
  void WriteInhibit(bool active) override;

 private:
  static const int kSlotThm = 1;    // P1-04THM is in slot 1
  static const int kSlotAna = 2;    // P1-4ADL2DAL-1 is in slot 2
  // Inhibit GPIO. MUST NOT be pin 5 — the P1AM-ETH shield hardwires the W5500
  // chip-select to D5, so driving D5 from this firmware breaks Ethernet SPI.
  // D6 is free on the P1AM-100 / P1AM-ETH stack.
  static const int kPinInhibit = 6;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
