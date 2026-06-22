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
  void WriteHeaterRelay(bool on) override;

  // Raw, unscaled diagnostic read of one analog input (channels 0-3) as the
  // 0-5 V the terminal sees (counts -> 0-20 mA -> 0-5 V across the input
  // burden). No process scaling — purely for the signal-diagnostics plot so
  // operators can troubleshoot the monitor card independent of calibration.
  float ReadAnalogInputRawVolts(int channel);

 private:
  // Actual bench-verified slot order via P1.printModules() on 2026-06-01:
  //   Slot 1 = P1-4ADL2DAL-1   (analog combo, 4 AI + 2 AO, 4-20 mA)
  //   Slot 2 = P1-04THM        (4-channel thermocouple)
  static const int kSlotAna = 1;
  static const int kSlotThm = 2;
  // Heater relay control GPIO (Arduino-header digital pin D2). Driven
  // active-HIGH: HIGH (3.3 V) = relay energized = heater ON; boots LOW. The
  // temperature controller commands this via Modbus coil 2, and the safety
  // interlock forces it LOW on any trip. NOTE: this is a 3.3 V logic output
  // (~7 mA) — drive a logic-level relay board / SSR, or a small
  // transistor/opto driver for a 24 V relay coil; it cannot switch 24 V itself.
  // Reserved pins to avoid: D5 (Ethernet W5500 CS), D6 (inhibit), A3/A4/33
  // (P1AM base controller).
  static const int kPinHeaterRelay = 2;
  // Inhibit GPIO. MUST NOT be pin 5 — the P1AM-ETH shield hardwires the W5500
  // chip-select to D5, so driving D5 from this firmware breaks Ethernet SPI.
  // D6 is free on the P1AM-100 / P1AM-ETH stack.
  static const int kPinInhibit = 6;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
