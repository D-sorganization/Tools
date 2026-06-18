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

 private:
  // Actual bench-verified slot order via P1.printModules() on 2026-06-01:
  //   Slot 1 = P1-4ADL2DAL-1   (analog combo, 4 AI + 2 AO, 4-20 mA)
  //   Slot 2 = P1-04THM        (4-channel thermocouple)
  static const int kSlotAna = 1;
  static const int kSlotThm = 2;
  // Heater relay discrete-output module. The bench stack currently has NO
  // discrete-output module, so this is -1 (disabled -> WriteHeaterRelay is a
  // no-op). When a P1-08TD*/P1-08TRS (or similar 24 V DO) module is installed,
  // set kSlotRelay to its slot (from P1.printModules()) and kChanRelay to the
  // wired channel (1-indexed), then reflash. The temperature controller drives
  // this via Modbus coil 2.
  static const int kSlotRelay = -1;
  static const int kChanRelay = 1;
  // Inhibit GPIO. MUST NOT be pin 5 — the P1AM-ETH shield hardwires the W5500
  // chip-select to D5, so driving D5 from this firmware breaks Ethernet SPI.
  // D6 is free on the P1AM-100 / P1AM-ETH stack.
  static const int kPinInhibit = 6;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
