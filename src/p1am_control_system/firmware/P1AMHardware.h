#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_

#include "HardwareInterface.h"

// Modbus coil 3 selects the P1-04THM open-thermocouple (burnout) fail
// direction at runtime, mirroring the coil map documented in firmware.ino
// (coil 0 = save-to-flash, coil 1 = E-stop reset, coil 2 = heater relay).
// Coil 3 = 1 -> high-side burnout (open TC reads full-scale/hot, fail-safe for
// a heater); coil 3 = 0 -> low-side burnout (open TC reads 0 C/cold). Applied
// via P1AMHardware::ConfigureThm, and only when the selection changes.
const int kThmBurnoutCoil = 3;

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

  // Reconfigure the P1-04THM open-thermocouple (burnout) fail direction at
  // runtime. highSideBurnout=true sets byte[3]=0x03 (open TC reads full-scale/
  // hot, fail-safe); false sets byte[3]=0x01 (open TC reads 0 C/cold). Every
  // other byte is identical to the boot config (channel types, Celsius).
  // Rebuilds the 20-byte config, calls P1.configureModule for the THM slot,
  // reads back and verifies (same serial logging/verification as Begin()),
  // records the applied direction, and returns the configureModule result.
  // Call only when the direction changes — a live reconfigure briefly
  // glitches temperature reads.
  bool ConfigureThm(bool highSideBurnout);

  // Currently-applied THM burnout direction: true = high-side (0x6003), false
  // = low-side (0x6001). The control loop uses this to reconfigure only when
  // Modbus coil 3 changes.
  bool ThmHighSide() const { return thm_high_side_; }

  // Raw, unscaled diagnostic read of one analog input (channels 0-3) as the
  // 0-5 V the terminal sees (counts -> 0-20 mA -> 0-5 V across the input
  // burden). No process scaling — purely for the signal-diagnostics plot so
  // operators can troubleshoot the monitor card independent of calibration.
  float ReadAnalogInputRawVolts(int channel);

 private:
  // Bench slot order (left-to-right, contiguous) via P1.printModules():
  //   Slot 1 = P1-4ADL2DAL-1   (analog combo, 4 AI + 2 AO, 4-20 mA)
  //   Slot 2 = P1-04THM        (4-channel thermocouple)
  //   Slot 3 = P1-08TD2        (8-pt 12-24 VDC sourcing discrete output)
  static const int kSlotAna = 1;
  static const int kSlotThm = 2;
  // Heater relay drive: channel 1 of the P1-08TD2 discrete-output module in
  // slot 3. This is a real 24 VDC sourcing output (sources the field supply on
  // ON, ~0 V on OFF) — wire it directly to the 24 V relay coil. The temperature
  // controller commands it via Modbus coil 2; the safety interlock forces it
  // OFF on any trip. If the module is moved, update kSlotHeaterDO (1-indexed
  // slot) / kChanHeaterDO (1-indexed channel).
  static const int kSlotHeaterDO = 3;
  static const int kChanHeaterDO = 1;
  // Inhibit GPIO. MUST NOT be pin 5 — the P1AM-ETH shield hardwires the W5500
  // chip-select to D5, so driving D5 from this firmware breaks Ethernet SPI.
  // D6 is free on the P1AM-100 / P1AM-ETH stack.
  static const int kPinInhibit = 6;

  // Applied P1-04THM burnout direction (true = high-side / 0x6003, false =
  // low-side / 0x6001). Set by ConfigureThm; boot default is low-side.
  bool thm_high_side_ = false;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
