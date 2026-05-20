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
  static const int kPinInhibit = 5;  // GPIO Inhibit Pin 5
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_P1AM_HARDWARE_H_
