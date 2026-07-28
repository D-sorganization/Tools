#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_HARDWARE_INTERFACE_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_HARDWARE_INTERFACE_H_

class HardwareInterface {
 public:
  virtual ~HardwareInterface() = default;

  // Initialize hardware modules.
  virtual void Begin() = 0;

  // Perform any periodic background reading or synchronization.
  virtual void Update() = 0;

  // Read temperature from thermocouple module.
  // Precondition: 0 <= channel < 4
  // Postcondition: Returns temperature in Celsius.
  virtual float ReadThermocouple(int channel) = 0;

  // Read analog input from the analog-combo module.
  // Precondition: 0 <= channel < 4 (AI0/AI1 = 0-20 mA PSU monitors;
  //   AI2/AI3 = 4-20 mA signal-conditioned thermocouples).
  // Postcondition: Returns value scaled to 0.0 - 100.0% over the channel's span.
  virtual float ReadAnalogInput(int channel) = 0;

  // Write analog output to DAC module.
  // Precondition: 0 <= channel < 2, 0.0 <= value <= 100.0
  virtual void WriteAnalogOutput(int channel, float value) = 0;

  // Drive the safety Inhibit GPIO.
  // Precondition: none
  virtual void WriteInhibit(bool active) = 0;

  // Drive the heater relay discrete output (24 V DO -> relay -> 110 V heater).
  // Precondition: none. Postcondition: relay energized iff `on`. Implemented as
  // a safe no-op until a discrete-output module slot is configured.
  virtual void WriteHeaterRelay(bool on) = 0;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_HARDWARE_INTERFACE_H_
