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

  // Read analog input from DAC module.
  // Precondition: 0 <= channel < 2
  // Postcondition: Returns value scaled to 0.0 - 100.0%.
  virtual float ReadAnalogInput(int channel) = 0;

  // Write analog output to DAC module.
  // Precondition: 0 <= channel < 2, 0.0 <= value <= 100.0
  virtual void WriteAnalogOutput(int channel, float value) = 0;

  // Drive the safety Inhibit GPIO.
  // Precondition: none
  virtual void WriteInhibit(bool active) = 0;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_HARDWARE_INTERFACE_H_
