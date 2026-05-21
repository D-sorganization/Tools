#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_MOCK_HARDWARE_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_MOCK_HARDWARE_H_

#include <cassert>
#include "HardwareInterface.h"

class MockHardware : public HardwareInterface {
 public:
  MockHardware()
      : inhibit_active_(false),
        begin_called_(false),
        update_called_count_(0) {
    for (int i = 0; i < 4; ++i) {
      thermocouples_[i] = 0.0f;
    }
    for (int i = 0; i < 2; ++i) {
      analog_inputs_[i] = 0.0f;
      analog_outputs_[i] = 0.0f;
    }
  }

  void Begin() override {
    begin_called_ = true;
  }

  void Update() override {
    update_called_count_++;
  }

  float ReadThermocouple(int channel) override {
    // DbC: Preconditions
    assert(channel >= 0 && channel < 4);
    return thermocouples_[channel];
  }

  float ReadAnalogInput(int channel) override {
    // DbC: Preconditions
    assert(channel >= 0 && channel < 2);
    return analog_inputs_[channel];
  }

  void WriteAnalogOutput(int channel, float value) override {
    // DbC: Preconditions
    assert(channel >= 0 && channel < 2);
    assert(value >= 0.0f && value <= 100.0f);
    analog_outputs_[channel] = value;
  }

  void WriteInhibit(bool active) override {
    inhibit_active_ = active;
  }

  // Helpers for testing
  void SetThermocouple(int channel, float val) {
    assert(channel >= 0 && channel < 4);
    thermocouples_[channel] = val;
  }

  void SetAnalogInput(int channel, float val) {
    assert(channel >= 0 && channel < 2);
    assert(val >= 0.0f && val <= 100.0f);
    analog_inputs_[channel] = val;
  }

  float GetAnalogOutput(int channel) const {
    assert(channel >= 0 && channel < 2);
    return analog_outputs_[channel];
  }

  bool GetInhibitActive() const {
    return inhibit_active_;
  }

  bool WasBeginCalled() const {
    return begin_called_;
  }

  int GetUpdateCalledCount() const {
    return update_called_count_;
  }

 private:
  float thermocouples_[4];
  float analog_inputs_[2];
  float analog_outputs_[2];
  bool inhibit_active_;
  bool begin_called_;
  int update_called_count_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_MOCK_HARDWARE_H_
