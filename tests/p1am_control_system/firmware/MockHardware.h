#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_MOCK_HARDWARE_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_MOCK_HARDWARE_H_

#include <cassert>
#include "HardwareInterface.h"

// Host-side fake for HardwareInterface.
//
// Beyond recording the latest value written to each output, this fake counts
// writes and remembers the highest value an output was ever commanded to.
// Safety tests need to distinguish "reads 0 now" from "was never energized",
// which a bare last-value fake cannot express.
class MockHardware : public HardwareInterface {
 public:
  MockHardware()
      : inhibit_active_(false),
        heater_relay_on_(false),
        heater_relay_write_count_(0),
        begin_called_(false),
        update_called_count_(0) {
    for (int i = 0; i < 4; ++i) {
      thermocouples_[i] = 0.0f;
    }
    for (int i = 0; i < 2; ++i) {
      analog_inputs_[i] = 0.0f;
      analog_outputs_[i] = 0.0f;
      analog_output_max_seen_[i] = 0.0f;
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
    if (value > analog_output_max_seen_[channel]) {
      analog_output_max_seen_[channel] = value;
    }
  }

  void WriteInhibit(bool active) override {
    inhibit_active_ = active;
  }

  void WriteHeaterRelay(bool on) override {
    heater_relay_on_ = on;
    heater_relay_write_count_++;
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

  // Highest value this channel was ever commanded to. Lets a test assert an
  // output was never energized, not merely that it is de-energized now.
  float GetAnalogOutputMaxSeen(int channel) const {
    assert(channel >= 0 && channel < 2);
    return analog_output_max_seen_[channel];
  }

  bool GetInhibitActive() const {
    return inhibit_active_;
  }

  bool GetHeaterRelayOn() const {
    return heater_relay_on_;
  }

  int GetHeaterRelayWriteCount() const {
    return heater_relay_write_count_;
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
  float analog_output_max_seen_[2];
  bool inhibit_active_;
  bool heater_relay_on_;
  int heater_relay_write_count_;
  bool begin_called_;
  int update_called_count_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_MOCK_HARDWARE_H_
