#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_SIGNAL_BROKER_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_SIGNAL_BROKER_H_

#include "HardwareInterface.h"

class SignalBroker {
 public:
  static const int kNumTags = 32;
  static const int kNumInputs = 6;    // 4 Thermocouples + 2 Analog Inputs
  static const int kNumOutputs = 2;   // 2 Analog Outputs
  static const int kUnmappedTag = 255;

  // Degrees C represented by a 100.0% thermocouple tag value. Type-K range is
  // ~0-1372 C; 0-1400 C spans the heater controller's full working range.
  //
  // This is the firmware half of a two-sided contract: the backend's
  // temp_full_scale_c must agree or every reported temperature is wrong by the
  // ratio (issue #3998). Exposed here so tests and any future read-back
  // register can reference one definition instead of copying the literal.
  static constexpr float kThermocoupleFullScaleC = 1400.0f;

  SignalBroker();

  // Reset the broker to default state (all tags at 0.0, all routing disabled).
  void Reset();

  // Read the value of a tag.
  // Precondition: 0 <= tag_id < kNumTags
  // Postcondition: Returns the stored value unchanged -- a percent-of-span
  // float that may sit below 0.0 or above 100.0 near over-range (issue
  // #4032) -- or NaN when the tag holds a bad-quality (non-finite) reading.
  // See SetTag.
  float GetTag(int tag_id) const;

  // Set the value of a tag.
  // Precondition: 0 <= tag_id < kNumTags
  // Postcondition: A finite value is stored exactly as given -- the broker
  // does NOT clamp (issue #4032). Tags are percent-of-span floats:
  // thermocouples are degC scaled by kThermocoupleFullScaleC in
  // ReadHardwareInputs and may read below 0 % or above 100 % near over-range;
  // AI/AO are 0-100 % by the module. The former global [0, 100] clamp
  // truncated genuine over-range readings and silently disabled any
  // interlock limit above the clamp ceiling, so it is gone; the physical AO
  // saturates at the DAC span in WriteHardwareOutputs instead, and the
  // actionable limit domain [0, 100] percent is enforced by the host at the
  // API boundary (hardware.INTERLOCK_LIMIT_MIN / _MAX). A non-finite value
  // is stored as NaN -- the broker's "bad quality" marker -- and is NEVER
  // coerced to 0.0: a sensor fault is not a measurement, and 0.0 % sits
  // below every low limit and looks like a valid cold reading to the host
  // (hardware.py `_require_finite_number`).
  void SetTag(int tag_id, float value);

  // True when the tag holds a finite reading, false for bad quality / invalid id.
  bool IsTagValid(int tag_id) const;

  // Set input routing: map hardware input channel to tag ID.
  // Precondition: 0 <= channel < kNumInputs
  // Precondition: tag_id is in [0, kNumTags-1] OR is kUnmappedTag
  void SetInputRouting(int channel, int tag_id);

  // Get input routing: get tag ID mapped to hardware input channel.
  // Precondition: 0 <= channel < kNumInputs
  // Postcondition: Returns tag ID or kUnmappedTag
  int GetInputRouting(int channel) const;

  // Set output routing: map hardware output channel to source tag ID.
  // Precondition: 0 <= channel < kNumOutputs
  // Precondition: tag_id is in [0, kNumTags-1] OR is kUnmappedTag
  void SetOutputRouting(int channel, int tag_id);

  // Get output routing: get tag ID driving hardware output channel.
  // Precondition: 0 <= channel < kNumOutputs
  // Postcondition: Returns tag ID or kUnmappedTag
  int GetOutputRouting(int channel) const;

  // Read hardware inputs, scale them, and write them to their mapped tags.
  // Precondition: hw is a valid reference to HardwareInterface
  void ReadHardwareInputs(HardwareInterface& hw);

  // Read mapped tag values and write them to hardware outputs.
  // Precondition: hw is a valid reference to HardwareInterface
  void WriteHardwareOutputs(HardwareInterface& hw);

 private:
  float tags_[kNumTags];
  int input_routing_[kNumInputs];    // Maps input index -> tag ID
  int output_routing_[kNumOutputs];  // Maps output index -> tag ID
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_SIGNAL_BROKER_H_
