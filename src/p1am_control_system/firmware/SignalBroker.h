#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_SIGNAL_BROKER_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_SIGNAL_BROKER_H_

#include "HardwareInterface.h"

class SignalBroker {
 public:
  static const int kNumTags = 32;
  static const int kNumInputs = 8;    // 4 Thermocouples + 4 Analog Inputs
  static const int kNumOutputs = 2;   // 2 Analog Outputs
  static const int kUnmappedTag = 255;

  SignalBroker();

  // Reset the broker to default state (all tags at 0.0, all routing disabled).
  void Reset();

  // Read the value of a tag.
  // Precondition: 0 <= tag_id < kNumTags
  // Postcondition: Returns tag value in the range [0.0, 100.0]
  float GetTag(int tag_id) const;

  // Set the value of a tag.
  // Precondition: 0 <= tag_id < kNumTags
  // Postcondition: The tag value is updated and clamped to [0.0, 100.0]
  void SetTag(int tag_id, float value);

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
