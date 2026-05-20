#include "SignalBroker.h"
#include <cassert>

SignalBroker::SignalBroker() {
  Reset();
}

void SignalBroker::Reset() {
  for (int i = 0; i < kNumTags; ++i) {
    tags_[i] = 0.0f;
  }
  for (int i = 0; i < kNumInputs; ++i) {
    input_routing_[i] = kUnmappedTag;
  }
  for (int i = 0; i < kNumOutputs; ++i) {
    output_routing_[i] = kUnmappedTag;
  }
}

float SignalBroker::GetTag(int tag_id) const {
  // DbC Precondition: tag_id must be within valid range
  assert(tag_id >= 0 && tag_id < kNumTags);

  float val = tags_[tag_id];

  // DbC Postcondition: returned value must be within [0.0, 100.0]
  assert(val >= 0.0f && val <= 100.0f);
  return val;
}

void SignalBroker::SetTag(int tag_id, float value) {
  // DbC Precondition: tag_id must be within valid range
  assert(tag_id >= 0 && tag_id < kNumTags);

  // Clamp value to standardized range [0.0, 100.0]
  if (value < 0.0f) {
    value = 0.0f;
  } else if (value > 100.0f) {
    value = 100.0f;
  }

  tags_[tag_id] = value;

  // DbC Postcondition: stored value is within [0.0, 100.0]
  assert(tags_[tag_id] >= 0.0f && tags_[tag_id] <= 100.0f);
}

void SignalBroker::SetInputRouting(int channel, int tag_id) {
  // DbC Precondition: channel and tag_id must be within valid range
  assert(channel >= 0 && channel < kNumInputs);
  assert(tag_id == kUnmappedTag || (tag_id >= 0 && tag_id < kNumTags));

  input_routing_[channel] = tag_id;
}

int SignalBroker::GetInputRouting(int channel) const {
  // DbC Precondition: channel must be within valid range
  assert(channel >= 0 && channel < kNumInputs);
  return input_routing_[channel];
}

void SignalBroker::SetOutputRouting(int channel, int tag_id) {
  // DbC Precondition: channel and tag_id must be within valid range
  assert(channel >= 0 && channel < kNumOutputs);
  assert(tag_id == kUnmappedTag || (tag_id >= 0 && tag_id < kNumTags));

  output_routing_[channel] = tag_id;
}

int SignalBroker::GetOutputRouting(int channel) const {
  // DbC Precondition: channel must be within valid range
  assert(channel >= 0 && channel < kNumOutputs);
  return output_routing_[channel];
}

void SignalBroker::ReadHardwareInputs(HardwareInterface& hw) {
  // Read and scale 4 Thermocouples (channels 0 to 3)
  // We assume thermocouple inputs range from 0.0 to 1000.0 Celsius.
  // We scale 0-1000 C -> 0.0% - 100.0% Tag values.
  for (int i = 0; i < 4; ++i) {
    int target_tag = input_routing_[i];
    if (target_tag != kUnmappedTag) {
      float temp = hw.ReadThermocouple(i);
      float scaled = temp / 10.0f;  // 0 - 1000 C maps to 0 - 100 %
      SetTag(target_tag, scaled);
    }
  }

  // Read and scale 2 Analog Inputs (channels 0 to 1)
  // Analog inputs are already assumed to be in the 0.0% - 100.0% scale.
  for (int i = 0; i < 2; ++i) {
    int target_tag = input_routing_[4 + i];
    if (target_tag != kUnmappedTag) {
      float analog_val = hw.ReadAnalogInput(i);
      SetTag(target_tag, analog_val);
    }
  }
}

void SignalBroker::WriteHardwareOutputs(HardwareInterface& hw) {
  // Write 2 Analog Outputs
  for (int i = 0; i < kNumOutputs; ++i) {
    int source_tag = output_routing_[i];
    if (source_tag != kUnmappedTag) {
      float val = GetTag(source_tag);
      hw.WriteAnalogOutput(i, val);
    } else {
      // If unmapped, write safe 0.0% output
      hw.WriteAnalogOutput(i, 0.0f);
    }
  }
}
