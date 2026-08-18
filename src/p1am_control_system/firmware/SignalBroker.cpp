#include "SignalBroker.h"
#include <cmath>

namespace {

bool IsValidTagId(int tag_id) {
  return tag_id >= 0 && tag_id < SignalBroker::kNumTags;
}

bool IsValidRoutingTagId(int tag_id) {
  return tag_id == SignalBroker::kUnmappedTag || IsValidTagId(tag_id);
}

float ClampTagValue(float value) {
  if (!std::isfinite(value) || value < 0.0f) {
    return 0.0f;
  }
  if (value > 100.0f) {
    return 100.0f;
  }
  return value;
}

}  // namespace

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
  if (!IsValidTagId(tag_id)) {
    return 0.0f;
  }

  return ClampTagValue(tags_[tag_id]);
}

void SignalBroker::SetTag(int tag_id, float value) {
  if (!IsValidTagId(tag_id)) {
    return;
  }

  tags_[tag_id] = ClampTagValue(value);
}

void SignalBroker::SetInputRouting(int channel, int tag_id) {
  if (channel < 0 || channel >= kNumInputs) {
    return;
  }
  if (!IsValidRoutingTagId(tag_id)) {
    tag_id = kUnmappedTag;
  }

  input_routing_[channel] = tag_id;
}

int SignalBroker::GetInputRouting(int channel) const {
  if (channel < 0 || channel >= kNumInputs) {
    return kUnmappedTag;
  }
  return input_routing_[channel];
}

void SignalBroker::SetOutputRouting(int channel, int tag_id) {
  if (channel < 0 || channel >= kNumOutputs) {
    return;
  }
  if (!IsValidRoutingTagId(tag_id)) {
    tag_id = kUnmappedTag;
  }

  output_routing_[channel] = tag_id;
}

int SignalBroker::GetOutputRouting(int channel) const {
  if (channel < 0 || channel >= kNumOutputs) {
    return kUnmappedTag;
  }
  return output_routing_[channel];
}

void SignalBroker::ReadHardwareInputs(HardwareInterface& hw) {
  // Read and scale 4 Thermocouples (channels 0 to 3).
  // Full scale is declared once in SignalBroker.h; see the contract note there.
  for (int i = 0; i < 4; ++i) {
    int target_tag = input_routing_[i];
    if (target_tag != kUnmappedTag) {
      float temp = hw.ReadThermocouple(i);
      float scaled = temp * (100.0f / kThermocoupleFullScaleC);  // C -> %
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
