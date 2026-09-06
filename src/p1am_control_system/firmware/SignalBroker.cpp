#include "SignalBroker.h"
#include <cmath>
#include <limits>

namespace {

bool IsValidTagId(int tag_id) {
  return tag_id >= 0 && tag_id < SignalBroker::kNumTags;
}

bool IsValidRoutingTagId(int tag_id) {
  return tag_id == SignalBroker::kUnmappedTag || IsValidTagId(tag_id);
}

// Percent-of-span passthrough: the broker does NOT clamp tag values (#4032).
// A finite value is stored and returned exactly as written -- thermocouple
// channels read degC scaled by kThermocoupleFullScaleC and may sit below 0 %
// or above 100 % near over-range, and an interlock limit above the old clamp
// ceiling must be able to trip. Only a non-finite value is mapped to NaN, the
// broker's bad-quality marker. See the SetTag contract in SignalBroker.h for
// why NaN is not mapped to 0.
float NormalizeTagValue(float value) {
  if (!std::isfinite(value)) {
    return std::numeric_limits<float>::quiet_NaN();
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

  return tags_[tag_id];
}

void SignalBroker::SetTag(int tag_id, float value) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  tags_[tag_id] = NormalizeTagValue(value);
}

bool SignalBroker::IsTagValid(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return false;
  }
  return std::isfinite(tags_[tag_id]);
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
      // The DAC contract is [0, 100] percent (4-20 mA; see
      // HardwareInterface.h). A tag may read outside that span -- an
      // over-range runaway (#4032) or a sub-zero degC reading -- so the
      // write seam saturates the command at the physical span. A bad-quality
      // source tag must not reach the DAC: drive the safe 0.0 %.
      if (!std::isfinite(val) || val < 0.0f) {
        val = 0.0f;
      } else if (val > 100.0f) {
        val = 100.0f;
      }
      hw.WriteAnalogOutput(i, val);
    } else {
      // If unmapped, write safe 0.0% output
      hw.WriteAnalogOutput(i, 0.0f);
    }
  }
}
