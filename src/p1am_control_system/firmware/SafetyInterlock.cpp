#include "SafetyInterlock.h"
#include <cassert>

SafetyInterlock::SafetyInterlock() {
  Reset();
}

void SafetyInterlock::Reset() {
  tripped_ = false;
  // Initialize limits to broad ranges that won't trip by default
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    high_limits_[i] = 99999.0f;  // Effectively disabled
    low_limits_[i] = -99999.0f;  // Effectively disabled
  }
}

void SafetyInterlock::Evaluate(SignalBroker& broker, HardwareInterface& hw) {
  bool trip_detected = false;

  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    float val = broker.GetTag(i);
    if (val > high_limits_[i] || val < low_limits_[i]) {
      trip_detected = true;
      break;
    }
  }

  if (trip_detected) {
    tripped_ = true;
  }

  if (tripped_) {
    // Force output tags to 0.0% in broker
    for (int i = 0; i < SignalBroker::kNumOutputs; ++i) {
      int tag_id = broker.GetOutputRouting(i);
      if (tag_id != SignalBroker::kUnmappedTag) {
        broker.SetTag(tag_id, 0.0f);
      }
    }
    // Force physical hardware outputs to 0.0% immediately
    for (int i = 0; i < SignalBroker::kNumOutputs; ++i) {
      hw.WriteAnalogOutput(i, 0.0f);
    }
    // Drive Inhibit GPIO high
    hw.WriteInhibit(true);
  } else {
    // Normal operation: Write mapped tag values to outputs
    broker.WriteHardwareOutputs(hw);
    hw.WriteInhibit(false);
  }
}

float SafetyInterlock::GetHighLimit(int tag_id) const {
  assert(tag_id >= 0 && tag_id < SignalBroker::kNumTags);
  return high_limits_[tag_id];
}

void SafetyInterlock::SetHighLimit(int tag_id, float val) {
  assert(tag_id >= 0 && tag_id < SignalBroker::kNumTags);
  high_limits_[tag_id] = val;
}

float SafetyInterlock::GetLowLimit(int tag_id) const {
  assert(tag_id >= 0 && tag_id < SignalBroker::kNumTags);
  return low_limits_[tag_id];
}

void SafetyInterlock::SetLowLimit(int tag_id, float val) {
  assert(tag_id >= 0 && tag_id < SignalBroker::kNumTags);
  low_limits_[tag_id] = val;
}

bool SafetyInterlock::IsTripped() const {
  return tripped_;
}

void SafetyInterlock::ClearTrip() {
  tripped_ = false;
}
