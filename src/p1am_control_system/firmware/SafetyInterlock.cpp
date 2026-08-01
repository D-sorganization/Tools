#include "SafetyInterlock.h"
#include <cmath>

namespace {

bool IsValidTagId(int tag_id) {
  return tag_id >= 0 && tag_id < SignalBroker::kNumTags;
}

}  // namespace

SafetyInterlock::SafetyInterlock() {
  Reset();
}

void SafetyInterlock::Reset() {
  tripped_ = false;
  // Initialize limits to broad ranges that won't trip by default
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    lolo_limits_[i] = -99999.0f;
    low_limits_[i] = -99999.0f;
    high_limits_[i] = 99999.0f;
    hihi_limits_[i] = 99999.0f;
  }
}

bool SafetyInterlock::IsLimitEffective(float limit) {
  if (!std::isfinite(limit)) {
    return false;
  }
  // Matches the broker's tag domain; see SignalBroker::SetTag.
  return limit >= 0.0f && limit <= 100.0f;
}

void SafetyInterlock::Evaluate(SignalBroker& broker, HardwareInterface& hw) {
  bool trip_detected = false;

  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    // An unrouted tag holds 0.0 and is not a measurement of anything. The
    // stock config writes a lolo floor to all 32 tags, so evaluating unrouted
    // tags latched the plant off the moment that config was deployed.
    if (!broker.IsTagRouted(i)) {
      continue;
    }
    float val = broker.GetTag(i);
    // Trip on the hihi/lolo tier. A limit outside the tag domain is either the
    // "never trip" sentinel or an unreachable entry; skip it either way rather
    // than comparing against a threshold that can never be crossed.
    if (IsLimitEffective(hihi_limits_[i]) && val >= hihi_limits_[i]) {
      trip_detected = true;
      break;
    }
    if (IsLimitEffective(lolo_limits_[i]) && val <= lolo_limits_[i]) {
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

float SafetyInterlock::GetLoloLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return -99999.0f;
  }
  return lolo_limits_[tag_id];
}

void SafetyInterlock::SetLoloLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = -99999.0f;
  }
  lolo_limits_[tag_id] = val;
}

float SafetyInterlock::GetLowLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return -99999.0f;
  }
  return low_limits_[tag_id];
}

void SafetyInterlock::SetLowLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = -99999.0f;
  }
  low_limits_[tag_id] = val;
}

float SafetyInterlock::GetHighLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return 99999.0f;
  }
  return high_limits_[tag_id];
}

void SafetyInterlock::SetHighLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = 99999.0f;
  }
  high_limits_[tag_id] = val;
}

float SafetyInterlock::GetHihiLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return 99999.0f;
  }
  return hihi_limits_[tag_id];
}

void SafetyInterlock::SetHihiLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = 99999.0f;
  }
  hihi_limits_[tag_id] = val;
}

bool SafetyInterlock::IsTripped() const {
  return tripped_;
}

void SafetyInterlock::ClearTrip() {
  tripped_ = false;
}
