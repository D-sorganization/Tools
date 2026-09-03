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
  trip_tag_id_ = kNoTripTag;
  // Initialize limits to the disabled sentinels so nothing trips by default.
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    lolo_limits_[i] = kDisabledLowLimit;
    low_limits_[i] = kDisabledLowLimit;
    high_limits_[i] = kDisabledHighLimit;
    hihi_limits_[i] = kDisabledHighLimit;
  }
}

bool SafetyInterlock::IsInterlocked(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return false;
  }
  return low_limits_[tag_id] > kDisabledLowLimit ||
         high_limits_[tag_id] < kDisabledHighLimit;
}

int SafetyInterlock::FindTripCause(const SignalBroker& broker) const {
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    if (!IsInterlocked(i)) {
      // Unrouted / un-interlocked tags read 0.0 (or NaN) and are not process
      // measurements; they must never be able to trip the plant (#4001).
      continue;
    }
    float val = broker.GetTag(i);
    if (!std::isfinite(val)) {
      // Sensor fault on an interlocked channel: fail safe (see header).
      return i;
    }
    if (val > high_limits_[i] || val < low_limits_[i]) {
      return i;
    }
  }
  return kNoTripTag;
}

void SafetyInterlock::Evaluate(SignalBroker& broker, HardwareInterface& hw) {
  const int cause = FindTripCause(broker);
  if (cause != kNoTripTag) {
    tripped_ = true;
    trip_tag_id_ = cause;
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
    return kDisabledLowLimit;
  }
  return lolo_limits_[tag_id];
}

void SafetyInterlock::SetLoloLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = kDisabledLowLimit;
  }
  lolo_limits_[tag_id] = val;
}

float SafetyInterlock::GetLowLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return kDisabledLowLimit;
  }
  return low_limits_[tag_id];
}

void SafetyInterlock::SetLowLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = kDisabledLowLimit;
  }
  low_limits_[tag_id] = val;
}

float SafetyInterlock::GetHighLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return kDisabledHighLimit;
  }
  return high_limits_[tag_id];
}

void SafetyInterlock::SetHighLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = kDisabledHighLimit;
  }
  high_limits_[tag_id] = val;
}

float SafetyInterlock::GetHihiLimit(int tag_id) const {
  if (!IsValidTagId(tag_id)) {
    return kDisabledHighLimit;
  }
  return hihi_limits_[tag_id];
}

void SafetyInterlock::SetHihiLimit(int tag_id, float val) {
  if (!IsValidTagId(tag_id)) {
    return;
  }
  if (!std::isfinite(val)) {
    val = kDisabledHighLimit;
  }
  hihi_limits_[tag_id] = val;
}

bool SafetyInterlock::IsTripped() const {
  return tripped_;
}

int SafetyInterlock::GetTripTagId() const {
  return trip_tag_id_;
}

bool SafetyInterlock::ClearTrip(const SignalBroker& broker) {
  if (!tripped_) {
    return true;
  }
  if (FindTripCause(broker) != kNoTripTag) {
    // Cause still present: refuse. The latch and the forced-safe outputs
    // stay exactly as they are; the next Evaluate() re-asserts them.
    return false;
  }
  tripped_ = false;
  trip_tag_id_ = kNoTripTag;
  return true;
}
