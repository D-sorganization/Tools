#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_

#include "SignalBroker.h"
#include "HardwareInterface.h"

class SafetyInterlock {
 public:
  // "Disabled" limit sentinels. A tag whose low limit is at (or below)
  // kDisabledLowLimit AND whose high limit is at (or above) kDisabledHighLimit
  // is not interlocked: it is skipped by Evaluate() entirely, so an unrouted
  // tag sitting at 0.0 % -- or a NaN on a channel nobody trusts -- cannot trip
  // the plant (issue #4001). Reset() initialises every tag to these values.
  //
  // This is the firmware half of a two-sided contract: the backend encodes a
  // limit of ``None`` as exactly these values (hardware.INTERLOCK_DISABLED_*),
  // and tests/p1am_control_system parse this header to keep them equal.
  static constexpr float kDisabledLowLimit = -99999.0f;
  static constexpr float kDisabledHighLimit = 99999.0f;

  // Returned by GetTripTagId() when no trip is latched.
  static const int kNoTripTag = -1;

  SafetyInterlock();

  // Reset interlock status and reset limits to default disabled values.
  void Reset();

  // Evaluate tags against trip limits. If tripped, forces outputs to 0 and inhibits hardware.
  // The trip is a LATCH: once set it stays set until ClearTrip() succeeds,
  // even if the causing tag returns inside its band.
  // Precondition: broker is a valid reference, hw is a valid reference
  void Evaluate(SignalBroker& broker, HardwareInterface& hw);

  // Four-limit accessors: lolo/low/high/hihi.
  // Only low/high trip Evaluate(); lolo/hihi are stored for host visibility
  // and reserved for future alarm-vs-trip semantics.
  float GetLoloLimit(int tag_id) const;
  void SetLoloLimit(int tag_id, float val);

  float GetLowLimit(int tag_id) const;
  void SetLowLimit(int tag_id, float val);

  float GetHighLimit(int tag_id) const;
  void SetHighLimit(int tag_id, float val);

  float GetHihiLimit(int tag_id) const;
  void SetHihiLimit(int tag_id, float val);

  // True when at least one of low/high is narrower than the disabled sentinel.
  // Precondition: none (an invalid tag_id reports false).
  bool IsInterlocked(int tag_id) const;

  // Return the first tag currently violating its trip band, or kNoTripTag.
  // A NaN reading on an interlocked tag counts as a violation: a sensor fault
  // on a channel the operator chose to interlock cannot be proven safe, and
  // letting it compare False against every threshold would silently disarm
  // the interlock (issue #4032). NaN on a non-interlocked tag is ignored.
  int FindTripCause(const SignalBroker& broker) const;

  bool IsTripped() const;

  // Tag that latched the current trip, or kNoTripTag.
  int GetTripTagId() const;

  // Host-requested reset (Modbus coil 1). Latch semantics: the trip clears
  // ONLY when no tag is currently violating its band. If the cause is still
  // present the request is refused, the latch stays set and outputs stay
  // forced safe. Returns true when the interlock is clear after the call.
  // Precondition: broker is a valid reference
  bool ClearTrip(const SignalBroker& broker);

 private:
  float lolo_limits_[SignalBroker::kNumTags];
  float low_limits_[SignalBroker::kNumTags];
  float high_limits_[SignalBroker::kNumTags];
  float hihi_limits_[SignalBroker::kNumTags];
  bool tripped_;
  int trip_tag_id_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_
