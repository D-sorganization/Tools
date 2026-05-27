#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_

#include "SignalBroker.h"
#include "HardwareInterface.h"

class SafetyInterlock {
 public:
  SafetyInterlock();

  // Reset interlock status and reset limits to default disabled values.
  void Reset();

  // Evaluate tags against trip limits. If tripped, forces outputs to 0 and inhibits hardware.
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

  bool IsTripped() const;
  void ClearTrip();

 private:
  float lolo_limits_[SignalBroker::kNumTags];
  float low_limits_[SignalBroker::kNumTags];
  float high_limits_[SignalBroker::kNumTags];
  float hihi_limits_[SignalBroker::kNumTags];
  bool tripped_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_
