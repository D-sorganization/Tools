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

  // Getters/Setters for limits
  float GetHighLimit(int tag_id) const;
  void SetHighLimit(int tag_id, float val);

  float GetLowLimit(int tag_id) const;
  void SetLowLimit(int tag_id, float val);

  bool IsTripped() const;
  void ClearTrip();

 private:
  float high_limits_[SignalBroker::kNumTags];
  float low_limits_[SignalBroker::kNumTags];
  bool tripped_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_SAFETY_INTERLOCK_H_
