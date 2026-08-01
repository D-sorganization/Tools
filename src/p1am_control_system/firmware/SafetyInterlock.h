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
  //
  // Trips on the HIHI/LOLO tier only. low/high is the SCADA layer's
  // severity-1 *warning* band; tripping the plant on a warning is what made
  // the stock config (low=5.0 on every tag) unrunnable (issue #4001).
  //
  // Only tags that are routed as an input or an output are evaluated. An
  // unrouted tag sits at 0.0 and is not a process measurement, so it must not
  // be able to latch the plant off.
  //
  // Precondition: broker is a valid reference, hw is a valid reference
  void Evaluate(SignalBroker& broker, HardwareInterface& hw);

  // True if `limit` is a value the interlock can actually act on.
  //
  // Broker tags are clamped to [0, 100], so a limit outside that range can
  // never be crossed. Reset() uses +/-99999 as a deliberate "never trip"
  // sentinel; anything else out of range is an unreachable operator entry --
  // typically a limit typed in engineering units (900 meaning 900 degC) on a
  // percent-scaled tag, which silently disables the trip (issue #4032).
  //
  // The host uses this to reject such a configuration at the API boundary
  // rather than accepting a limit that does nothing.
  static bool IsLimitEffective(float limit);

  // Four-limit accessors: lolo/low/high/hihi.
  // lolo/hihi are the trip band evaluated by Evaluate(); low/high are stored
  // for host-side warning annunciation.
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
