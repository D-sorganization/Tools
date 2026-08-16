#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_COMMS_WATCHDOG_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_COMMS_WATCHDOG_H_

// Dead-man timer on the Modbus link to the SCADA host.
//
// Without this the PLC holds its last commanded state forever: the heater
// relay stays closed and the analog outputs keep driving whatever the host
// last wrote, even after the host has lost power, been killed, or been
// unplugged. The host cannot cover this case because the host is the thing
// that died (issue #3999).
//
// Deliberately free of Arduino headers so it is unit-testable on the host:
// the caller supplies `millis()` rather than the class reading it. That also
// makes the rollover behaviour testable, which matters because millis() wraps
// roughly every 49.7 days and a naive comparison would disarm the watchdog for
// another 49 days at the wrap.
class CommsWatchdog {
 public:
  // Precondition: timeout_ms > 0. Should be several scan periods so ordinary
  // jitter does not trip it, but short enough to be a meaningful backstop.
  explicit CommsWatchdog(unsigned long timeout_ms);

  // Arm the watchdog at boot. Until the host talks to us the timer is running,
  // so a PLC that boots into a dead network safes itself rather than waiting.
  void Begin(unsigned long now_ms);

  // Record that the host was heard from.
  void RecordActivity(unsigned long now_ms);

  // True once `timeout_ms` has elapsed since the last activity.
  // Correct across millis() rollover: the unsigned subtraction wraps with the
  // counter. Elapsed == timeout counts as expired -- fail safe, not fail late.
  bool IsExpired(unsigned long now_ms) const;

  unsigned long TimeoutMs() const;

 private:
  unsigned long timeout_ms_;
  unsigned long last_activity_ms_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_COMMS_WATCHDOG_H_
