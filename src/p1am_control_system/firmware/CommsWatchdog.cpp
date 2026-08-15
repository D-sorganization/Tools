#include "CommsWatchdog.h"

namespace {
// A zero or negative timeout would disable the watchdog entirely, which is the
// failure this class exists to prevent. Fall back to a conservative default
// rather than accepting a configuration that silently does nothing.
const unsigned long kFallbackTimeoutMs = 2000UL;
}  // namespace

CommsWatchdog::CommsWatchdog(unsigned long timeout_ms)
    : timeout_ms_(timeout_ms > 0UL ? timeout_ms : kFallbackTimeoutMs),
      last_activity_ms_(0UL) {}

void CommsWatchdog::Begin(unsigned long now_ms) {
  last_activity_ms_ = now_ms;
}

void CommsWatchdog::RecordActivity(unsigned long now_ms) {
  last_activity_ms_ = now_ms;
}

bool CommsWatchdog::IsExpired(unsigned long now_ms) const {
  // Unsigned subtraction wraps with the counter, so this stays correct when
  // millis() rolls over past 0xFFFFFFFF.
  const unsigned long elapsed = now_ms - last_activity_ms_;
  return elapsed >= timeout_ms_;
}

unsigned long CommsWatchdog::TimeoutMs() const {
  return timeout_ms_;
}
