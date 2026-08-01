import { useEffect, useRef } from "react";

/**
 * Run a refresh on mount and then on a fixed interval (#4042a, #4011).
 *
 * Two defects share this shape. The event log had a single writer whose only
 * call site was inside the alarm-acknowledge handler, so the Events tab stayed
 * blank until an unrelated acknowledgement happened and was frozen after that.
 * And the active-alarm list had no refresh path of its own at all — nothing
 * polled `/api/alarms/active` — so any stream-side failure to parse the alarm
 * map was silent and permanent for the whole session.
 *
 * The callback is held in a ref so a caller may pass a freshly-created closure
 * on every render (as `App` does at ~10 Hz) without resetting the timer, which
 * would otherwise mean the poll never actually fires.
 *
 * @param refresh - the work to run; may be async (rejections are the caller's).
 * @param intervalMs - milliseconds between runs; must be > 0.
 * @param enabled - when false, nothing is scheduled. Flipping it to true runs
 *   the refresh immediately.
 * @throws RangeError if `intervalMs` is not a positive finite number.
 */
export function usePollingRefresh(
  refresh: () => void | Promise<unknown>,
  intervalMs: number,
  enabled: boolean = true,
): void {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
    throw new RangeError("intervalMs must be a positive finite number");
  }

  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;

  useEffect(() => {
    if (!enabled) return;
    const run = () => void refreshRef.current();
    run();
    const timer = setInterval(run, intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs, enabled]);
}
