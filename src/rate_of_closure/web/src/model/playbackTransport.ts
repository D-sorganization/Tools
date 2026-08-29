/**
 * Runtime-neutral playback transport math for 3D shot playback (#4800 P8).
 *
 * One implementation of the timeline transport semantics — time
 * normalization, scrub-index quantization, and wall-clock advance under a
 * speed multiplier — consumed by every playback surface. The Python twin is
 * `rate_of_closure/simulation/playback_transport.py`; both are pinned by the
 * shared golden fixture `__fixtures__/playback_transport_golden_v1.json`.
 *
 * Trajectory-source independence (the putting seam): nothing here knows
 * about ball flight. It operates on physical seconds only, so the putting
 * vertical (#4800 P6/P7) drives the same functions with putt-result
 * timelines unchanged.
 *
 * Camera seam (#4571): camera state belongs to `cameraCommands.ts` and the
 * canvas viewports; this module deliberately owns only timeline math.
 */

/** Canonical speed multipliers offered by every playback surface. */
export const PLAYBACK_SPEEDS: readonly number[] = [0.25, 0.5, 1, 2, 4];

/** Real-time playback rate; both surfaces default their selector to this. */
export const DEFAULT_SPEED = 1;

/** Scrub-slider quantization shared by the Qt and React timelines. */
export const SCRUB_STEPS = 10_000;

/** Result of advancing playback by one wall-clock interval. */
export interface PlaybackAdvance {
  timeS: number;
  finished: boolean;
}

function requireDuration(durationS: number): number {
  if (!Number.isFinite(durationS) || durationS < 0) {
    throw new Error("duration_s must be finite and >= 0");
  }
  return durationS;
}

function requireSteps(steps: number): number {
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new Error("steps must be a positive integer");
  }
  return steps;
}

/** Normalize a finite requested time onto the `[0, duration]` timeline. */
export function clampTime(timeS: number, durationS: number): number {
  if (!Number.isFinite(timeS)) throw new Error("time_s must be finite");
  const duration = requireDuration(durationS);
  return Math.min(Math.max(timeS, 0), duration);
}

/**
 * Quantize a physical time to its integer scrub-slider position.
 *
 * Half-up rounding (`floor(x + 0.5)`) so both runtime twins agree at exact
 * half-steps; an empty timeline always maps to position zero.
 */
export function scrubValue(
  timeS: number,
  durationS: number,
  steps: number = SCRUB_STEPS,
): number {
  const quantum = requireSteps(steps);
  const duration = requireDuration(durationS);
  const time = clampTime(timeS, duration);
  if (duration <= 0) return 0;
  return Math.floor(quantum * (time / duration) + 0.5);
}

/** Physical time for an integer scrub-slider position in `[0, steps]`. */
export function timeAtScrub(
  value: number,
  durationS: number,
  steps: number = SCRUB_STEPS,
): number {
  const quantum = requireSteps(steps);
  const duration = requireDuration(durationS);
  if (!Number.isInteger(value) || value < 0 || value > quantum) {
    throw new Error("value must lie within [0, steps]");
  }
  return duration * (value / quantum);
}

/**
 * Advance playback by an elapsed wall-clock interval at a speed multiplier.
 *
 * Physical timestamps are never altered — the multiplier scales only the
 * wall-clock interval. The result clamps at the timeline end and reports
 * `finished` so callers stop their animation loops identically.
 */
export function advancePlayback(
  timeS: number,
  elapsedS: number,
  speed: number,
  durationS: number,
): PlaybackAdvance {
  if (!Number.isFinite(elapsedS) || elapsedS < 0) {
    throw new Error("elapsed_s must be finite and >= 0");
  }
  if (!Number.isFinite(speed) || speed <= 0) {
    throw new Error("speed must be finite and > 0");
  }
  const duration = requireDuration(durationS);
  const time = clampTime(timeS, duration);
  const advanced = Math.min(duration, time + elapsedS * speed);
  return { timeS: advanced, finished: advanced >= duration };
}
