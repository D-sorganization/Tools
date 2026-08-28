/**
 * Runtime-neutral lift of a recorded putt onto the shared playback
 * timeline (#4800 P8) — the TypeScript twin of
 * `rate_of_closure/simulation/putt_playback.py`.
 *
 * Playback frames must come from the retained integrator samples, never
 * from re-simulation, so this module does exactly one thing: it reads
 * the `PuttResult` the tab already accepted and the `GreenSurface` that
 * result was integrated on, and returns the structural `TimedSample`
 * timeline `PlaybackTimeline` already knows how to interpolate.
 *
 * The sample->frame mapping is where P8 pins twin parity: both runtimes
 * are pinned by the `putt` block of the shared golden fixture
 * `__fixtures__/playback_transport_golden_v1.json`.
 *
 * Transport semantics live in `playbackTransport.ts` and camera state
 * belongs to #4571; neither is re-implemented here.
 */

import type { TimedSample } from "./flightPlayback";
import {
  GOLF_BALL_RADIUS_M,
  surfaceHeightM,
  type GreenSurface,
  type PuttResult,
} from "./puttingGreen";

/**
 * Lift one integrated putt to the shared playback samples.
 *
 * Positions are green-frame metres: `x` along the target line, `y`
 * lateral (left positive) and `z` the ball-centre elevation read off
 * the same surface the integrator ran on.
 */
export function puttPlaybackSamples(
  result: PuttResult,
  surface: GreenSurface,
): TimedSample[] {
  const count = result.timesS.length;
  if (result.pathXM.length !== count || result.pathYM.length !== count) {
    throw new Error("putt samples must carry one x and one y per time");
  }
  if (count === 0) throw new Error("putt playback requires at least one sample");
  return result.timesS.map((time, index): TimedSample => {
    const xM = result.pathXM[index];
    const yM = result.pathYM[index];
    return {
      time,
      position: [xM, yM, surfaceHeightM(surface, xM, yM) + GOLF_BALL_RADIUS_M],
    };
  });
}
