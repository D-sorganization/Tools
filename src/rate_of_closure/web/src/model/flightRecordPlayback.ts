/**
 * Runtime-neutral lift of an imported trajectory record onto the shared
 * playback timeline (ADR-0047 H4, UD #9353) — the TypeScript twin of
 * `rate_of_closure/simulation/flight_record_playback.py`.
 *
 * The Impact Explorer's 3D playback already derives its frames from
 * retained samples (#4800 P8) regardless of who produced them; this
 * module is the missing seam that lets it replay a
 * `swing_sim.ball_flight_trajectory/1` record (ADR-0047 H1) — whether
 * produced by this repo's `swing_sim.flight` or by UpstreamDrift's
 * named published models — without re-simulating or resampling a
 * single value.
 *
 * The wire codec itself stays Python-only (see the module docstring of
 * `shared.python.swing_sim.flight_interchange.trajectory`); this twin
 * covers only the one piece of logic a TypeScript consumer of an
 * already-parsed record needs and that P8 pins cross-runtime — the
 * frame conversion — pinned by the `imported_trajectory` block of the
 * shared golden fixture
 * `__fixtures__/playback_transport_golden_v1.json`.
 *
 * Transport semantics live in `playbackTransport.ts` / `flightPlayback.ts`
 * and camera state belongs to #4571; neither is re-implemented here.
 */

import { fromFlightFrame, type Vec3 } from "./impactPhysics";
import type { TimedSample } from "./flightPlayback";

/** UpstreamDrift flight frame: x forward, y left, z up; ground at z = 0. */
export const FLIGHT_FRAME_ID = "flight_xfwd_yleft_zup";

/** AffineDrift app frame: x target, y up, z right; ground at y = 0. */
export const APP_FRAME_ID = "app_xtarget_yup_zright";

/** One retained sample of an imported `ball_flight_trajectory/1` record. */
export interface BallFlightRecordSample {
  readonly timeS: number;
  readonly positionM: Vec3;
}

/**
 * Raised when a record declares a frame this loader cannot place.
 *
 * The wire's `frame_id` is a closed enum, but this loader converts
 * explicitly rather than defaulting, so a future frame added to the
 * wire is refused here — loudly — until this module is taught the new
 * conversion, instead of silently drawing an unconverted (and likely
 * mirrored or rotated) trajectory. Mirrors
 * `UnsupportedTrajectoryFrameError` in the Python twin.
 */
export class UnsupportedTrajectoryFrameError extends Error {
  readonly frameId: string;

  constructor(frameId: string) {
    super(
      `unsupported ball_flight_trajectory frame_id: ${JSON.stringify(frameId)}; ` +
        `this playback loader converts only "${FLIGHT_FRAME_ID}" and "${APP_FRAME_ID}"`,
    );
    this.name = "UnsupportedTrajectoryFrameError";
    this.frameId = frameId;
  }
}

/**
 * Lift one imported `ball_flight_trajectory/1` record's samples onto the
 * shared playback timeline's `TimedSample` shape.
 *
 * @param frameId - The record's declared frame (`FLIGHT_FRAME_ID` or
 *   `APP_FRAME_ID`); anything else is refused.
 * @param samples - The record's retained samples, replayed exactly —
 *   never re-simulated or resampled.
 * @returns App-frame `TimedSample`s ready for `PlaybackTimeline`.
 * @throws {UnsupportedTrajectoryFrameError} If `frameId` is not one of
 *   the two frames this loader converts.
 */
export function timedSamplesFromBallFlightRecord(
  frameId: string,
  samples: readonly BallFlightRecordSample[],
): TimedSample[] {
  if (frameId !== APP_FRAME_ID && frameId !== FLIGHT_FRAME_ID) {
    throw new UnsupportedTrajectoryFrameError(frameId);
  }
  const convert =
    frameId === FLIGHT_FRAME_ID ? fromFlightFrame : (v: Vec3): Vec3 => v;
  return samples.map((sample) => ({
    time: sample.timeS,
    position: convert(sample.positionM),
  }));
}
