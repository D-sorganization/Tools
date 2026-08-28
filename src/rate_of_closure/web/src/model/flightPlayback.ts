/**
 * Deterministic interpolation over solver-owned trajectory timestamps
 * (#4200, #4800 P8).
 *
 * Trajectory-source independent (the putting seam): `PlaybackTimeline`
 * accepts any strictly increasing `TimedSample` timeline — recorded ball
 * flight today, putt-result break trajectories later — so the putting
 * vertical consumes this class unchanged. The Python twin is
 * `rate_of_closure/simulation/flight_playback.py` (`TimedTrajectory`); the
 * sample->frame mapping of both twins is pinned by the shared golden
 * fixture `__fixtures__/playback_transport_golden_v1.json`.
 */

import type { Vec3 } from "./simulation";

/** One recorded trajectory sample: physical time [s] + app-frame metres. */
export interface TimedSample {
  time: number;
  position: Vec3;
}

export interface PlaybackFrame {
  time: number;
  position: Vec3;
  lowerIndex: number;
  fraction: number;
  isLanding: boolean;
}

function finiteVector(vector: Vec3): boolean {
  return vector.every(Number.isFinite);
}

/** Validate the immutable playback boundary before UI animation begins. */
export function validatePlaybackPoints(points: readonly TimedSample[]): void {
  if (points.length === 0) throw new Error("playback requires at least one point");
  points.forEach((point, index) => {
    if (!Number.isFinite(point.time) || !finiteVector(point.position)) {
      throw new Error(`playback point ${index} must contain finite time and position`);
    }
    if (point.time < 0) throw new Error("playback timestamps must be non-negative");
    if (index > 0 && point.time <= points[index - 1].time) {
      throw new Error("playback timestamps must be strictly increasing");
    }
  });
}

function interpolate(left: Vec3, right: Vec3, fraction: number): Vec3 {
  return [
    left[0] + (right[0] - left[0]) * fraction,
    left[1] + (right[1] - left[1]) * fraction,
    left[2] + (right[2] - left[2]) * fraction,
  ];
}

/** Validate once, then provide logarithmic-time interpolation for animation. */
export class PlaybackTimeline {
  private readonly points: ReadonlyArray<{ time: number; position: Vec3 }>;
  readonly duration: number;
  readonly apexTime: number;

  constructor(points: readonly TimedSample[]) {
    validatePlaybackPoints(points);
    this.points = points.map((point) => ({
      time: point.time,
      position: [...point.position],
    }));
    this.duration = this.points[this.points.length - 1].time;
    const apex = this.points.reduce((best, point) =>
      point.position[1] > best.position[1] ? point : best,
    );
    this.apexTime = apex.time;
  }

  /** Interpolate app-frame position at finite physical time, clamped to endpoints. */
  frameAt(requestedTime: number): PlaybackFrame {
    if (!Number.isFinite(requestedTime)) throw new Error("playback time must be finite");
    const time = Math.min(Math.max(requestedTime, 0), this.duration);
    if (time <= this.points[0].time || this.points.length === 1) {
      return this.endpointFrame(0, time);
    }
    if (time >= this.duration) return this.endpointFrame(this.points.length - 1, time);
    let lower = 0;
    let upper = this.points.length - 1;
    while (upper - lower > 1) {
      const middle = Math.floor((lower + upper) / 2);
      if (this.points[middle].time <= time) lower = middle;
      else upper = middle;
    }
    const span = this.points[upper].time - this.points[lower].time;
    const fraction = (time - this.points[lower].time) / span;
    return {
      time,
      position: interpolate(this.points[lower].position, this.points[upper].position, fraction),
      lowerIndex: lower,
      fraction,
      isLanding: false,
    };
  }

  /** Return the adjacent solver-owned sample time, clamped to the trajectory. */
  stepTime(requestedTime: number, direction: -1 | 1): number {
    if (!Number.isFinite(requestedTime)) {
      throw new Error("playback time must be finite");
    }
    if (direction !== -1 && direction !== 1) {
      throw new Error("playback step direction must be -1 or 1");
    }
    const frame = this.frameAt(requestedTime);
    if (direction === 1) {
      const nextIndex = Math.min(frame.lowerIndex + 1, this.points.length - 1);
      return this.points[nextIndex].time;
    }
    const previousIndex = frame.fraction > 0
      ? frame.lowerIndex
      : Math.max(frame.lowerIndex - 1, 0);
    return this.points[previousIndex].time;
  }

  private endpointFrame(index: number, time: number): PlaybackFrame {
    return {
      time,
      position: [...this.points[index].position],
      lowerIndex: index,
      fraction: 0,
      isLanding: index === this.points.length - 1,
    };
  }
}

/** One-shot interpolation convenience for non-animated consumers. */
export function frameAtTime(points: readonly TimedSample[], time: number): PlaybackFrame {
  return new PlaybackTimeline(points).frameAt(time);
}
