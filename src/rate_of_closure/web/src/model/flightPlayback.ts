/** Deterministic interpolation over solver-owned ball-flight timestamps. */

import type { FlightPoint } from "./flight";
import type { Vec3 } from "./simulation";

export interface PlaybackFrame {
  time: number;
  position: Vec3;
  lowerIndex: number;
  fraction: number;
  isImpact: boolean;
}

function finiteVector(vector: Vec3): boolean {
  return vector.every(Number.isFinite);
}

/** Validate the immutable playback boundary before UI animation begins. */
export function validatePlaybackPoints(points: readonly FlightPoint[]): void {
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

  constructor(points: readonly FlightPoint[]) {
    validatePlaybackPoints(points);
    this.points = points.map((point) => ({
      time: point.time,
      position: [...point.position],
    }));
    this.duration = this.points[this.points.length - 1].time;
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
      isImpact: false,
    };
  }

  private endpointFrame(index: number, time: number): PlaybackFrame {
    return {
      time,
      position: [...this.points[index].position],
      lowerIndex: index,
      fraction: 0,
      isImpact: index === this.points.length - 1,
    };
  }
}

/** One-shot interpolation convenience for non-animated consumers. */
export function frameAtTime(points: readonly FlightPoint[], time: number): PlaybackFrame {
  return new PlaybackTimeline(points).frameAt(time);
}
