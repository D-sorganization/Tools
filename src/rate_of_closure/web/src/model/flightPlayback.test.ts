import { describe, expect, it } from "vitest";

import { PlaybackTimeline, frameAtTime, validatePlaybackPoints } from "./flightPlayback";
import type { FlightPoint } from "./flight";

const points: FlightPoint[] = [
  { time: 0, position: [0, 0, 0], velocity: [1, 0, 1] },
  { time: 1, position: [4, 6, 0], velocity: [1, 0, 0] },
  { time: 3, position: [8, 0, 2], velocity: [1, 0, -1] },
];

describe("flight playback timeline", () => {
  it("interpolates by physical timestamps and exposes apex/landing", () => {
    expect(frameAtTime(points, -1).position).toEqual([0, 0, 0]);
    expect(frameAtTime(points, 2)).toMatchObject({
      time: 2,
      position: [6, 3, 1],
      lowerIndex: 1,
      fraction: 0.5,
      isLanding: false,
    });
    const timeline = new PlaybackTimeline(points);
    expect(timeline.apexTime).toBe(1);
    expect(frameAtTime(points, 99).isLanding).toBe(true);
  });

  it("rejects malformed or non-monotonic samples", () => {
    expect(() => validatePlaybackPoints([])).toThrow(/at least one/);
    expect(() =>
      validatePlaybackPoints([{ ...points[0], time: Number.NaN }]),
    ).toThrow(/finite/);
    expect(() => validatePlaybackPoints([points[0], { ...points[1], time: 0 }]))
      .toThrow(/strictly increasing/);
  });
});
