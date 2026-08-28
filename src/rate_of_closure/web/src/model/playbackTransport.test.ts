import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/playback_transport_golden_v1.json";
import { PlaybackTimeline, type TimedSample } from "./flightPlayback";
import {
  DEFAULT_SPEED,
  PLAYBACK_SPEEDS,
  SCRUB_STEPS,
  advancePlayback,
  clampTime,
  scrubValue,
  timeAtScrub,
} from "./playbackTransport";
import type { Vec3 } from "./simulation";

const samples: TimedSample[] = fixture.trajectory.times_s.map(
  (time, index) => ({
    time,
    position: fixture.trajectory.positions_m[index] as Vec3,
  }),
);

describe("playback transport golden parity (#4800 P8)", () => {
  it("pins the shared constants the Python twin exposes", () => {
    expect(fixture.schema).toBe("rate-of-closure-playback-transport/v1");
    expect(SCRUB_STEPS).toBe(fixture.scrub_steps);
    expect([...PLAYBACK_SPEEDS]).toEqual(fixture.speeds);
    expect(DEFAULT_SPEED).toBe(fixture.default_speed);
    expect(PLAYBACK_SPEEDS).toContain(DEFAULT_SPEED);
  });

  it("reproduces every golden sample->frame mapping", () => {
    const timeline = new PlaybackTimeline(samples);
    expect(timeline.duration).toBe(fixture.trajectory.duration_s);
    expect(timeline.apexTime).toBe(fixture.trajectory.apex_time_s);
    for (const goldenFrame of fixture.frames) {
      const frame = timeline.frameAt(goldenFrame.requested_time_s);
      expect(frame.time).toBeCloseTo(goldenFrame.time_s, 12);
      expect(frame.lowerIndex).toBe(goldenFrame.lower_index);
      expect(frame.fraction).toBeCloseTo(goldenFrame.fraction, 12);
      expect(frame.isLanding).toBe(goldenFrame.is_landing);
      frame.position.forEach((component, axis) => {
        expect(component).toBeCloseTo(goldenFrame.position_m[axis], 12);
      });
    }
  });

  it("reproduces every golden adjacent-sample step", () => {
    const timeline = new PlaybackTimeline(samples);
    for (const goldenStep of fixture.steps) {
      expect(
        timeline.stepTime(goldenStep.time_s, goldenStep.direction as -1 | 1),
      ).toBeCloseTo(goldenStep.stepped_time_s, 12);
    }
  });

  it("reproduces the golden scrub quantization in both directions", () => {
    for (const golden of fixture.scrub_values) {
      expect(scrubValue(golden.time_s, golden.duration_s)).toBe(golden.value);
    }
    for (const golden of fixture.scrub_times) {
      expect(timeAtScrub(golden.value, golden.duration_s)).toBeCloseTo(
        golden.time_s,
        12,
      );
    }
  });

  it("reproduces the golden wall-clock advances and finish flags", () => {
    for (const golden of fixture.advances) {
      const step = advancePlayback(
        golden.time_s,
        golden.elapsed_s,
        golden.speed,
        golden.duration_s,
      );
      expect(step.timeS).toBeCloseTo(golden.next_time_s, 12);
      expect(step.finished).toBe(golden.finished);
    }
  });
});

describe("playback transport contract", () => {
  it("normalizes finite times onto the timeline and rejects non-finite input", () => {
    expect(clampTime(-1, 3)).toBe(0);
    expect(clampTime(9, 3)).toBe(3);
    expect(() => clampTime(Number.NaN, 3)).toThrow(/finite/);
    expect(() => clampTime(0, -1)).toThrow(/duration/);
  });

  it("rejects malformed scrub positions and step counts", () => {
    expect(() => scrubValue(1, 3, 0)).toThrow(/positive integer/);
    expect(() => timeAtScrub(-1, 3)).toThrow(/within/);
    expect(() => timeAtScrub(SCRUB_STEPS + 1, 3)).toThrow(/within/);
    expect(() => timeAtScrub(0.5, 3)).toThrow(/within/);
  });

  it("rejects non-physical advance requests", () => {
    expect(() => advancePlayback(0, -0.1, 1, 3)).toThrow(/elapsed/);
    expect(() => advancePlayback(0, 0.1, 0, 3)).toThrow(/speed/);
    expect(() => advancePlayback(0, 0.1, Number.POSITIVE_INFINITY, 3)).toThrow(
      /speed/,
    );
  });
});
