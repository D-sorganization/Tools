import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/playback_transport_golden_v1.json";
import {
  APP_FRAME_ID,
  FLIGHT_FRAME_ID,
  UnsupportedTrajectoryFrameError,
  timedSamplesFromBallFlightRecord,
  type BallFlightRecordSample,
} from "./flightRecordPlayback";
import { PlaybackTimeline, type TimedSample } from "./flightPlayback";
import { puttPlaybackSamples } from "./puttPlayback";
import {
  GOLF_BALL_RADIUS_M,
  type GreenSurface,
  type PuttResult,
} from "./puttingGreen";
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

describe("putt playback golden parity (#4800 P8)", () => {
  const putt = fixture.putt;
  const surface: GreenSurface = {
    kind: "planar",
    gradePercent: putt.surface.grade_percent,
    aspectDeg: putt.surface.aspect_deg,
  };
  const result = {
    pathXM: putt.result.path_x_m,
    pathYM: putt.result.path_y_m,
    speedsMps: putt.result.times_s.map(() => 0),
    timesS: putt.result.times_s,
    skidEndIndex: putt.result.skid_end_index,
    skidDistanceM: 0,
    totalDistanceM: 0,
    timeS: putt.result.times_s[putt.result.times_s.length - 1],
    breakM: putt.result.path_y_m[putt.result.path_y_m.length - 1],
    holed: false,
    speedAtHoleMps: null,
    marginMps: null,
    missDistanceM: 0,
  } satisfies PuttResult;

  it("pins the ball radius the Python twin lifts elevations with", () => {
    expect(GOLF_BALL_RADIUS_M).toBeCloseTo(putt.ball_radius_m, 12);
  });

  it("lifts the retained samples onto the green exactly", () => {
    const samples = puttPlaybackSamples(result, surface);
    expect(samples.map((sample) => sample.time)).toEqual(putt.result.times_s);
    samples.forEach((sample, index) => {
      sample.position.forEach((component, axis) => {
        expect(component).toBeCloseTo(putt.samples_m[index][axis], 12);
      });
    });
  });

  it("reproduces every golden putt frame on the shared timeline", () => {
    const timeline = new PlaybackTimeline(puttPlaybackSamples(result, surface));
    expect(timeline.duration).toBeCloseTo(putt.duration_s, 12);
    for (const goldenFrame of putt.frames) {
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

  it("refuses ragged putt samples", () => {
    expect(() =>
      puttPlaybackSamples({ ...result, pathYM: [0] }, surface),
    ).toThrow(/one x and one y/);
  });
});

describe("imported trajectory record playback golden parity (ADR-0047 H4)", () => {
  const importedTrajectory = fixture.imported_trajectory;
  const recordSamples: BallFlightRecordSample[] =
    importedTrajectory.samples.map((sample) => ({
      timeS: sample.time_s,
      positionM: sample.position_m as Vec3,
    }));

  it("converts flight-frame samples to the golden app-frame positions", () => {
    const samples = timedSamplesFromBallFlightRecord(
      importedTrajectory.frame_id,
      recordSamples,
    );
    expect(samples.map((sample) => sample.time)).toEqual(
      importedTrajectory.samples.map((sample) => sample.time_s),
    );
    samples.forEach((sample, index) => {
      sample.position.forEach((component, axis) => {
        expect(component).toBeCloseTo(
          importedTrajectory.app_positions_m[index][axis],
          12,
        );
      });
    });
  });

  it("reproduces the golden duration and apex time on the shared timeline", () => {
    const samples = timedSamplesFromBallFlightRecord(
      importedTrajectory.frame_id,
      recordSamples,
    );
    const timeline = new PlaybackTimeline(samples);
    expect(timeline.duration).toBeCloseTo(importedTrajectory.duration_s, 12);
    expect(timeline.apexTime).toBeCloseTo(importedTrajectory.apex_time_s, 12);
  });

  it("passes app-frame samples through unconverted", () => {
    const samples = timedSamplesFromBallFlightRecord(
      APP_FRAME_ID,
      recordSamples,
    );
    samples.forEach((sample, index) => {
      expect(sample.position).toEqual(recordSamples[index].positionM);
    });
  });

  it("refuses a frame id this loader does not convert", () => {
    expect(() =>
      timedSamplesFromBallFlightRecord("some_future_frame", recordSamples),
    ).toThrow(UnsupportedTrajectoryFrameError);
    expect(() =>
      timedSamplesFromBallFlightRecord("some_future_frame", recordSamples),
    ).toThrow(/some_future_frame/);
  });

  it("exposes the same frame ids the Python wire declares", () => {
    expect(FLIGHT_FRAME_ID).toBe("flight_xfwd_yleft_zup");
    expect(APP_FRAME_ID).toBe("app_xtarget_yup_zright");
    expect(importedTrajectory.frame_id).toBe(FLIGHT_FRAME_ID);
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
