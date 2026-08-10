import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_reference_pipeline_golden_v1.json";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import { GroundPlaybackTimeline } from "./groundPlayback";

describe("GroundPlaybackTimeline", () => {
  const timeline = () => new GroundPlaybackTimeline(
    parseFlightToGroundResultRecord(fixture.result),
  );

  it("uses absolute result time and exposes phase landmarks", () => {
    const value = timeline();
    expect(value.startTimeS).toBeCloseTo(1.005);
    expect(value.durationS).toBeCloseTo(0.49966094435);
    expect(value.phaseTime("skid")).toBeCloseTo(1.00907886485);
    expect(value.endLabel).toBe("Rest");
  });

  it("holds the lower sample across a phase boundary", () => {
    const value = timeline();
    const before = value.frameAt(1.13);
    const at = value.frameAt(1.14047658257);
    expect(before.phase).toBe("skid");
    expect(before.positionM).toEqual([0.08819335004, 0.02135, 0]);
    expect(before.interpolationFraction).toBe(0);
    expect(at.phase).toBe("roll");
  });

  it("interpolates within a phase and steps exact samples", () => {
    const value = timeline();
    const midpoint = value.frameAt((1.205 + 1.305) / 2);
    expect(midpoint.phase).toBe("roll");
    expect(midpoint.positionM[0]).toBeCloseTo(
      (0.15677340009 + 0.20574015016) / 2,
    );
    expect(midpoint.interpolationFraction).toBeCloseTo(0.5);
    expect(value.stepTime(1.205, 1)).toBeCloseTo(1.305);
    expect(value.stepTime(1.205, -1)).toBeCloseTo(1.14047658257);
  });

  it("labels a censored partial result as an observed end", () => {
    const payload = structuredClone(fixture.result) as Record<string, unknown>;
    const trajectory = (payload.trajectory as Array<Record<string, unknown>>).slice(0, -1);
    const events = (payload.events as Array<Record<string, unknown>>).slice(0, -1);
    const final = trajectory[trajectory.length - 1];
    const position = final.position_m as number[];
    const summary = payload.summary as Record<string, unknown>;
    payload.status = "partial";
    payload.trajectory = trajectory;
    payload.events = events;
    summary.final_downrange_m = position[0];
    summary.final_offline_m = position[2];
    summary.total_distance_m = position[0];
    payload.termination = { completed: false, reason: "time_limit", time_s: final.time_s };

    const value = new GroundPlaybackTimeline(parseFlightToGroundResultRecord(payload));
    expect(value.isComplete).toBe(false);
    expect(value.endLabel).toBe("Observed end");
  });
});
