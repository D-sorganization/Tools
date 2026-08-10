import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_reference_pipeline_golden_v1.json";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import { GroundPlaybackTimeline } from "./groundPlayback";
import {
  GroundPlaybackComparison,
  groundComparisonCsv,
  groundComparisonJson,
} from "./groundPlaybackComparison";

describe("GroundPlaybackTimeline", () => {
  const timeline = () =>
    new GroundPlaybackTimeline(parseFlightToGroundResultRecord(fixture.result));

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
    const trajectory = (
      payload.trajectory as Array<Record<string, unknown>>
    ).slice(0, -1);
    const events = (payload.events as Array<Record<string, unknown>>).slice(
      0,
      -1,
    );
    const final = trajectory[trajectory.length - 1];
    const position = final.position_m as number[];
    const summary = payload.summary as Record<string, unknown>;
    payload.status = "partial";
    payload.trajectory = trajectory;
    payload.events = events;
    summary.final_downrange_m = position[0];
    summary.final_offline_m = position[2];
    summary.total_distance_m = position[0];
    payload.termination = {
      completed: false,
      reason: "time_limit",
      time_s: final.time_s,
    };

    const value = new GroundPlaybackTimeline(
      parseFlightToGroundResultRecord(payload),
    );
    expect(value.isComplete).toBe(false);
    expect(value.endLabel).toBe("Observed end");
  });
});

describe("GroundPlaybackComparison", () => {
  const timelines = () => {
    const primaryRecord = structuredClone(fixture.result) as Record<
      string,
      unknown
    >;
    const comparisonRecord = structuredClone(fixture.result) as Record<
      string,
      unknown
    >;
    comparisonRecord.request_id = "comparison-run";
    (comparisonRecord.provenance as Record<string, unknown>).input_sha256 =
      "b".repeat(64);
    for (const point of comparisonRecord.trajectory as Array<
      Record<string, unknown>
    >) {
      point.time_s = Number(point.time_s) + 0.2;
    }
    for (const event of comparisonRecord.events as Array<
      Record<string, unknown>
    >) {
      event.time_s = Number(event.time_s) + 0.2;
    }
    const termination = comparisonRecord.termination as Record<string, unknown>;
    termination.time_s = Number(termination.time_s) + 0.2;
    return {
      primary: new GroundPlaybackTimeline(
        parseFlightToGroundResultRecord(primaryRecord),
      ),
      comparison: new GroundPlaybackTimeline(
        parseFlightToGroundResultRecord(comparisonRecord),
      ),
    };
  };

  it("synchronizes on absolute time and labels held observations", () => {
    const { primary, comparison } = timelines();
    const session = new GroundPlaybackComparison(primary, comparison);
    expect(session.startTimeS).toBeCloseTo(primary.startTimeS);
    expect(session.endTimeS).toBeCloseTo(comparison.endTimeS);
    expect(session.frameAt(primary.startTimeS).comparisonState).toBe(
      "waiting for first contact",
    );
    const later = session.frameAt(primary.endTimeS + 0.1);
    expect(later.primaryState).toBe("held at rest");
    expect(later.comparisonState).toBe("active");
  });

  it("exports every scalar row and provenance deterministically", () => {
    const { primary, comparison } = timelines();
    const session = new GroundPlaybackComparison(primary, comparison);
    expect(session.metricRows).toHaveLength(14);
    expect(
      session.metricRows.find(({ metricId }) => metricId === "start_time_s")
        ?.delta,
    ).toBe(0.2);
    expect(
      session.metricRows.find(({ metricId }) => metricId === "end_time_s")
        ?.delta,
    ).toBe(0.2);
    expect(
      session.metricRows.find(({ metricId }) => metricId === "duration_s")
        ?.delta,
    ).toBe(0);
    expect(session.provenanceRows[0]).toEqual({
      field: "Request ID",
      primary: "surface-run-analytic",
      comparison: "comparison-run",
    });
    expect(session.provenanceRows).toHaveLength(12);
    expect(session.provenanceRows).toContainEqual({
      field: "Calibration kind",
      primary: "literature",
      comparison: "literature",
    });
    expect(session.provenanceRows).toContainEqual({
      field: "Calibration source",
      primary: "documented literature basis",
      comparison: "documented literature basis",
    });
    expect(session.provenanceRows).toContainEqual({
      field: "Calibration confidence",
      primary: "0.6",
      comparison: "0.6",
    });
    const encoded = groundComparisonJson(session);
    expect(encoded).toBe(groundComparisonJson(session));
    expect(JSON.parse(encoded).delta_definition).toBe(
      "comparison_minus_primary",
    );
    expect(groundComparisonCsv(session)).toContain(
      "metric_id,label,unit,primary,comparison,comparison_minus_primary",
    );
    expect(groundComparisonCsv(session)).toContain(
      "duration_s,Observed duration,s,0.49966094435,0.49966094435,0\n",
    );
  });
});
