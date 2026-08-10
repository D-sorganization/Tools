/** Observation-bounded comparison of two strict ground playback timelines. */

import { canonicalGroundJson } from "./flightGroundContract";
import type { FlightToGroundResult, GroundSummary } from "./flightGroundTypes";
import {
  GroundPlaybackTimeline,
  type GroundPlaybackFrame,
} from "./groundPlayback";

export const GROUND_PLAYBACK_COMPARISON_SCHEMA =
  "rate-of-closure-ground-playback-comparison/v1" as const;

export type GroundComparisonState =
  | "waiting for first contact"
  | "active"
  | `held at ${string}`;

export interface GroundComparisonFrame {
  readonly timeS: number;
  readonly primary: GroundPlaybackFrame;
  readonly comparison: GroundPlaybackFrame;
  readonly primaryState: GroundComparisonState;
  readonly comparisonState: GroundComparisonState;
}

export interface GroundComparisonMetric {
  readonly metricId: string;
  readonly label: string;
  readonly unit: string;
  readonly primary: number;
  readonly comparison: number;
  readonly delta: number;
}

export interface GroundComparisonProvenance {
  readonly field: string;
  readonly primary: string;
  readonly comparison: string;
}

const scalarDefinitions = [
  ["carry_distance_m", "Carry distance", "m"],
  ["bounce_air_distance_m", "Bounce air distance", "m"],
  ["skid_distance_m", "Skid distance", "m"],
  ["roll_distance_m", "Roll distance", "m"],
  ["surface_path_distance_m", "Surface path distance", "m"],
  ["total_distance_m", "Total distance", "m"],
  ["final_downrange_m", "Final downrange", "m"],
  ["final_offline_m", "Final offline", "m"],
  ["bounce_count", "Bounce count", "count"],
] as const satisfies ReadonlyArray<readonly [keyof GroundSummary, string, string]>;

const metric = (
  metricId: string,
  label: string,
  unit: string,
  primary: number,
  comparison: number,
): GroundComparisonMetric => Object.freeze({
  metricId, label, unit, primary, comparison, delta: comparison - primary,
});

export class GroundPlaybackComparison {
  readonly primary: GroundPlaybackTimeline;
  readonly comparison: GroundPlaybackTimeline;

  constructor(primary: GroundPlaybackTimeline, comparison: GroundPlaybackTimeline) {
    if (!(primary instanceof GroundPlaybackTimeline)
      || !(comparison instanceof GroundPlaybackTimeline)) {
      throw new TypeError("comparison requires two GroundPlaybackTimeline values");
    }
    if (primary.result.unit_system !== comparison.result.unit_system) {
      throw new RangeError("comparison requires matching unit systems");
    }
    if (primary.result.frame !== comparison.result.frame) {
      throw new RangeError("comparison requires matching coordinate frames");
    }
    this.primary = primary;
    this.comparison = comparison;
  }

  get startTimeS(): number {
    return Math.min(this.primary.startTimeS, this.comparison.startTimeS);
  }

  get endTimeS(): number {
    return Math.max(this.primary.endTimeS, this.comparison.endTimeS);
  }

  get durationS(): number { return this.endTimeS - this.startTimeS; }

  frameAt(timeS: number): GroundComparisonFrame {
    if (!Number.isFinite(timeS)) throw new RangeError("playback time must be finite");
    return Object.freeze({
      timeS: Math.min(Math.max(timeS, this.startTimeS), this.endTimeS),
      primary: this.primary.frameAt(timeS),
      comparison: this.comparison.frameAt(timeS),
      primaryState: this.state(this.primary, timeS),
      comparisonState: this.state(this.comparison, timeS),
    });
  }

  stepTime(currentTimeS: number, direction: -1 | 1): number {
    if (!Number.isFinite(currentTimeS)) throw new RangeError("playback time must be finite");
    const times = [...new Set([
      ...this.primary.result.trajectory.map(({ time_s }) => time_s),
      ...this.comparison.result.trajectory.map(({ time_s }) => time_s),
    ])].sort((left, right) => left - right);
    if (direction === 1) {
      return times.find((time) => time > currentTimeS + 1e-12) ?? this.endTimeS;
    }
    return [...times].reverse().find((time) => time < currentTimeS - 1e-12)
      ?? this.startTimeS;
  }

  private state(timeline: GroundPlaybackTimeline, timeS: number): GroundComparisonState {
    if (timeS < timeline.startTimeS) return "waiting for first contact";
    if (timeS > timeline.endTimeS) return `held at ${timeline.endLabel.toLowerCase()}`;
    return "active";
  }

  get metricRows(): readonly GroundComparisonMetric[] {
    const left = this.primary.result.summary;
    const right = this.comparison.result.summary;
    if (left === null || right === null) throw new Error("playable comparison requires summaries");
    const rows = scalarDefinitions.map(([id, label, unit]) =>
      metric(id, label, unit, left[id], right[id]));
    rows.push(
      metric("start_time_s", "First contact time", "s", this.primary.startTimeS, this.comparison.startTimeS),
      metric("end_time_s", "Observed end time", "s", this.primary.endTimeS, this.comparison.endTimeS),
      metric("duration_s", "Observed duration", "s", this.primary.durationS, this.comparison.durationS),
      metric("event_count", "Event count", "count", this.primary.result.events.length, this.comparison.result.events.length),
      metric("trajectory_sample_count", "Trajectory samples", "count", this.primary.result.trajectory.length, this.comparison.result.trajectory.length),
    );
    return Object.freeze(rows);
  }

  get provenanceRows(): readonly GroundComparisonProvenance[] {
    const left = this.primary.result;
    const right = this.comparison.result;
    const rows: Array<readonly [string, string, string]> = [
      ["Request ID", left.request_id, right.request_id],
      ["Status", left.status, right.status],
      ["Surface ID", left.surface_id, right.surface_id],
      ["Model", `${left.model_id} ${left.model_version}`, `${right.model_id} ${right.model_version}`],
      ["Termination", left.termination.reason, right.termination.reason],
      ["Producer", producer(left), producer(right)],
      ["Source revision", left.provenance.source_revision, right.provenance.source_revision],
      ["Input SHA-256", left.provenance.input_sha256, right.provenance.input_sha256],
      ["Calibration ID", left.calibration.calibration_id, right.calibration.calibration_id],
    ];
    return Object.freeze(rows.map(([field, primary, comparison]) =>
      Object.freeze({ field, primary, comparison })));
  }
}

const producer = (result: FlightToGroundResult): string =>
  `${result.provenance.producer} ${result.provenance.producer_version}`;

export const groundComparisonJson = (comparison: GroundPlaybackComparison): string =>
  canonicalGroundJson({
    schema_version: GROUND_PLAYBACK_COMPARISON_SCHEMA,
    delta_definition: "comparison_minus_primary",
    primary: comparison.primary.result,
    comparison: comparison.comparison.result,
    metrics: comparison.metricRows.map((row) => ({
      metric_id: row.metricId,
      label: row.label,
      unit: row.unit,
      primary: row.primary,
      comparison: row.comparison,
      comparison_minus_primary: row.delta,
    })),
  });

const csvCell = (value: string): string => /[",\n\r]/.test(value)
  ? `"${value.replace(/"/g, '""')}"` : value;

export const groundComparisonCsv = (comparison: GroundPlaybackComparison): string => {
  const rows = [
    ["metric_id", "label", "unit", "primary", "comparison", "comparison_minus_primary"],
    ...comparison.metricRows.map((row) => [
      row.metricId, row.label, row.unit, canonicalGroundJson(row.primary),
      canonicalGroundJson(row.comparison), canonicalGroundJson(row.delta),
    ]),
  ];
  return rows.map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
};
