/** Strict portable persistence and deterministic exports for ground playback. */

import { canonicalGroundJson } from "./flightGroundContract";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import type { FlightToGroundResult, GroundVec3 } from "./flightGroundTypes";
import { GroundPlaybackTimeline } from "./groundPlayback";
import { parseUniqueJson } from "./strictJson";

export const GROUND_PLAYBACK_WORKSPACE_SCHEMA =
  "rate-of-closure-ground-playback-workspace/v1" as const;
export const SUPPORTED_PLAYBACK_SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

export interface GroundPlaybackState {
  readonly timeS: number;
  readonly speed: number;
  readonly loop: boolean;
}

export interface GroundPlaybackViewState {
  readonly yawDeg: number;
  readonly pitchDeg: number;
  readonly zoom: number;
}

export interface GroundPlaybackWorkspace {
  readonly schemaVersion: typeof GROUND_PLAYBACK_WORKSPACE_SCHEMA;
  readonly result: FlightToGroundResult;
  readonly playback: GroundPlaybackState;
  readonly view: GroundPlaybackViewState;
}

/** Normalize an interactive orbit yaw into the persisted closed-open interval. */
export const normalizeGroundPlaybackYawDegrees = (value: number): number =>
  ((value + 180) % 360 + 360) % 360 - 180;

const exactObject = (value: unknown, fields: readonly string[], name: string): Record<string, unknown> => {
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw new TypeError(`${name} must be an object`);
  }
  const payload = value as Record<string, unknown>;
  const actual = Object.keys(payload).sort();
  const expected = [...fields].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new RangeError(`${name} fields do not match v1 schema`);
  }
  return payload;
};

const finiteNumber = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return value;
};

const parsePlayback = (value: unknown, timeline: GroundPlaybackTimeline): GroundPlaybackState => {
  const payload = exactObject(value, ["time_s", "speed", "loop"], "playback");
  const timeS = finiteNumber(payload.time_s, "playback time_s");
  const speed = finiteNumber(payload.speed, "playback speed");
  if (!SUPPORTED_PLAYBACK_SPEEDS.some((candidate) => candidate === speed)) {
    throw new RangeError("speed must be a supported playback speed");
  }
  if (timeS < timeline.startTimeS || timeS > timeline.endTimeS) {
    throw new RangeError("playback time_s must lie within the result timeline");
  }
  if (typeof payload.loop !== "boolean") throw new TypeError("playback loop must be a boolean");
  return Object.freeze({ timeS, speed, loop: payload.loop });
};

const parseView = (value: unknown): GroundPlaybackViewState => {
  const payload = exactObject(value, ["yaw_deg", "pitch_deg", "zoom"], "view");
  const yawDeg = finiteNumber(payload.yaw_deg, "view yaw_deg");
  const pitchDeg = finiteNumber(payload.pitch_deg, "view pitch_deg");
  const zoom = finiteNumber(payload.zoom, "view zoom");
  if (yawDeg < -180 || yawDeg > 180) throw new RangeError("yaw_deg must lie within [-180, 180]");
  if (pitchDeg < -90 || pitchDeg > 90) throw new RangeError("pitch_deg must lie within [-90, 90]");
  if (zoom < 0.4 || zoom > 4) throw new RangeError("zoom must lie within [0.4, 4.0]");
  return Object.freeze({ yawDeg, pitchDeg, zoom });
};

const workspacePayload = (workspace: GroundPlaybackWorkspace): Record<string, unknown> => ({
  playback: {
    loop: workspace.playback.loop,
    speed: workspace.playback.speed,
    time_s: workspace.playback.timeS,
  },
  result: workspace.result,
  schema_version: workspace.schemaVersion,
  view: {
    pitch_deg: workspace.view.pitchDeg,
    yaw_deg: workspace.view.yawDeg,
    zoom: workspace.view.zoom,
  },
});

export const groundWorkspaceToJson = (workspace: GroundPlaybackWorkspace): string => {
  const validated = groundWorkspaceFromJson(canonicalGroundJson(workspacePayload(workspace)));
  return canonicalGroundJson(workspacePayload(validated));
};

export const groundWorkspaceFromJson = (text: string): GroundPlaybackWorkspace => {
  const payload = exactObject(
    parseUniqueJson(text), ["schema_version", "result", "playback", "view"], "workspace",
  );
  if (payload.schema_version !== GROUND_PLAYBACK_WORKSPACE_SCHEMA) {
    throw new RangeError("unsupported ground playback workspace schema_version");
  }
  const result = parseFlightToGroundResultRecord(payload.result);
  const timeline = new GroundPlaybackTimeline(result);
  return Object.freeze({
    schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA,
    result,
    playback: parsePlayback(payload.playback, timeline),
    view: parseView(payload.view),
  });
};

export const groundResultJson = (result: FlightToGroundResult): string =>
  canonicalGroundJson(parseFlightToGroundResultRecord(result));

const numeric = (value: number): string => canonicalGroundJson(value);
const csvCell = (value: string): string => /[",\n\r]/.test(value)
  ? `"${value.replace(/"/g, '""')}"` : value;
const csv = (headers: readonly string[], rows: readonly (readonly string[])[]): string =>
  [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
const vector = (value: GroundVec3): string[] => value.map(numeric);

export const groundTrajectoryCsv = (result: FlightToGroundResult): string => csv([
  "sample_index", "time_s", "phase", "frame",
  "position_x_m", "position_y_m", "position_z_m",
  "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s",
  "angular_velocity_x_rad_s", "angular_velocity_y_rad_s", "angular_velocity_z_rad_s",
], parseFlightToGroundResultRecord(result).trajectory.map((point, index) => [
  String(index), numeric(point.time_s), point.phase, point.frame,
  ...vector(point.position_m), ...vector(point.velocity_m_s),
  ...vector(point.angular_velocity_rad_s),
]));

export const groundEventCsv = (result: FlightToGroundResult): string => csv([
  "sequence", "event_type", "time_s", "frame",
  "position_x_m", "position_y_m", "position_z_m",
  "velocity_before_x_m_s", "velocity_before_y_m_s", "velocity_before_z_m_s",
  "velocity_after_x_m_s", "velocity_after_y_m_s", "velocity_after_z_m_s",
  "angular_velocity_before_x_rad_s", "angular_velocity_before_y_rad_s",
  "angular_velocity_before_z_rad_s", "angular_velocity_after_x_rad_s",
  "angular_velocity_after_y_rad_s", "angular_velocity_after_z_rad_s",
], parseFlightToGroundResultRecord(result).events.map((event) => [
  String(event.sequence), event.event_type, numeric(event.time_s), event.frame,
  ...vector(event.position_m), ...vector(event.velocity_before_m_s),
  ...vector(event.velocity_after_m_s), ...vector(event.angular_velocity_before_rad_s),
  ...vector(event.angular_velocity_after_rad_s),
]));
