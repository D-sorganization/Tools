/** Strict comparison-aware persistence for the ground playback workspace. */

import { canonicalGroundJson } from "./flightGroundContract";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import type { FlightToGroundResult } from "./flightGroundTypes";
import { GroundPlaybackTimeline } from "./groundPlayback";
import {
  GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  exactGroundWorkspaceObject,
  groundWorkspaceFromJson,
  parseGroundPlaybackState,
  parseGroundPlaybackViewState,
  type GroundPlaybackState,
  type GroundPlaybackViewState,
} from "./groundPlaybackWorkspace";
import { parseUniqueJson } from "./strictJson";

export const GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2 =
  "rate-of-closure-ground-playback-workspace/v2" as const;
export const GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2 = 11 * 1024 * 1024;
export const GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT = 100_000;
export const GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED = 200_000;

export interface GroundPlaybackComparisonState {
  readonly result: FlightToGroundResult;
  readonly visible: boolean;
}

export interface GroundPlaybackWorkspaceV2 {
  readonly schemaVersion: typeof GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2;
  readonly result: FlightToGroundResult;
  readonly comparison: GroundPlaybackComparisonState | null;
  readonly playback: GroundPlaybackState;
  readonly view: GroundPlaybackViewState;
}

export interface GroundPlaybackWorkspaceLimits {
  readonly maxBytes?: number;
  readonly maxPointsPerResult?: number;
  readonly maxCombinedPoints?: number;
}

export interface GroundPlaybackWorkspaceLoad {
  readonly workspace: GroundPlaybackWorkspaceV2;
  readonly sourceSchemaVersion:
    | typeof GROUND_PLAYBACK_WORKSPACE_SCHEMA
    | typeof GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2;
  readonly migratedFromV1: boolean;
}

interface ResolvedLimits {
  readonly maxBytes: number;
  readonly maxPointsPerResult: number;
  readonly maxCombinedPoints: number;
}

const positiveInteger = (value: number, name: string): number => {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
};

const boundedInteger = (
  value: number,
  name: string,
  hardCap: number,
): number => {
  const normalized = positiveInteger(value, name);
  if (normalized > hardCap) {
    throw new RangeError(`${name} cannot exceed the ${hardCap} hard cap`);
  }
  return normalized;
};

const resolveLimits = (
  limits: GroundPlaybackWorkspaceLimits,
): ResolvedLimits => ({
  maxBytes: boundedInteger(
    limits.maxBytes ?? GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
    "maxBytes",
    GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
  ),
  maxPointsPerResult: boundedInteger(
    limits.maxPointsPerResult ??
      GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    "maxPointsPerResult",
    GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
  ),
  maxCombinedPoints: boundedInteger(
    limits.maxCombinedPoints ?? GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
    "maxCombinedPoints",
    GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
  ),
});

const utf8Length = (text: string): number =>
  new TextEncoder().encode(text).byteLength;

const validatePointLimits = (
  result: FlightToGroundResult,
  comparison: GroundPlaybackComparisonState | null,
  limits: ResolvedLimits,
): void => {
  const counts = [result.trajectory.length];
  if (comparison !== null) counts.push(comparison.result.trajectory.length);
  if (counts.some((count) => count > limits.maxPointsPerResult)) {
    throw new RangeError("ground workspace exceeds the per-result point limit");
  }
  if (
    counts.reduce((sum, count) => sum + count, 0) > limits.maxCombinedPoints
  ) {
    throw new RangeError("ground workspace exceeds the combined point limit");
  }
};

const parseComparison = (
  value: unknown,
): GroundPlaybackComparisonState | null => {
  if (value === null) return null;
  const payload = exactGroundWorkspaceObject(
    value,
    ["result", "visible"],
    "comparison",
    "v2",
  );
  if (typeof payload.visible !== "boolean") {
    throw new TypeError("comparison visible must be a boolean");
  }
  const result = parseFlightToGroundResultRecord(payload.result);
  new GroundPlaybackTimeline(result);
  return Object.freeze({ result, visible: payload.visible });
};

const parseV2 = (
  payload: Record<string, unknown>,
): GroundPlaybackWorkspaceV2 => {
  const exact = exactGroundWorkspaceObject(
    payload,
    ["schema_version", "result", "comparison", "playback", "view"],
    "workspace",
    "v2",
  );
  const result = parseFlightToGroundResultRecord(exact.result);
  const primary = new GroundPlaybackTimeline(result);
  const comparison = parseComparison(exact.comparison);
  const comparisonTimeline =
    comparison === null ? null : new GroundPlaybackTimeline(comparison.result);
  const startTimeS = Math.min(
    primary.startTimeS,
    comparisonTimeline?.startTimeS ?? primary.startTimeS,
  );
  const endTimeS = Math.max(
    primary.endTimeS,
    comparisonTimeline?.endTimeS ?? primary.endTimeS,
  );
  return Object.freeze({
    schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
    result,
    comparison,
    playback: parseGroundPlaybackState(
      exact.playback,
      startTimeS,
      endTimeS,
      "union timeline",
    ),
    view: parseGroundPlaybackViewState(exact.view),
  });
};

const migrateV1 = (text: string): GroundPlaybackWorkspaceV2 => {
  const legacy = groundWorkspaceFromJson(text);
  return Object.freeze({
    schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
    result: legacy.result,
    comparison: null,
    playback: legacy.playback,
    view: legacy.view,
  });
};

const workspacePayload = (
  workspace: GroundPlaybackWorkspaceV2,
): Record<string, unknown> => ({
  comparison:
    workspace.comparison === null
      ? null
      : {
          result: workspace.comparison.result,
          visible: workspace.comparison.visible,
        },
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

const decodeWorkspaceDocument = (
  text: string,
  configuredLimits: GroundPlaybackWorkspaceLimits = {},
): readonly [Record<string, unknown>, ResolvedLimits] => {
  if (typeof text !== "string")
    throw new TypeError("ground playback workspace JSON must be text");
  const limits = resolveLimits(configuredLimits);
  if (utf8Length(text) > limits.maxBytes) {
    throw new RangeError(
      "ground playback workspace JSON exceeds the import size limit",
    );
  }
  const decoded = parseUniqueJson(text);
  if (
    decoded === null ||
    Array.isArray(decoded) ||
    typeof decoded !== "object"
  ) {
    throw new TypeError("workspace must be an object");
  }
  return [decoded as Record<string, unknown>, limits] as const;
};

const validateWorkspaceLimits = (
  workspace: GroundPlaybackWorkspaceV2,
  limits: ResolvedLimits,
): GroundPlaybackWorkspaceV2 => {
  validatePointLimits(workspace.result, workspace.comparison, limits);
  return workspace;
};

/** Parse only one strict v2 workspace under comparison-aware bounds. */
export const groundWorkspaceV2FromJson = (
  text: string,
  configuredLimits: GroundPlaybackWorkspaceLimits = {},
): GroundPlaybackWorkspaceV2 => {
  const [payload, limits] = decodeWorkspaceDocument(text, configuredLimits);
  if (payload.schema_version !== GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2) {
    throw new RangeError("expected ground playback workspace schema v2");
  }
  return validateWorkspaceLimits(parseV2(payload), limits);
};

/** Dispatch strict v1/v2 and normalize the result to workspace v2. */
export const loadGroundWorkspaceVersionedJson = (
  text: string,
  configuredLimits: GroundPlaybackWorkspaceLimits = {},
): GroundPlaybackWorkspaceLoad => {
  const [payload, limits] = decodeWorkspaceDocument(text, configuredLimits);
  let workspace: GroundPlaybackWorkspaceV2;
  if (payload.schema_version === GROUND_PLAYBACK_WORKSPACE_SCHEMA) {
    workspace = migrateV1(text);
  } else if (payload.schema_version === GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2) {
    workspace = parseV2(payload);
  } else {
    throw new RangeError(
      "unsupported ground playback workspace schema_version",
    );
  }
  const sourceSchemaVersion =
    payload.schema_version as GroundPlaybackWorkspaceLoad["sourceSchemaVersion"];
  return Object.freeze({
    workspace: validateWorkspaceLimits(workspace, limits),
    sourceSchemaVersion,
    migratedFromV1: sourceSchemaVersion === GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  });
};

/** Dispatch strict v1/v2 and return the normalized workspace-v2 object. */
export const groundWorkspaceFromVersionedJson = (
  text: string,
  configuredLimits: GroundPlaybackWorkspaceLimits = {},
): GroundPlaybackWorkspaceV2 =>
  loadGroundWorkspaceVersionedJson(text, configuredLimits).workspace;

/** Serialize exact v2 and enforce its UTF-8 output bound after canonicalization. */
export const groundWorkspaceV2ToJson = (
  workspace: GroundPlaybackWorkspaceV2,
  configuredLimits: GroundPlaybackWorkspaceLimits = {},
): string => {
  const limits = resolveLimits(configuredLimits);
  const document = canonicalGroundJson(workspacePayload(workspace)) + "\n";
  if (utf8Length(document) > limits.maxBytes) {
    throw new RangeError(
      "ground playback workspace JSON exceeds the output size limit",
    );
  }
  const validated = groundWorkspaceV2FromJson(document, limits);
  return canonicalGroundJson(workspacePayload(validated)) + "\n";
};
