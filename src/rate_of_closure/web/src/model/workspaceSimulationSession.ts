/** Versioned persistence for the browser's ball support and spatial target. */

import {
  ballSetupFromJson,
  ballSetupToJson,
  defaultBallSetupForClub,
  resolveBallSetup,
  type BallSetup,
} from "./ballSetup";
import type { ClubSpec } from "./club";
import { createSpatialTarget, type SpatialTargetTs } from "./spatialTarget";
import {
  spatialTargetFromJson,
  spatialTargetToJson,
} from "./spatialTargetSerialization";

export const SIMULATION_SETUP_SCHEMA = "rate_of_closure.simulation_setup";
export const SIMULATION_SETUP_SCHEMA_VERSION = 1;
export const BALL_SETUP_SELECTION_SCHEMA = "swing_sim.ball_setup_selection";
export const BALL_SETUP_SELECTION_SCHEMA_VERSION = 1;

const BALL_SETUP_FIELDS = [
  "support_mode",
  "tee_height_m",
  "height_reference",
  "ball_center_m",
] as const;

export interface SimulationWorkspaceSnapshot {
  readonly ballSetup: BallSetup;
  readonly ballSetupUserOverridden: boolean;
  readonly spatialTarget: SpatialTargetTs;
}

function record(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${context} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactRecord(
  value: unknown,
  keys: readonly string[],
  context: string,
): Record<string, unknown> {
  const data = record(value, context);
  const actual = Object.keys(data);
  if (actual.length !== keys.length || keys.some((key) => !(key in data))) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return data;
}

/** Validate the provenance/club relationship without relying on UI state. */
export function validatedSimulationWorkspace(
  snapshot: SimulationWorkspaceSnapshot,
  club: ClubSpec,
): SimulationWorkspaceSnapshot {
  if (typeof snapshot?.ballSetupUserOverridden !== "boolean") {
    throw new TypeError("ballSetupUserOverridden must be a boolean");
  }
  const ballSetup = resolveBallSetup(snapshot.ballSetup);
  const spatialTarget = createSpatialTarget(snapshot.spatialTarget);
  if (!snapshot.ballSetupUserOverridden) {
    const expected = defaultBallSetupForClub(club);
    if (
      ballSetup.supportMode !== expected.supportMode ||
      ballSetup.teeHeightM !== expected.teeHeightM
    ) {
      throw new RangeError("club-default ball setup does not match the persisted club");
    }
  }
  return Object.freeze({
    ballSetup,
    ballSetupUserOverridden: snapshot.ballSetupUserOverridden,
    spatialTarget,
  });
}

/** Preserve v1 live values and make a cross-club default an explicit override. */
export function migratedLegacySimulationFallback(
  snapshot: SimulationWorkspaceSnapshot,
  club: ClubSpec,
): SimulationWorkspaceSnapshot {
  if (typeof snapshot?.ballSetupUserOverridden !== "boolean") {
    throw new TypeError("legacy simulation fallback must be complete");
  }
  const ballSetup = resolveBallSetup(snapshot.ballSetup);
  const expected = defaultBallSetupForClub(club);
  const mustBecomeOverride = !snapshot.ballSetupUserOverridden && (
    ballSetup.supportMode !== expected.supportMode ||
    ballSetup.teeHeightM !== expected.teeHeightM
  );
  return validatedSimulationWorkspace({
    ballSetup,
    ballSetupUserOverridden:
      snapshot.ballSetupUserOverridden || mustBecomeOverride,
    spatialTarget: snapshot.spatialTarget,
  }, club);
}

function ballSetupDocument(
  snapshot: SimulationWorkspaceSnapshot,
  club: ClubSpec,
): Record<string, unknown> {
  const validated = validatedSimulationWorkspace(snapshot, club);
  return {
    schema: BALL_SETUP_SELECTION_SCHEMA,
    schema_version: BALL_SETUP_SELECTION_SCHEMA_VERSION,
    setup: ballSetupToJson(validated.ballSetup),
    provenance: {
      kind: validated.ballSetupUserOverridden ? "explicit_override" : "club_default",
      club_name: validated.ballSetupUserOverridden ? null : club.name,
    },
  };
}

/** Serialize the strict simulation subpayload embedded in explorer-session v2. */
export function simulationWorkspaceDocument(
  snapshot: SimulationWorkspaceSnapshot,
  club: ClubSpec,
): Record<string, unknown> {
  const validated = validatedSimulationWorkspace(snapshot, club);
  return {
    schema: SIMULATION_SETUP_SCHEMA,
    schema_version: SIMULATION_SETUP_SCHEMA_VERSION,
    data: {
      ball_setup: ballSetupDocument(validated, club),
      spatial_target: JSON.parse(spatialTargetToJson(validated.spatialTarget)) as unknown,
    },
  };
}

function ballSetupFromDocument(
  value: unknown,
  club: ClubSpec,
): Pick<SimulationWorkspaceSnapshot, "ballSetup" | "ballSetupUserOverridden"> {
  const selection = exactRecord(
    value,
    ["schema", "schema_version", "setup", "provenance"],
    "ball_setup",
  );
  if (
    selection.schema !== BALL_SETUP_SELECTION_SCHEMA ||
    selection.schema_version !== BALL_SETUP_SELECTION_SCHEMA_VERSION
  ) {
    throw new RangeError("unsupported ball setup selection payload");
  }
  const setupData = exactRecord(selection.setup, BALL_SETUP_FIELDS, "ball_setup.setup");
  const ballSetup = ballSetupFromJson(setupData);
  const provenance = exactRecord(
    selection.provenance,
    ["kind", "club_name"],
    "ball_setup.provenance",
  );
  if (provenance.kind === "club_default") {
    const expected = defaultBallSetupForClub(club);
    if (
      provenance.club_name !== club.name ||
      ballSetup.supportMode !== expected.supportMode ||
      ballSetup.teeHeightM !== expected.teeHeightM
    ) {
      throw new RangeError("club-default ball setup does not match the persisted club");
    }
    return { ballSetup, ballSetupUserOverridden: false };
  }
  if (provenance.kind === "explicit_override") {
    if (provenance.club_name !== null) {
      throw new RangeError("explicit ball setup provenance cannot name a default club");
    }
    return { ballSetup, ballSetupUserOverridden: true };
  }
  throw new RangeError(`unknown ball setup provenance ${String(provenance.kind)}`);
}

/** Parse the complete subpayload before any React setter is called. */
export function simulationWorkspaceFromDocument(
  value: unknown,
  club: ClubSpec,
): SimulationWorkspaceSnapshot {
  const envelope = exactRecord(
    value,
    ["schema", "schema_version", "data"],
    "simulation_setup",
  );
  if (
    envelope.schema !== SIMULATION_SETUP_SCHEMA ||
    envelope.schema_version !== SIMULATION_SETUP_SCHEMA_VERSION
  ) {
    throw new RangeError("unsupported simulation setup payload");
  }
  const data = exactRecord(
    envelope.data,
    ["ball_setup", "spatial_target"],
    "simulation_setup.data",
  );
  const ball = ballSetupFromDocument(data.ball_setup, club);
  return validatedSimulationWorkspace({
    ...ball,
    spatialTarget: spatialTargetFromJson(JSON.stringify(data.spatial_target)),
  }, club);
}
