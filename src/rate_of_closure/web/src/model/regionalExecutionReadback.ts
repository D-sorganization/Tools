/** Plan-bound readback of Python-produced regional execution evidence. */

import { canonicalGroundJson } from "./flightGroundContract";
import {
  groundRegionalExecutionResultFromJson,
  MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES,
  type GroundRegionalExecutionResult,
} from "./groundRegionalExecution";
import type { GroundRegionalMaterialPlanRequest } from "./groundRegionalPlan";
import type { GroundVec3 } from "./flightGroundTypes";

export interface RegionalExecutionReadback {
  readonly status: string;
  readonly failureReason: string | null;
  readonly planId: string;
  readonly surfaceId: string;
  readonly surfaceProviderId: string;
  readonly surfaceProviderVersion: string;
  readonly modelId: string;
  readonly modelVersion: string;
  readonly terminationReason: string | null;
  readonly groundTimeS: number | null;
  readonly completed: boolean | null;
  readonly transitionCount: number;
  readonly carryDistanceM: number | null;
  readonly bounceAirDistanceM: number | null;
  readonly skidDistanceM: number | null;
  readonly rollDistanceM: number | null;
  readonly surfacePathDistanceM: number | null;
  readonly totalDistanceM: number | null;
  readonly finalDownrangeM: number | null;
  readonly finalOfflineM: number | null;
  readonly bounceCount: number | null;
  readonly calibrationId: string | null;
  readonly calibrationKind: string | null;
  readonly calibrationSource: string | null;
  readonly calibrationConfidence: number | null;
  readonly observedPhases: readonly string[];
  readonly events: readonly RegionalExecutionEventReadback[];
  readonly transitions: readonly RegionalExecutionTransitionReadback[];
  readonly warnings: readonly RegionalExecutionWarningReadback[];
  readonly unitSystem: "SI";
  readonly executorSourceRevision: string;
  readonly executorInputSha256: string;
  readonly limitations: readonly string[];
}

export interface RegionalExecutionWarningReadback {
  readonly code: string;
  readonly severity: string;
  readonly message: string;
}

export interface RegionalExecutionEventReadback {
  readonly sequence: number;
  readonly eventType: string;
  readonly timeS: number;
  readonly frame: string;
  readonly positionM: GroundVec3;
  readonly velocityBeforeMps: GroundVec3;
  readonly velocityAfterMps: GroundVec3;
  readonly angularVelocityBeforeRadS: GroundVec3;
  readonly angularVelocityAfterRadS: GroundVec3;
}

export interface RegionalExecutionTransitionReadback {
  readonly eventSequence: number;
  readonly timeS: number;
  readonly positionM: GroundVec3;
  readonly fromRegionId: string | null;
  readonly toRegionId: string | null;
  readonly fromSurfaceId: string;
  readonly toSurfaceId: string;
}

export interface RegionalExecutionEvidence {
  readonly result: GroundRegionalExecutionResult;
  readonly readback: RegionalExecutionReadback;
}

export interface RegionalExecutionFile {
  readonly name: string;
  readonly size: number;
  text(): Promise<string>;
}

export const regionalExecutionReadback = (
  result: GroundRegionalExecutionResult,
  currentPlan: GroundRegionalMaterialPlanRequest,
): RegionalExecutionReadback => {
  if (canonicalGroundJson(result.regional_plan) !== canonicalGroundJson(currentPlan)) {
    throw new RangeError("execution evidence does not match the current regional plan");
  }
  const ground = result.ground_result;
  const summary = ground?.summary ?? null;
  const phases = Object.freeze(Array.from(new Set(
    ground?.trajectory.map((point) => point.phase) ?? [],
  )));
  const warnings = Object.freeze((ground?.warnings ?? []).map((warning) =>
    Object.freeze({
      code: warning.code,
      severity: warning.severity,
      message: warning.message,
    })));
  const events = Object.freeze((ground?.events ?? []).map((event) => Object.freeze({
    sequence: event.sequence,
    eventType: event.event_type,
    timeS: event.time_s,
    frame: event.frame,
    positionM: event.position_m,
    velocityBeforeMps: event.velocity_before_m_s,
    velocityAfterMps: event.velocity_after_m_s,
    angularVelocityBeforeRadS: event.angular_velocity_before_rad_s,
    angularVelocityAfterRadS: event.angular_velocity_after_rad_s,
  })));
  const transitions = Object.freeze(result.transitions.map((item) => Object.freeze({
    eventSequence: item.event_sequence,
    timeS: item.time_s,
    positionM: item.position_m,
    fromRegionId: item.from_region_id,
    toRegionId: item.to_region_id,
    fromSurfaceId: item.from_surface_id,
    toSurfaceId: item.to_surface_id,
  })));
  return Object.freeze({
    status: result.status,
    failureReason: result.failure_reason,
    planId: result.plan_id,
    surfaceId: result.surface_id,
    surfaceProviderId: result.regional_plan.base_surface.provider_id,
    surfaceProviderVersion: result.regional_plan.base_surface.provider_version,
    modelId: result.model_id,
    modelVersion: result.model_version,
    terminationReason: ground?.termination.reason ?? null,
    groundTimeS: ground?.termination.time_s ?? null,
    completed: ground?.termination.completed ?? null,
    transitionCount: result.transitions.length,
    carryDistanceM: summary?.carry_distance_m ?? null,
    bounceAirDistanceM: summary?.bounce_air_distance_m ?? null,
    skidDistanceM: summary?.skid_distance_m ?? null,
    rollDistanceM: summary?.roll_distance_m ?? null,
    surfacePathDistanceM: summary?.surface_path_distance_m ?? null,
    totalDistanceM: summary?.total_distance_m ?? null,
    finalDownrangeM: summary?.final_downrange_m ?? null,
    finalOfflineM: summary?.final_offline_m ?? null,
    bounceCount: summary?.bounce_count ?? null,
    calibrationId: ground?.calibration.calibration_id ?? null,
    calibrationKind: ground?.calibration.kind ?? null,
    calibrationSource: ground?.calibration.source ?? null,
    calibrationConfidence: ground?.calibration.confidence ?? null,
    observedPhases: phases,
    events,
    transitions,
    warnings,
    unitSystem: result.unit_system,
    executorSourceRevision: result.executor_provenance.source_revision,
    executorInputSha256: result.executor_provenance.input_sha256,
    limitations: result.limitations,
  });
};

export const readRegionalExecutionEvidenceFile = async (
  file: RegionalExecutionFile,
  currentPlan: GroundRegionalMaterialPlanRequest,
): Promise<RegionalExecutionEvidence> => {
  if (file.size > MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES) {
    throw new RangeError("regional execution evidence exceeds maximum wire size");
  }
  const result = groundRegionalExecutionResultFromJson(await file.text());
  return Object.freeze({
    result,
    readback: regionalExecutionReadback(result, currentPlan),
  });
};
