import { apiFetch, apiResponse } from "./client";
import {
  ladderExplorerSchema,
  alicatListSchema,
  eventLogSchema,
  activeAlarmsSchema,
  tuningResultSchema,
  mpcSimResultSchema,
  plantHierarchySchema,
  captureStatusSchema,
  captureClearResultSchema,
  captureConfigSchema,
  performanceConfigSchema,
  professionalAlarmSchema,
  professionalAlarmsSchema,
  configurationDiffSchema,
  configurationRevisionSchema,
  configurationRevisionsSchema,
  systemHealthSchema,
  type CaptureStatus,
  type CaptureClearResult,
  type CaptureConfig,
  type PerformanceConfig,
  type PerformanceMode,
  type LadderTagInfo,
  type AlicatMFCState,
  type EventLogEntry,
  type ActiveAlarm,
  type TuningResult,
  type MpcSimResult,
  type HierarchicalArea,
  type ProfessionalAlarm,
  type ConfigurationDiffEntry,
  type ConfigurationRevision,
  type SystemHealth,
} from "./schemas";

/**
 * Named, typed endpoint functions for the P1AM SCADA backend (#3542).
 *
 * Components call these instead of re-implementing `fetch("/api/...")`. Each
 * read endpoint validates its response against the matching zod schema so a
 * backend contract drift fails loudly rather than corrupting the UI.
 */

// --- Routing / configuration -------------------------------------------------

export function getRouting(): Promise<unknown> {
  // The routing payload is normalized in the caller (TAG_x <-> index mapping);
  // returned untyped here and mapped by RoutingConfig logic.
  return apiFetch("/routing");
}

export function createConfigurationDraft(
  payload: unknown,
  reason: string,
): Promise<ConfigurationRevision> {
  return apiFetch("/configurations/drafts", {
    method: "POST",
    json: { payload, reason },
    schema: configurationRevisionSchema,
  });
}

export function getConfigurationRevisions(): Promise<ConfigurationRevision[]> {
  return apiFetch("/configurations", { schema: configurationRevisionsSchema });
}

export function getConfigurationDiff(
  revisionId: string,
): Promise<ConfigurationDiffEntry[]> {
  return apiFetch(`/configurations/${encodeURIComponent(revisionId)}/diff`, {
    schema: configurationDiffSchema,
  });
}

function transitionConfiguration(
  revisionId: string,
  transition: "validate" | "review" | "activate",
): Promise<ConfigurationRevision> {
  return apiFetch(
    `/configurations/${encodeURIComponent(revisionId)}/${transition}`,
    { method: "POST", schema: configurationRevisionSchema },
  );
}

export const validateConfiguration = (revisionId: string) =>
  transitionConfiguration(revisionId, "validate");
export const reviewConfiguration = (revisionId: string) =>
  transitionConfiguration(revisionId, "review");
export const activateConfiguration = (revisionId: string) =>
  transitionConfiguration(revisionId, "activate");

export function approveConfiguration(
  revisionId: string,
  reason: string,
): Promise<ConfigurationRevision> {
  return apiFetch(`/configurations/${encodeURIComponent(revisionId)}/approve`, {
    method: "POST",
    json: { reason },
    schema: configurationRevisionSchema,
  });
}

export function rollbackConfiguration(
  revisionId: string,
  reason: string,
): Promise<ConfigurationRevision> {
  return apiFetch(`/configurations/${encodeURIComponent(revisionId)}/rollback`, {
    method: "POST",
    json: { reason },
    schema: configurationRevisionSchema,
  });
}

// --- System identity, health, and recovery ----------------------------------

export function getSystemHealth(): Promise<SystemHealth> {
  return apiFetch("/system/health", { schema: systemHealthSchema });
}

export type RecoveryDownload = {
  payload: Blob;
  sha256: string;
  configurationRevision: string;
};

export async function downloadRecoveryPackage(): Promise<RecoveryDownload> {
  const response = await apiResponse("/system/backups", { method: "POST" });
  const sha256 = response.headers.get("X-Artifact-SHA256");
  const configurationRevision = response.headers.get("X-Configuration-Revision");
  if (!sha256 || !configurationRevision) {
    throw new Error("Recovery response omitted identity headers");
  }
  return {
    payload: await response.blob(),
    sha256,
    configurationRevision,
  };
}

export async function restoreRecoveryPackage(
  payload: Blob,
  sha256: string,
  reason: string,
): Promise<ConfigurationRevision> {
  const response = await apiResponse("/system/restores", {
    method: "POST",
    body: payload,
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Artifact-SHA256": sha256,
      "X-Change-Reason": reason,
    },
  });
  const parsed = configurationRevisionSchema.safeParse(await response.json());
  if (!parsed.success) {
    throw new Error("Restore response did not match the revision contract");
  }
  return parsed.data;
}

export type EvidenceDownload = {
  payload: Blob;
  sha256: string;
  evidenceId: string;
  passed: boolean;
};

export async function runRepresentativeScenario(): Promise<EvidenceDownload> {
  const scenario = await apiFetch("/acceptance/scenarios/representative");
  const response = await apiResponse("/acceptance/scenarios/run", {
    method: "POST",
    body: JSON.stringify(scenario),
    headers: { "Content-Type": "application/json" },
  });
  const sha256 = response.headers.get("X-Artifact-SHA256");
  const evidenceId = response.headers.get("X-Evidence-ID");
  const passed = response.headers.get("X-Evidence-Passed");
  if (!sha256 || !evidenceId || !passed) {
    throw new Error("Acceptance response omitted evidence identity headers");
  }
  return {
    payload: await response.blob(),
    sha256,
    evidenceId,
    passed: passed === "true",
  };
}

// --- Tags --------------------------------------------------------------------

export function getLadderExplorer(): Promise<LadderTagInfo[]> {
  return apiFetch("/project/ladder-explorer", { schema: ladderExplorerSchema });
}

export function forceTag(tagId: number | string, value: number): Promise<unknown> {
  return apiFetch(`/tags/${tagId}`, { method: "POST", json: { value } });
}

// --- Safety ------------------------------------------------------------------

export function triggerEStop(): Promise<unknown> {
  return apiFetch("/estop", { method: "POST" });
}

export function clearEStop(): Promise<unknown> {
  return apiFetch("/estop/clear", { method: "POST" });
}

// --- Alarms & events ---------------------------------------------------------

export function getActiveAlarms(): Promise<ActiveAlarm[]> {
  return apiFetch("/alarms/active", { schema: activeAlarmsSchema });
}

export function getEvents(limit = 50): Promise<EventLogEntry[]> {
  return apiFetch(`/events?limit=${limit}`, { schema: eventLogSchema });
}

export function acknowledgeAlarm(tagId: string): Promise<unknown> {
  return apiFetch(`/alarms/${tagId}/acknowledge`, { method: "POST" });
}

export function getProfessionalAlarms(): Promise<ProfessionalAlarm[]> {
  return apiFetch("/alarm-management/active", {
    schema: professionalAlarmsSchema,
  });
}

export function acknowledgeProfessionalAlarm(
  tag: string,
): Promise<ProfessionalAlarm> {
  return apiFetch(`/alarm-management/${encodeURIComponent(tag)}/acknowledge`, {
    method: "POST",
    schema: professionalAlarmSchema,
  });
}

export function shelfProfessionalAlarm(
  tag: string,
  reason: string,
  durationSeconds: number,
): Promise<ProfessionalAlarm> {
  return apiFetch(`/alarm-management/${encodeURIComponent(tag)}/shelf`, {
    method: "POST",
    json: { reason, duration_seconds: durationSeconds },
    schema: professionalAlarmSchema,
  });
}

export function unshelveProfessionalAlarm(tag: string): Promise<ProfessionalAlarm> {
  return apiFetch(`/alarm-management/${encodeURIComponent(tag)}/shelf`, {
    method: "DELETE",
    schema: professionalAlarmSchema,
  });
}

// --- Alicat mass-flow controllers --------------------------------------------

export function getAlicats(): Promise<AlicatMFCState[]> {
  return apiFetch("/alicats", { schema: alicatListSchema });
}

export function setAlicatSetpoint(deviceId: string, setpoint: number): Promise<unknown> {
  return apiFetch(`/alicats/${deviceId}/setpoint`, {
    method: "POST",
    json: { setpoint },
  });
}

export function setAlicatGas(deviceId: string, gas: string): Promise<unknown> {
  return apiFetch(`/alicats/${deviceId}/gas`, { method: "POST", json: { gas } });
}

// --- PID tuning / MPC --------------------------------------------------------

export function startTuning(index: number): Promise<unknown> {
  return apiFetch(`/pid/${index}/tuning/start`, { method: "POST" });
}

export function stepTuning(index: number, stepValue: number): Promise<unknown> {
  return apiFetch(`/pid/${index}/tuning/step`, {
    method: "POST",
    json: { step_value: stepValue },
  });
}

export function stopTuning(index: number): Promise<TuningResult> {
  return apiFetch(`/pid/${index}/tuning/stop`, {
    method: "POST",
    schema: tuningResultSchema,
  });
}

export function simulateMpc(params: unknown): Promise<MpcSimResult> {
  return apiFetch("/mpc/simulate", {
    method: "POST",
    json: params,
    schema: mpcSimResultSchema,
  });
}

// --- Misc read endpoints -----------------------------------------------------

export function getPlant(): Promise<HierarchicalArea[]> {
  return apiFetch("/plant", { schema: plantHierarchySchema });
}

// --- Data capture / historian ------------------------------------------------

export function getCaptureStatus(): Promise<CaptureStatus> {
  return apiFetch("/capture/status", { schema: captureStatusSchema });
}

export function clearCapture(includeEvents = true): Promise<CaptureClearResult> {
  return apiFetch("/capture/clear", {
    method: "POST",
    json: { include_events: includeEvents },
    schema: captureClearResultSchema,
  });
}

export function getCaptureConfig(): Promise<CaptureConfig> {
  return apiFetch("/capture/config", { schema: captureConfigSchema });
}

export function setCaptureConfig(intervalSeconds: number): Promise<CaptureConfig> {
  return apiFetch("/capture/config", {
    method: "PUT",
    json: { interval_s: intervalSeconds },
    schema: captureConfigSchema,
  });
}

// --- Performance mode ---------------------------------------------------------

export function getPerformance(): Promise<PerformanceConfig> {
  return apiFetch("/performance", { schema: performanceConfigSchema });
}

export function setPerformanceMode(mode: PerformanceMode): Promise<PerformanceConfig> {
  return apiFetch("/performance", {
    method: "PUT",
    json: { mode },
    schema: performanceConfigSchema,
  });
}
