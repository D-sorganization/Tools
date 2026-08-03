import { apiFetch } from "./client";
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

export function deployRouting(payload: unknown): Promise<unknown> {
  return apiFetch("/routing", { method: "POST", json: payload });
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
