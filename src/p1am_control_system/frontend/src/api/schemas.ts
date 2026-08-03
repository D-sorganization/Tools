import { z } from "zod";

/**
 * Runtime contracts for server payloads (#3545).
 *
 * Responses used to be trusted via `as` casts and `any`, so a backend field
 * drift surfaced as a runtime `undefined.toFixed` deep in the UI. These zod
 * schemas are parsed in {@link apiFetch} so a contract mismatch fails loudly
 * and early with a descriptive ApiError instead.
 */

export const ladderTagInfoSchema = z.object({
  name: z.string(),
  tag_type: z.enum(["Real", "Boolean", "Integer", "String"]),
  description: z.string(),
  rw_mode: z.enum(["Read-only", "Read/Write"]),
  register_type: z.string().nullable(),
  register_num: z.number().nullable(),
  data_format: z.string().nullable(),
  scale_factor: z.number().nullable(),
  area: z.string(),
  unit: z.string(),
  equipment: z.string(),
});
export type LadderTagInfo = z.infer<typeof ladderTagInfoSchema>;

export const ladderExplorerSchema = z.array(ladderTagInfoSchema);

export const alicatMfcStateSchema = z.object({
  device_id: z.string(),
  name: z.string(),
  gas: z.string(),
  setpoint: z.number(),
  mass_flow: z.number(),
  volumetric_flow: z.number(),
  pressure: z.number(),
  temperature: z.number(),
  max_flow: z.number(),
  port_or_ip: z.string().nullable(),
  connection_state: z.string(),
});
export type AlicatMFCState = z.infer<typeof alicatMfcStateSchema>;

export const alicatListSchema = z.array(alicatMfcStateSchema);

export const eventLogEntrySchema = z.object({
  id: z.number(),
  event_type: z.string(),
  description: z.string(),
  severity: z.number(),
  timestamp: z.string(),
});
export type EventLogEntry = z.infer<typeof eventLogEntrySchema>;

export const eventLogSchema = z.array(eventLogEntrySchema);

export const activeAlarmSchema = z.object({
  tag_id: z.string(),
  // The backend labels each alarm with the tag name; older frames omit it.
  tag_name: z.string().optional(),
  state: z.string(),
  // `value` is NOT sent by the alarm engine — keep it optional. A strict
  // `value: z.number()` here made every active alarm fail validation, which
  // failed the whole telemetry frame and silently black-holed the live stream
  // (HMI stuck OFFLINE, controls dead) whenever any alarm was active.
  value: z.number().optional(),
  severity: z.number(),
  acknowledged: z.boolean(),
  timestamp: z.string(),
});
export type ActiveAlarm = z.infer<typeof activeAlarmSchema>;

export const activeAlarmsSchema = z.array(activeAlarmSchema);

export const signalQualitySchema = z.enum([
  "good",
  "uncertain",
  "bad",
  "stale",
  "simulated",
]);

export const signalSampleSchema = z.object({
  value: z.number(),
  source_timestamp: z.string(),
  server_timestamp: z.string(),
  quality: signalQualitySchema,
  diagnostic_reason: z.string().nullable(),
  sequence: z.number().int().positive(),
  source: z.string().min(1),
});
export type SignalSample = z.infer<typeof signalSampleSchema>;

export const commsHealthSchema = z.object({
  quality: signalQualitySchema,
  diagnostic_reason: z.string().nullable(),
  sequence: z.number().int().positive().nullable(),
  server_timestamp: z.string().nullable(),
  source: z.string().min(1),
});
export type CommsHealth = z.infer<typeof commsHealthSchema>;

/**
 * Live telemetry frame pushed over the `/api/stream` WebSocket.
 *
 * Every field is optional because the backend emits partial frames; consumers
 * read what is present rather than duck-typing each message inline.
 *
 * Resilience: each validated field is wrapped in `.catch(undefined)` so a schema
 * drift in ONE field (e.g. an alarm gaining/losing a key) drops just that field
 * instead of failing the whole frame. A rejected frame would leave the HMI
 * OFFLINE with dead controls — the live stream must degrade gracefully, never go
 * dark, on a single-field mismatch.
 */
export const telemetryFrameSchema = z.object({
  tags: z.array(z.number()).optional().catch(undefined),
  tags_dict: z.record(z.string(), z.number()).optional().catch(undefined),
  tag_samples: z
    .record(z.string(), signalSampleSchema)
    .optional()
    .catch(undefined),
  comms_health: commsHealthSchema.optional().catch(undefined),
  alicats: z.array(alicatMfcStateSchema).optional().catch(undefined),
  active_alarms: z
    .record(z.string(), activeAlarmSchema)
    .optional()
    .catch(undefined),
  e_stop_active: z.boolean().optional().catch(undefined),
  power_supply: z.unknown().optional(),
  temperature: z.unknown().optional(),
});
export type TelemetryFrame = z.infer<typeof telemetryFrameSchema>;

// --- Data capture / historian ------------------------------------------------

export const captureStatusSchema = z.object({
  capturing: z.boolean(),
  total_rows: z.number(),
  distinct_tags: z.number(),
  oldest_timestamp: z.string().nullable(),
  newest_timestamp: z.string().nullable(),
  span_seconds: z.number(),
  db_bytes: z.number(),
  event_rows: z.number(),
});
export type CaptureStatus = z.infer<typeof captureStatusSchema>;

export const captureClearResultSchema = z.object({
  tag_rows_deleted: z.number(),
  db_bytes_before: z.number(),
  db_bytes_after: z.number(),
});
export type CaptureClearResult = z.infer<typeof captureClearResultSchema>;

export const captureConfigSchema = z.object({
  /** Minimum seconds between historian writes (0 = every scan). */
  interval_s: z.number(),
});
export type CaptureConfig = z.infer<typeof captureConfigSchema>;

export const performanceConfigSchema = z.object({
  mode: z.enum(["performance", "lightweight"]),
  poll_interval_s: z.number(),
});
export type PerformanceConfig = z.infer<typeof performanceConfigSchema>;
export type PerformanceMode = PerformanceConfig["mode"];

/** PID auto-tuning result returned by `/api/pid/{i}/tuning/stop`. */
export const tuningResultSchema = z.object({
  status: z.string(),
  message: z.string(),
  parameters: z.object({
    kp: z.number(),
    tau: z.number(),
    theta: z.number(),
  }),
  recommended_pid: z.object({
    kp: z.number(),
    ki: z.number(),
    kd: z.number(),
  }),
});
export type TuningResult = z.infer<typeof tuningResultSchema>;

// --- Plant hierarchy ---------------------------------------------------------

export const hierarchicalTagSchema = z.object({
  name: z.string(),
  tag_type: z.string(),
  description: z.string(),
  rw_mode: z.string(),
  register_type: z.string().nullable(),
  register_num: z.number().nullable(),
  data_format: z.string().nullable(),
  scale_factor: z.number().nullable(),
});
export type HierarchicalTag = z.infer<typeof hierarchicalTagSchema>;

export const hierarchicalEquipmentSchema = z.object({
  name: z.string(),
  tags: z.array(hierarchicalTagSchema),
});
export type HierarchicalEquipment = z.infer<typeof hierarchicalEquipmentSchema>;

export const hierarchicalUnitSchema = z.object({
  name: z.string(),
  equipment: z.array(hierarchicalEquipmentSchema),
});
export type HierarchicalUnit = z.infer<typeof hierarchicalUnitSchema>;

export const hierarchicalAreaSchema = z.object({
  name: z.string(),
  units: z.array(hierarchicalUnitSchema),
});
export type HierarchicalArea = z.infer<typeof hierarchicalAreaSchema>;

export const plantHierarchySchema = z.array(hierarchicalAreaSchema);

/** MPC vs PID simulation comparison returned by `/api/mpc/simulate`. */
export const mpcSimResultSchema = z.object({
  status: z.string(),
  time: z.array(z.number()),
  pid: z.object({ pv: z.array(z.number()), cv: z.array(z.number()) }),
  mpc: z.object({ pv: z.array(z.number()), cv: z.array(z.number()) }),
});
export type MpcSimResult = z.infer<typeof mpcSimResultSchema>;
