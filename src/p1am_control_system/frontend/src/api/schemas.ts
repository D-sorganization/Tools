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

export const professionalAlarmSchema = z.object({
  tag: z.string(),
  priority: z.enum(["critical", "high", "medium", "low"]),
  lifecycle: z.enum([
    "inactive",
    "unacknowledged",
    "acknowledged",
    "returned_unacknowledged",
    "shelved",
    "suppressed",
  ]),
  condition: z.string(),
  acknowledged_by: z.string().nullable(),
  shelved_by: z.string().nullable(),
  shelf_reason: z.string().nullable(),
  shelf_until: z.string().nullable(),
  suppression_rule: z.string().nullable(),
  first_out_sequence: z.number().int().positive().nullable(),
  active_since: z.string().nullable(),
  help_text: z.string(),
});
export const professionalAlarmsSchema = z.array(professionalAlarmSchema);
export type ProfessionalAlarm = z.infer<typeof professionalAlarmSchema>;

export const configurationStateSchema = z.enum([
  "draft",
  "validated",
  "in_review",
  "approved",
  "active",
  "superseded",
]);
export const configurationRevisionSchema = z.object({
  revision_id: z.string(),
  version: z.number().int().positive(),
  state: configurationStateSchema,
  payload: z.unknown(),
  payload_sha256: z.string().regex(/^[0-9a-f]{64}$/),
  reason: z.string(),
  created_by: z.string(),
  created_at: z.string(),
  validated_by: z.string().nullable(),
  reviewed_by: z.string().nullable(),
  approved_by: z.string().nullable(),
  activated_by: z.string().nullable(),
  activated_at: z.string().nullable(),
  activation_identity: z.string().nullable(),
  source_revision_id: z.string().nullable(),
});
export const configurationRevisionsSchema = z.array(configurationRevisionSchema);
export const configurationDiffSchema = z.array(
  z.object({
    path: z.string(),
    before: z.unknown().nullable(),
    after: z.unknown().nullable(),
  }),
);
export type ConfigurationRevision = z.infer<typeof configurationRevisionSchema>;
export type ConfigurationDiffEntry = z.infer<typeof configurationDiffSchema>[number];

export const deploymentIdentitySchema = z.object({
  software_revision: z.string(),
  configuration_revision: z.string(),
  configuration_sha256: z.string().nullable(),
  configuration_state: z.string(),
});
export const systemHealthSchema = z.object({
  generated_at: z.string(),
  overall: z.enum(["good", "degraded", "bad"]),
  identity: deploymentIdentitySchema,
  checks: z.array(
    z.object({
      name: z.string(),
      status: z.enum(["good", "degraded", "bad"]),
      detail: z.string(),
    }),
  ),
});
export type DeploymentIdentity = z.infer<typeof deploymentIdentitySchema>;
export type SystemHealth = z.infer<typeof systemHealthSchema>;

// --- Representative operator workspace -------------------------------------

export const faceplateValueSchema = z.object({
  value: z.number(),
  unit: z.string().min(1),
  source_timestamp: z.string(),
});
export const assetFaceplateSchema = z.object({
  asset_id: z.string().startsWith("SYNTHETIC."),
  label: z.string(),
  asset_type: z.enum(["pump", "valve", "vessel", "heater", "separator"]),
  primary_value: faceplateValueSchema,
  quality: z.enum(["good", "uncertain", "bad", "stale", "simulated"]),
  mode: z.enum(["off", "manual", "automatic", "unavailable"]),
  alarm_state: z.enum(["normal", "active", "shelved", "suppressed"]),
  interlock_state: z.enum(["clear", "permissive_missing", "tripped"]),
  detail_route: z.string(),
  trend_tags: z.array(z.string().startsWith("SYNTHETIC.")).min(1),
});
export const processOverviewSchema = z.object({
  overview_id: z.string().startsWith("SYNTHETIC."),
  title: z.string(),
  areas: z.array(
    z.object({
      area_id: z.string().startsWith("SYNTHETIC."),
      label: z.string(),
      detail_route: z.string(),
      assets: z.array(assetFaceplateSchema),
    }),
  ),
  data_classification: z.literal("synthetic"),
  not_for_live_control: z.literal(true),
});
export const protectionDefinitionSchema = z.object({
  protection_id: z.string().startsWith("SYNTHETIC."),
  category: z.enum(["control", "interlock", "independent_protection"]),
  consequences: z.array(z.string()).min(1),
  bypassable: z.boolean(),
});
export const tripRecordSchema = z.object({
  protection_id: z.string().startsWith("SYNTHETIC."),
  group_id: z.string(),
  category: z.enum(["control", "interlock", "independent_protection"]),
  consequences: z.array(z.string()),
  occurred_at: z.string(),
  first_out: z.boolean(),
});
export const managedBypassSchema = z.object({
  protection_id: z.string().startsWith("SYNTHETIC."),
  actor: z.string(),
  reason: z.string(),
  requested_at: z.string(),
  expires_at: z.string(),
  banner_required: z.literal(true),
  active: z.literal(true),
});
export const protectionSnapshotSchema = z.object({
  definitions: z.array(protectionDefinitionSchema),
  trips: z.array(tripRecordSchema),
  active_bypasses: z.array(managedBypassSchema),
});
export type AssetFaceplate = z.infer<typeof assetFaceplateSchema>;
export type ProcessOverview = z.infer<typeof processOverviewSchema>;
export type ProtectionSnapshot = z.infer<typeof protectionSnapshotSchema>;

export const assetHealthReportSchema = z.object({
  asset_id: z.string().startsWith("SYNTHETIC."),
  generated_at: z.string(),
  counters: z.object({
    runtime_seconds: z.number().nonnegative(),
    start_count: z.number().int().nonnegative(),
  }),
  statistics: z.object({
    sample_count: z.number().int().positive(),
    minimum: z.number(),
    maximum: z.number(),
    mean: z.number(),
    standard_deviation: z.number().nonnegative(),
  }),
  advisories: z.array(
    z.object({
      code: z.enum([
        "calibration_due",
        "drift",
        "flatline",
        "command_feedback_mismatch",
        "noisy_signal",
      ]),
      asset_id: z.string().startsWith("SYNTHETIC."),
      detected_at: z.string(),
      detail: z.string(),
      classification: z.literal("maintenance_advisory"),
      authoritative_trip: z.literal(false),
    }),
  ),
  data_classification: z.literal("synthetic"),
});
export const shiftEntrySchema = z.object({
  entry_id: z.string(),
  shift_id: z.string().startsWith("SYNTHETIC."),
  run_id: z.string().startsWith("SYNTHETIC."),
  summary: z.string(),
  unresolved_actions: z.array(z.string()),
  event_references: z.array(
    z.object({ event_id: z.string().startsWith("SYNTHETIC."), occurred_at: z.string() }),
  ),
  trend_references: z.array(
    z.object({
      investigation_id: z.string().startsWith("SYNTHETIC."),
      content_sha256: z.string().regex(/^[0-9a-f]{64}$/),
    }),
  ),
  created_by: z.string(),
  created_at: z.string(),
  data_classification: z.literal("synthetic"),
});
export const shiftEntriesSchema = z.array(shiftEntrySchema);
export type AssetHealthReport = z.infer<typeof assetHealthReportSchema>;
export type ShiftEntry = z.infer<typeof shiftEntrySchema>;

export const productStatusSchema = z.object({
  procedure_state: z.enum([
    "idle",
    "starting",
    "running",
    "holding",
    "stopping",
    "aborted",
    "recovering",
  ]),
  procedure_events: z.array(z.unknown()),
  connectors: z.array(
    z.object({
      connector_id: z.string().startsWith("SYNTHETIC."),
      version: z.string(),
      details: z.record(z.string(), z.unknown()),
    }),
  ),
  samples: z.record(
    z.string(),
    z.object({
      value: z.number().nullable(),
      quality: z.enum(["good", "bad"]),
      diagnostic: z.string(),
      connector_id: z.string().startsWith("SYNTHETIC."),
    }),
  ),
  notification_policy: z.object({
    primary_recipient: z.string(),
    escalation_recipient: z.string(),
  }).passthrough(),
  notification_audit: z.array(z.unknown()),
  availability: z.object({
    recovery_time_objective_seconds: z.number().positive(),
    recovery_point_objective_seconds: z.number().positive(),
    clock_ordering_reliable: z.boolean(),
    command_authority: z.string().nullable(),
    transport_available: z.boolean(),
    hmi_available: z.boolean(),
    buffered_samples: z.number().int().nonnegative(),
    data_classification: z.literal("synthetic"),
  }),
  data_classification: z.literal("synthetic"),
  not_for_live_control: z.literal(true),
});
export type ProductStatus = z.infer<typeof productStatusSchema>;

const sha256Schema = z.string().regex(/^[0-9a-f]{64}$/);
export const advisoryResultSchema = z.object({
  advisory_id: z.string().startsWith("ADV-"),
  generated_at: z.string(),
  model: z.object({
    model_id: z.literal("SYNTHETIC.MODEL.ADVISORY"),
    version: z.string(),
    algorithm: z.string(),
    artifact_sha256: sha256Schema,
  }),
  data: z.object({
    dataset_id: z.string().startsWith("SYNTHETIC."),
    content_sha256: sha256Schema,
    feature_names: z.array(z.string()),
  }),
  constraints: z.object({
    minimum: z.number(),
    maximum: z.number(),
    unit: z.string(),
  }),
  confidence: z.object({
    level: z.number().gt(0).lt(1),
    lower: z.number(),
    estimate: z.number(),
    upper: z.number(),
  }),
  recommended_setpoint: z.number(),
  recommendation: z.string(),
  limitation: z.string(),
  replay: z.object({
    input_sha256: sha256Schema,
    result_sha256: sha256Schema,
    verified: z.literal(true),
  }),
  authoritative_write_available: z.literal(false),
  data_classification: z.literal("synthetic"),
  not_for_live_control: z.literal(true),
});
export type AdvisoryResult = z.infer<typeof advisoryResultSchema>;

export const advisoryDispositionSchema = z.object({
  advisory_id: z.string(),
  decision: z.enum(["accepted_for_review", "rejected", "deferred"]),
  reason: z.string(),
  actor: z.string(),
  recorded_at: z.string(),
  applied_to_control: z.literal(false),
});
export type AdvisoryDisposition = z.infer<typeof advisoryDispositionSchema>;

/**
 * One entry of the `active_alarms` map, with resilience at the ENTRY level.
 *
 * `.catch(undefined)` deliberately sits here rather than on the enclosing
 * record (#4011): on the record, ONE malformed alarm object erased the entire
 * map, `applyFrame`'s `if (frame.active_alarms)` went false, `setActiveAlarms`
 * was never called again, and the HMI kept rendering its last list — including
 * "All normal — no active alarms" — while alarms fired on the PLC. Silent and
 * permanent for the session. Per entry, one bad alarm costs you that alarm.
 */
export const activeAlarmEntrySchema = activeAlarmSchema.optional().catch(undefined);

/**
 * An `active_alarms` map as it survives parsing: an entry is `undefined` when
 * that single alarm failed validation and was dropped.
 */
export type ActiveAlarmMap = Record<string, ActiveAlarm | undefined>;

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
 *
 * CAUTION: because every field is optional, `{}` — and any object made only of
 * fields the HMI does not know — parses SUCCESSFULLY. Parse success therefore
 * says nothing about whether a live frame arrived. Use {@link hasTelemetryContent}
 * to decide liveness (#4010).
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
  active_alarms: z.record(z.string(), activeAlarmEntrySchema).optional().catch(undefined),
  e_stop_active: z.boolean().optional().catch(undefined),
  power_supply: z.unknown().optional(),
  temperature: z.unknown().optional(),
});
export type TelemetryFrame = z.infer<typeof telemetryFrameSchema>;

/**
 * The payload fields the HMI actually understands. Single source of truth for
 * "does this frame carry telemetry at all".
 */
export const TELEMETRY_CONTENT_FIELDS = [
  "tags",
  "tags_dict",
  "alicats",
  "active_alarms",
  "e_stop_active",
  "power_supply",
  "temperature",
] as const satisfies readonly (keyof TelemetryFrame)[];

/**
 * True when a parsed frame carries at least one recognised telemetry field.
 *
 * The backend's `latest_frame` starts life as `{}` and is only ever reassigned
 * on a SUCCESSFUL poll — never cleared, never aged, always served with HTTP
 * 200. So when the poll loop dies, `/api/snapshot` keeps returning a payload
 * that parses fine and carries nothing. Counting that as a live frame is what
 * kept the HMI green on a dead backend.
 *
 * @param frame - a frame already validated by {@link telemetryFrameSchema}.
 * @returns whether the frame should refresh the data-age clock.
 */
export function hasTelemetryContent(frame: TelemetryFrame): boolean {
  for (const field of TELEMETRY_CONTENT_FIELDS) {
    if (frame[field] !== undefined) return true;
  }
  return false;
}

/** Well-formed alarms plus the ids of entries that failed validation. */
export interface AlarmPartition {
  alarms: ActiveAlarm[];
  /** Map keys whose alarm object was malformed and therefore dropped. */
  droppedIds: string[];
}

/**
 * Split a parsed `active_alarms` map into surviving alarms and dropped ids.
 *
 * Callers render `alarms` and surface a degraded-data banner whenever
 * `droppedIds` is non-empty, so a partially-parsed alarm map is visible to the
 * operator rather than silently shrinking the list.
 *
 * @param map - the parsed map, or `undefined` when the frame carried none.
 */
export function partitionAlarmMap(map: ActiveAlarmMap | undefined): AlarmPartition {
  const alarms: ActiveAlarm[] = [];
  const droppedIds: string[] = [];
  if (!map) return { alarms, droppedIds };
  for (const key of Object.keys(map)) {
    const alarm = map[key];
    if (alarm === undefined) droppedIds.push(key);
    else alarms.push(alarm);
  }
  return { alarms, droppedIds };
}

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
