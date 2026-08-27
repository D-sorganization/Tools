/** Strict browser consumer of the packaged locus-execution authority. */

import rawContract from "../../../locus_execution_capabilities.v1.json";

export type TimeWindowPolicy = "forbidden" | "required_half_open_seconds";
export type PointLocusPolicy = "forbidden" | "required_exact_topological";

export interface LocusExecutionCapabilityTs {
  readonly variableKey: string;
  readonly supported: boolean;
  readonly adapterId: string | null;
  readonly wholeRun: boolean;
  readonly timeWindowPolicy: TimeWindowPolicy;
  readonly pointLocusPolicy: PointLocusPolicy;
  readonly pointIds: readonly string[];
  readonly unsupportedReason: string | null;
}

export interface LocusExecutionContractTs {
  readonly schemaVersion: "rate-locus-execution-capabilities/v1";
  readonly mode: string;
  readonly sourceKind: string;
  readonly pointIdSemantics: string;
  readonly timeWindowSemantics: string;
  readonly capabilities: ReadonlyMap<string, LocusExecutionCapabilityTs>;
}

const record = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};

const text = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !value || value.trim() !== value) {
    throw new Error(`${label} must be a non-empty trimmed string`);
  }
  return value;
};

const nullableText = (value: unknown, label: string): string | null =>
  value === null ? null : text(value, label);

const bool = (value: unknown, label: string): boolean => {
  if (typeof value !== "boolean") throw new Error(`${label} must be Boolean`);
  return value;
};

const exactFields = (
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void => {
  const observed = Object.keys(value).sort();
  if (JSON.stringify(observed) !== JSON.stringify([...expected].sort())) {
    throw new Error(`${label} fields do not match the v1 schema`);
  }
};

const parseCapability = (raw: unknown): LocusExecutionCapabilityTs => {
  const row = record(raw, "capability");
  exactFields(row, [
    "variable_key", "supported", "adapter_id", "whole_run",
    "time_window_policy", "point_locus_policy", "point_ids",
    "unsupported_reason",
  ], "capability");
  const timeWindowPolicy = text(row.time_window_policy, "time_window_policy");
  if (!(["forbidden", "required_half_open_seconds"] as string[])
    .includes(timeWindowPolicy)) throw new Error("unsupported time_window_policy");
  const pointLocusPolicy = text(row.point_locus_policy, "point_locus_policy");
  if (!(["forbidden", "required_exact_topological"] as string[])
    .includes(pointLocusPolicy)) throw new Error("unsupported point_locus_policy");
  if (!Array.isArray(row.point_ids)) throw new Error("point_ids must be an array");
  const pointIds = row.point_ids.map((point) => text(point, "point_id"));
  if (new Set(pointIds).size !== pointIds.length) throw new Error("point_ids must be unique");
  const capability: LocusExecutionCapabilityTs = Object.freeze({
    variableKey: text(row.variable_key, "variable_key"),
    supported: bool(row.supported, "supported"),
    adapterId: nullableText(row.adapter_id, "adapter_id"),
    wholeRun: bool(row.whole_run, "whole_run"),
    timeWindowPolicy: timeWindowPolicy as TimeWindowPolicy,
    pointLocusPolicy: pointLocusPolicy as PointLocusPolicy,
    pointIds: Object.freeze(pointIds),
    unsupportedReason: nullableText(row.unsupported_reason, "unsupported_reason"),
  });
  validateCapability(capability);
  return capability;
};

const validateCapability = (capability: LocusExecutionCapabilityTs): void => {
  if (capability.wholeRun && capability.timeWindowPolicy !== "forbidden") {
    throw new Error("whole-run capability cannot require a time window");
  }
  if (capability.wholeRun && capability.pointLocusPolicy !== "forbidden") {
    throw new Error("whole-run capability cannot require a point locus");
  }
  if (capability.pointLocusPolicy === "forbidden" && capability.pointIds.length) {
    throw new Error("forbidden point locus must have no point_ids");
  }
  if (capability.pointLocusPolicy === "required_exact_topological" &&
      !(capability.timeWindowPolicy === "required_half_open_seconds" &&
        capability.pointIds.length === 1)) {
    throw new Error("localized capability requires one exact point_id");
  }
  if (capability.supported) {
    if (capability.adapterId === null || capability.unsupportedReason !== null) {
      throw new Error("supported capability requires only adapter_id");
    }
  } else if (capability.adapterId !== null || capability.wholeRun ||
      capability.timeWindowPolicy !== "forbidden" ||
      capability.pointLocusPolicy !== "forbidden" || capability.pointIds.length ||
      capability.unsupportedReason === null) {
    throw new Error("unsupported capability requires unsupported_reason and no execution locus");
  }
};

export const parseLocusExecutionContract = (
  payload: unknown,
  registeredKeys: Iterable<string>,
): LocusExecutionContractTs => {
  const document = record(payload, "locus execution contract");
  exactFields(document, [
    "schema_version", "mode", "source_kind", "point_id_semantics",
    "time_window_semantics", "capabilities",
  ], "top-level");
  if (document.schema_version !== "rate-locus-execution-capabilities/v1") {
    throw new Error("unsupported schema_version");
  }
  if (!Array.isArray(document.capabilities)) throw new Error("capabilities must be an array");
  const capabilities = new Map<string, LocusExecutionCapabilityTs>();
  for (const raw of document.capabilities) {
    const capability = parseCapability(raw);
    if (capabilities.has(capability.variableKey)) {
      throw new Error(`duplicate variable_key: ${capability.variableKey}`);
    }
    capabilities.set(capability.variableKey, capability);
  }
  const expected = new Set(registeredKeys);
  const missing = [...expected].filter((key) => !capabilities.has(key));
  const extra = [...capabilities.keys()].filter((key) => !expected.has(key));
  if (missing.length || extra.length) {
    throw new Error(`registry coverage mismatch: missing=${missing}, extra=${extra}`);
  }
  return Object.freeze({
    schemaVersion: "rate-locus-execution-capabilities/v1" as const,
    mode: text(document.mode, "mode"),
    sourceKind: text(document.source_kind, "source_kind"),
    pointIdSemantics: text(document.point_id_semantics, "point_id_semantics"),
    timeWindowSemantics: text(document.time_window_semantics, "time_window_semantics"),
    capabilities,
  });
};

const AUTHORITY_KEYS = (rawContract.capabilities as Array<{ variable_key: string }>)
  .map(({ variable_key: variableKey }) => variableKey);

export const LOCUS_EXECUTION_CONTRACT = parseLocusExecutionContract(
  rawContract,
  AUTHORITY_KEYS,
);

export const capabilityFor = (variableKey: string): LocusExecutionCapabilityTs => {
  const capability = LOCUS_EXECUTION_CONTRACT.capabilities.get(variableKey);
  if (capability === undefined) throw new Error(`variable is not declared: ${variableKey}`);
  return capability;
};

export const validateLocusMetadata = (
  variableKey: string,
  timeWindowS: readonly [number, number] | null | undefined,
  pointIds: readonly string[],
  maxDurationS: number,
): void => {
  const capability = capabilityFor(variableKey);
  if (capability.timeWindowPolicy === "forbidden") {
    if (timeWindowS != null || pointIds.length) {
      throw new Error(`locus metadata is forbidden for whole-run variable ${variableKey}`);
    }
    return;
  }
  if (timeWindowS == null) throw new Error("localized torque requires a finite half-open time window");
  const [start, end] = timeWindowS;
  if (!(0 <= start && start < end && end <= maxDurationS)) {
    throw new Error(`localized torque window must satisfy 0 <= start < end <= ${maxDurationS} s`);
  }
  if (pointIds.length !== 1 || pointIds[0] !== capability.pointIds[0]) {
    throw new Error(
      `localized torque requires exact topological joint ${capability.pointIds[0]}; swing.* IDs are spatial`,
    );
  }
};
