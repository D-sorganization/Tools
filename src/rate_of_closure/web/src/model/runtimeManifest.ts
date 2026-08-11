/** Strict, immutable calculation-runtime manifest v1 contract. */

import { canonicalNumericJson } from "./flightGroundContract";
import { parseUniqueJson } from "./strictJson";

export const RUNTIME_MANIFEST_SCHEMA = "calculation-runtime-manifest/v1" as const;
const SURFACES = [
  "tools.pyqt6", "tools.react", "upstreamdrift.pyqt6", "upstreamdrift.react",
] as const;
const DOMAINS = ["impact", "flight", "ground"] as const;
const SOURCE_KINDS = [
  "installed_package", "source_checkout", "embedded_web_build", "test_fixture",
] as const;
const STABLE_ID = /^[a-z0-9][a-z0-9._/-]*$/;
const SEMVER = /^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
const SHA = /^[0-9a-f]{40}$/;
const PLACEHOLDER = /\b(?:fixme|placeholder|tbd|todo|unknown)\b/i;

export type RuntimeSurfaceId = typeof SURFACES[number];
export type CalculationDomain = typeof DOMAINS[number];
export type RuntimeAvailability = "available" | "unavailable";
export type RuntimeSourceKind = typeof SOURCE_KINDS[number];
export type RuntimeOptionValue = boolean | number | string;

export interface RuntimeOption {
  readonly option_id: string;
  readonly value: RuntimeOptionValue;
  readonly unit: string | null;
}

export interface RuntimeBuild {
  readonly package_name: string;
  readonly package_version: string;
  readonly tools_commit: string;
  readonly build_id: string;
}

export interface CalculationAuthority {
  readonly domain: CalculationDomain;
  readonly status: RuntimeAvailability;
  readonly reason: string | null;
  readonly model_id: string | null;
  readonly model_version: string | null;
  readonly implementation_authority: string | null;
  readonly backend: string | null;
  readonly integrator: string | null;
  readonly request_schema: string | null;
  readonly result_schema: string | null;
  readonly frame_id: string | null;
  readonly unit_system_id: string | null;
  readonly numerical_options: readonly RuntimeOption[];
}

export interface RuntimeProvenance {
  readonly source_kind: RuntimeSourceKind;
  readonly source_reference: string;
  readonly evidence_ids: readonly string[];
}

export interface CalculationRuntimeManifest {
  readonly schema_version: typeof RUNTIME_MANIFEST_SCHEMA;
  readonly surface_id: RuntimeSurfaceId;
  readonly build: RuntimeBuild;
  readonly calculations: readonly CalculationAuthority[];
  readonly provenance: RuntimeProvenance;
}

type UnknownRecord = Record<string, unknown>;

const record = (value: unknown, fields: readonly string[], name: string): UnknownRecord => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  const item = value as UnknownRecord;
  const keys = Object.keys(item).sort();
  const expected = [...fields].sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new RangeError(`${name} fields do not match v1 schema`);
  }
  return item;
};

const text = (value: unknown, name: string, stable = false): string => {
  if (typeof value !== "string" || !value.trim()) throw new TypeError(`${name} must be nonempty text`);
  if (/[\uD800-\uDFFF]/.test(value)) throw new RangeError(`${name} must not contain surrogates`);
  if (PLACEHOLDER.test(value)) throw new RangeError(`${name} must not contain a placeholder`);
  if (stable && !STABLE_ID.test(value)) throw new RangeError(`${name} must be a stable identifier`);
  return value;
};

const nullableStableId = (value: unknown, name: string): string | null =>
  value === null ? null : text(value, name, true);

const member = <T extends string>(value: unknown, values: readonly T[], name: string): T => {
  if (typeof value !== "string" || !values.includes(value as T)) {
    throw new RangeError(`${name} is unsupported`);
  }
  return value as T;
};

const optionValue = (value: unknown): RuntimeOptionValue => {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new RangeError("option value must be finite");
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw new RangeError("option value exceeds the cross-runtime safe integer range");
    }
    return value;
  }
  return text(value, "option value");
};

const parseOption = (value: unknown): RuntimeOption => {
  const item = record(value, ["option_id", "value", "unit"], "runtime option");
  const parsedValue = optionValue(item.value);
  const unit = item.unit === null ? null : text(item.unit, "option unit");
  const numeric = typeof parsedValue === "number";
  if (numeric !== (unit !== null)) {
    throw new RangeError("numeric options require a unit; text/bool options require null");
  }
  return Object.freeze({ option_id: text(item.option_id, "option_id", true), value: parsedValue, unit });
};

const parseBuild = (value: unknown): RuntimeBuild => {
  const item = record(value, ["package_name", "package_version", "tools_commit", "build_id"], "build");
  const version = text(item.package_version, "package_version");
  const commit = text(item.tools_commit, "tools_commit");
  if (!SEMVER.test(version)) throw new RangeError("package_version must use semantic version text");
  if (!SHA.test(commit)) throw new RangeError("tools_commit must be an exact lowercase SHA");
  return Object.freeze({
    package_name: text(item.package_name, "package_name", true), package_version: version,
    tools_commit: commit, build_id: text(item.build_id, "build_id", true),
  });
};

const AUTHORITY_FIELDS = [
  "model_id", "model_version", "implementation_authority", "backend", "integrator",
  "request_schema", "result_schema", "frame_id", "unit_system_id",
] as const;

const authorityIdentities = (item: UnknownRecord): Record<typeof AUTHORITY_FIELDS[number], string | null> =>
  Object.fromEntries(AUTHORITY_FIELDS.map((field) =>
    [field, nullableStableId(item[field], field)])) as Record<typeof AUTHORITY_FIELDS[number], string | null>;

const parseAuthority = (value: unknown): CalculationAuthority => {
  const fields = ["domain", "status", "reason", ...AUTHORITY_FIELDS, "numerical_options"];
  const item = record(value, fields, "calculation authority");
  const domain = member(item.domain, DOMAINS, "calculation domain");
  const status = member(item.status, ["available", "unavailable"] as const, "availability");
  const reason = item.reason === null ? null : text(item.reason, "reason");
  const identities = authorityIdentities(item);
  if (!Array.isArray(item.numerical_options)) throw new TypeError("numerical_options must be an array");
  const options = Object.freeze(item.numerical_options.map(parseOption));
  if (new Set(options.map(({ option_id }) => option_id)).size !== options.length) {
    throw new RangeError("numerical option IDs must be unique");
  }
  validateAvailability(status, reason, identities, options);
  return Object.freeze({ domain, status, reason, ...identities, numerical_options: options });
};

const validateAvailability = (
  status: RuntimeAvailability,
  reason: string | null,
  identities: Record<typeof AUTHORITY_FIELDS[number], string | null>,
  options: readonly RuntimeOption[],
): void => {
  const values = Object.values(identities);
  if (status === "available" && (reason !== null || values.some((value) => value === null))) {
    throw new RangeError("available calculation requires all identities and null reason");
  }
  if (status === "unavailable" &&
      (reason === null || values.some((value) => value !== null) || options.length)) {
    throw new RangeError("unavailable calculation requires reason, null identities, and no options");
  }
};

const parseProvenance = (value: unknown): RuntimeProvenance => {
  const item = record(value, ["source_kind", "source_reference", "evidence_ids"], "provenance");
  if (!Array.isArray(item.evidence_ids)) throw new TypeError("evidence_ids must be an array");
  const evidence = item.evidence_ids.map((entry) => text(entry, "evidence_id", true));
  if (!evidence.length || new Set(evidence).size !== evidence.length) {
    throw new RangeError("evidence_ids must be nonempty and unique");
  }
  return Object.freeze({
    source_kind: member(item.source_kind, SOURCE_KINDS, "source_kind"),
    source_reference: text(item.source_reference, "source_reference", true),
    evidence_ids: Object.freeze(evidence),
  });
};

/** Parse and deeply freeze one exact v1 manifest. */
export const parseRuntimeManifest = (value: unknown): CalculationRuntimeManifest => {
  const item = record(value, ["schema_version", "surface_id", "build", "calculations", "provenance"], "runtime manifest");
  if (item.schema_version !== RUNTIME_MANIFEST_SCHEMA) throw new RangeError("unsupported runtime manifest schema");
  if (!Array.isArray(item.calculations)) throw new TypeError("calculations must be an array");
  const parsed = Object.freeze(item.calculations.map(parseAuthority));
  if (parsed.length !== DOMAINS.length || parsed.some(({ domain }, index) => domain !== DOMAINS[index])) {
    throw new RangeError("calculations must contain impact, flight, ground in order");
  }
  return Object.freeze({
    schema_version: RUNTIME_MANIFEST_SCHEMA,
    surface_id: member(item.surface_id, SURFACES, "surface_id"),
    build: parseBuild(item.build), calculations: parsed, provenance: parseProvenance(item.provenance),
  });
};

export interface CreateRuntimeManifestInput {
  readonly surfaceId: RuntimeSurfaceId;
  readonly build: RuntimeBuild;
  readonly calculations: readonly CalculationAuthority[];
  readonly provenance: RuntimeProvenance;
}

/** Build a manifest only from explicit, caller-supplied identities. */
export const createRuntimeManifest = (input: CreateRuntimeManifestInput): CalculationRuntimeManifest =>
  parseRuntimeManifest({
    schema_version: RUNTIME_MANIFEST_SCHEMA, surface_id: input.surfaceId,
    build: input.build, calculations: input.calculations, provenance: input.provenance,
  });

/** Parse JSON with duplicate-field rejection before contract validation. */
export const runtimeManifestFromJson = (textValue: string): CalculationRuntimeManifest =>
  parseRuntimeManifest(parseUniqueJson(textValue));

/** Serialize with stable keys and the shared 11-decimal numeric policy. */
export const stableRuntimeManifestJson = (manifest: CalculationRuntimeManifest): string =>
  canonicalNumericJson(manifest);
