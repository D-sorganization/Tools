/** UI-neutral prescribed joint-torque profile contract shared with Python. */

import { parseUniqueJson } from "./strictJson";

export const TORQUE_PROFILE_SCHEMA_VERSION = 1;
export const TORQUE_UNIT = "N*m";
export const COEFFICIENT_ORDER = "ascending_c0_first";

export const TorqueProfileSource = Object.freeze({
  DIRECT: "direct",
  DRAWN: "drawn",
  IMPORTED: "imported",
  OPTIMIZED: "optimized",
  FITTED_RUN: "fitted_run",
} as const);
export type TorqueProfileSource =
  (typeof TorqueProfileSource)[keyof typeof TorqueProfileSource];

type JsonObject = Record<string, unknown>;

const PROFILE_FIELDS = [
  "schema_version", "profile_id", "model_id", "name", "description", "source",
  "source_metadata", "created_at_utc", "modified_at_utc", "torque_unit",
  "coefficient_order", "time_domain_s", "assignments",
] as const;
const ASSIGNMENT_FIELDS = ["joint_id", "coefficients", "fit_metadata"] as const;
const FIT_FIELDS = [
  "degree", "rmse_nm", "max_abs_error_nm", "r_squared", "condition_number",
  "original_sample_sha256",
] as const;
const STABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]*$/;
const METADATA_KEY = /^[a-z][a-z0-9_]*$/;
const SHA256 = /^[0-9a-f]{64}$/;
const UTC_TIMESTAMP =
  /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?Z$/;

function jsonObject(value: unknown, label: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return value as JsonObject;
}

function exactObject(
  value: unknown,
  fields: readonly string[],
  label: string,
): JsonObject {
  const object = jsonObject(value, label);
  const actual = Object.keys(object).sort();
  const expected = [...fields].sort();
  if (actual.length !== expected.length || actual.some((key, i) => key !== expected[i])) {
    throw new Error(`${label} fields must match the schema exactly`);
  }
  return object;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite real number`);
  }
  return value;
}

function finiteNumbers(value: unknown, label: string): readonly number[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  const result = value.map((item) => finiteNumber(item, label));
  if (result.length === 0) throw new Error(`${label} must not be empty`);
  return Object.freeze(result);
}

function stableId(value: unknown, label: string): string {
  if (typeof value !== "string" || !STABLE_ID.test(value)) {
    throw new Error(`invalid ${label}`);
  }
  return value;
}

function requiredText(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} is required`);
  }
  return value;
}

function immutableMetadata(value: unknown): Readonly<Record<string, string>> {
  const object = jsonObject(value, "source_metadata");
  const entries = Object.entries(object).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  if (entries.length === 0) throw new Error("source_metadata must not be empty");
  const normalized: Record<string, string> = {};
  for (const [key, item] of entries) {
    if (!METADATA_KEY.test(key)) throw new Error("invalid source_metadata key");
    normalized[key] = requiredText(item, "source_metadata value");
  }
  return Object.freeze(normalized);
}

function timestampMicros(value: unknown, label: string): bigint {
  if (typeof value !== "string") throw new Error(`invalid ${label}`);
  const match = UTC_TIMESTAMP.exec(value);
  if (!match) throw new Error(`invalid ${label}`);
  const milliseconds = Date.parse(`${match[1]}Z`);
  if (
    !Number.isFinite(milliseconds) ||
    new Date(milliseconds).toISOString().slice(0, 19) !== match[1]
  ) {
    throw new Error(`invalid ${label}`);
  }
  const fractionalMicros = BigInt((match[2] ?? "").padEnd(6, "0") || "0");
  return BigInt(milliseconds) * 1000n + fractionalMicros;
}

function timeDomain(value: unknown): readonly [number, number] {
  const domain = finiteNumbers(value, "time_domain_s");
  if (domain.length !== 2) {
    throw new Error("time_domain_s must contain exactly two values");
  }
  if (!(domain[0] < domain[1])) {
    throw new Error("time_domain_s must be strictly ordered");
  }
  return Object.freeze([domain[0], domain[1]]) as readonly [number, number];
}

function profileSource(value: unknown): TorqueProfileSource {
  if (!Object.values(TorqueProfileSource).includes(value as TorqueProfileSource)) {
    throw new Error("invalid profile source");
  }
  return value as TorqueProfileSource;
}

export interface FitMetadataInput {
  degree: number;
  rmseNm: number;
  maxAbsErrorNm: number;
  rSquared: number;
  conditionNumber: number;
  originalSampleSha256?: string | null;
}

export class FitMetadata {
  readonly degree: number;
  readonly rmseNm: number;
  readonly maxAbsErrorNm: number;
  readonly rSquared: number;
  readonly conditionNumber: number;
  readonly originalSampleSha256: string | null;

  constructor(input: FitMetadataInput) {
    if (!Number.isInteger(input.degree) || input.degree < 0) {
      throw new Error("degree must be a non-negative integer");
    }
    this.degree = input.degree;
    this.rmseNm = finiteNumber(input.rmseNm, "rmse_nm");
    this.maxAbsErrorNm = finiteNumber(input.maxAbsErrorNm, "max_abs_error_nm");
    this.rSquared = finiteNumber(input.rSquared, "r_squared");
    this.conditionNumber = finiteNumber(input.conditionNumber, "condition_number");
    this.originalSampleSha256 = input.originalSampleSha256 ?? null;
    if (this.rmseNm < 0 || this.maxAbsErrorNm < 0) {
      throw new Error("fit errors must be non-negative");
    }
    if (this.rSquared > 1) throw new Error("r_squared must be <= 1");
    if (!(this.conditionNumber > 0)) throw new Error("condition_number must be > 0");
    if (this.originalSampleSha256 !== null && !SHA256.test(this.originalSampleSha256)) {
      throw new Error("original_sample_sha256 must be lowercase SHA-256");
    }
    Object.freeze(this);
  }

  toJsonObject(): JsonObject {
    return {
      condition_number: this.conditionNumber,
      degree: this.degree,
      max_abs_error_nm: this.maxAbsErrorNm,
      original_sample_sha256: this.originalSampleSha256,
      r_squared: this.rSquared,
      rmse_nm: this.rmseNm,
    };
  }

  static fromJsonObject(value: unknown): FitMetadata {
    const data = exactObject(value, FIT_FIELDS, "fit_metadata");
    return new FitMetadata({
      degree: finiteNumber(data.degree, "degree"),
      rmseNm: finiteNumber(data.rmse_nm, "rmse_nm"),
      maxAbsErrorNm: finiteNumber(data.max_abs_error_nm, "max_abs_error_nm"),
      rSquared: finiteNumber(data.r_squared, "r_squared"),
      conditionNumber: finiteNumber(data.condition_number, "condition_number"),
      originalSampleSha256: data.original_sample_sha256 as string | null,
    });
  }
}

export class TorquePolynomial {
  readonly coefficients: readonly number[];
  readonly fitMetadata: FitMetadata | null;

  constructor(coefficients: readonly number[], fitMetadata: FitMetadata | null = null) {
    this.coefficients = finiteNumbers(coefficients, "coefficients");
    if (fitMetadata !== null && !(fitMetadata instanceof FitMetadata)) {
      throw new Error("fit_metadata must be FitMetadata");
    }
    if (fitMetadata !== null && fitMetadata.degree !== this.coefficients.length - 1) {
      throw new Error("fit degree must match polynomial coefficients");
    }
    this.fitMetadata = fitMetadata;
    Object.freeze(this);
  }

  evaluate(timeS: number): number {
    return evaluateValidatedPolynomial(
      this.coefficients,
      finiteNumber(timeS, "time_s"),
    );
  }
}

function evaluateValidatedPolynomial(
  coefficients: readonly number[],
  timeS: number,
): number {
  let result = 0;
  for (let index = coefficients.length - 1; index >= 0; index -= 1) {
    result = result * timeS + coefficients[index];
  }
  return finiteNumber(result, "evaluated torque");
}

export function evaluateAscendingPolynomial(
  coefficients: readonly number[],
  timeS: number,
): number {
  const normalized = finiteNumbers(coefficients, "coefficients");
  const time = finiteNumber(timeS, "time_s");
  return evaluateValidatedPolynomial(normalized, time);
}

export class JointTorqueAssignment {
  readonly jointId: string;
  readonly polynomial: TorquePolynomial;

  constructor(jointId: string, polynomial: TorquePolynomial) {
    this.jointId = stableId(jointId, "joint_id");
    if (!(polynomial instanceof TorquePolynomial)) {
      throw new Error("polynomial must be TorquePolynomial");
    }
    this.polynomial = polynomial;
    Object.freeze(this);
  }

  toJsonObject(): JsonObject {
    return {
      coefficients: [...this.polynomial.coefficients],
      fit_metadata: this.polynomial.fitMetadata?.toJsonObject() ?? null,
      joint_id: this.jointId,
    };
  }

  static fromJsonObject(value: unknown): JointTorqueAssignment {
    const data = exactObject(value, ASSIGNMENT_FIELDS, "assignment");
    const metadata = data.fit_metadata === null
      ? null
      : FitMetadata.fromJsonObject(data.fit_metadata);
    return new JointTorqueAssignment(
      stableId(data.joint_id, "joint_id"),
      new TorquePolynomial(finiteNumbers(data.coefficients, "coefficients"), metadata),
    );
  }
}

export interface PrescribedTorqueProfileInput {
  profileId: string;
  modelId: string;
  name: string;
  description: string;
  source: TorqueProfileSource;
  sourceMetadata: Readonly<Record<string, string>>;
  createdAtUtc: string;
  modifiedAtUtc: string;
  timeDomainS: readonly [number, number];
  assignments: readonly JointTorqueAssignment[];
}

export class PrescribedTorqueProfile {
  readonly profileId: string;
  readonly modelId: string;
  readonly name: string;
  readonly description: string;
  readonly source: TorqueProfileSource;
  readonly sourceMetadata: Readonly<Record<string, string>>;
  readonly createdAtUtc: string;
  readonly modifiedAtUtc: string;
  readonly timeDomainS: readonly [number, number];
  readonly assignments: readonly JointTorqueAssignment[];

  constructor(input: PrescribedTorqueProfileInput) {
    this.profileId = stableId(input.profileId, "profile_id");
    this.modelId = stableId(input.modelId, "model_id");
    this.name = requiredText(input.name, "name");
    this.description = requiredText(input.description, "description");
    this.source = profileSource(input.source);
    this.sourceMetadata = immutableMetadata(input.sourceMetadata);
    const created = timestampMicros(input.createdAtUtc, "created_at_utc");
    const modified = timestampMicros(input.modifiedAtUtc, "modified_at_utc");
    if (modified < created) throw new Error("modified_at_utc must not precede creation");
    this.createdAtUtc = input.createdAtUtc;
    this.modifiedAtUtc = input.modifiedAtUtc;
    this.timeDomainS = timeDomain(input.timeDomainS);
    this.assignments = Object.freeze([...input.assignments]);
    if (this.assignments.length === 0) throw new Error("assignments must not be empty");
    if (this.assignments.some((item) => !(item instanceof JointTorqueAssignment))) {
      throw new Error("assignments must contain JointTorqueAssignment values");
    }
    const jointIds = this.assignments.map((item) => item.jointId);
    if (new Set(jointIds).size !== jointIds.length) throw new Error("joint IDs must be unique");
    Object.freeze(this);
  }

  evaluate(timeS: number): Record<string, number> {
    const time = finiteNumber(timeS, "time_s");
    if (time < this.timeDomainS[0] || time > this.timeDomainS[1]) {
      throw new Error("time_s is outside profile domain");
    }
    return Object.fromEntries(
      this.assignments.map((item) => [item.jointId, item.polynomial.evaluate(time)]),
    );
  }

  toJsonObject(): JsonObject {
    return {
      assignments: this.assignments.map((item) => item.toJsonObject()),
      coefficient_order: COEFFICIENT_ORDER,
      created_at_utc: this.createdAtUtc,
      description: this.description,
      model_id: this.modelId,
      modified_at_utc: this.modifiedAtUtc,
      name: this.name,
      profile_id: this.profileId,
      schema_version: TORQUE_PROFILE_SCHEMA_VERSION,
      source: this.source,
      source_metadata: { ...this.sourceMetadata },
      time_domain_s: [...this.timeDomainS],
      torque_unit: TORQUE_UNIT,
    };
  }

  dumps(): string {
    return JSON.stringify(this.toJsonObject(), null, 2);
  }

  static fromJsonObject(value: unknown): PrescribedTorqueProfile {
    const data = exactObject(value, PROFILE_FIELDS, "profile");
    if (data.schema_version !== TORQUE_PROFILE_SCHEMA_VERSION) {
      throw new Error("unsupported schema_version");
    }
    if (data.torque_unit !== TORQUE_UNIT) throw new Error("unsupported torque_unit");
    if (data.coefficient_order !== COEFFICIENT_ORDER) {
      throw new Error("unsupported coefficient_order");
    }
    if (!Array.isArray(data.assignments)) throw new Error("assignments must be an array");
    return new PrescribedTorqueProfile({
      profileId: stableId(data.profile_id, "profile_id"),
      modelId: stableId(data.model_id, "model_id"),
      name: requiredText(data.name, "name"),
      description: requiredText(data.description, "description"),
      source: profileSource(data.source),
      sourceMetadata: immutableMetadata(data.source_metadata),
      createdAtUtc: requiredText(data.created_at_utc, "created_at_utc"),
      modifiedAtUtc: requiredText(data.modified_at_utc, "modified_at_utc"),
      timeDomainS: timeDomain(data.time_domain_s),
      assignments: data.assignments.map(JointTorqueAssignment.fromJsonObject),
    });
  }

  static loads(text: string): PrescribedTorqueProfile {
    if (typeof text !== "string") throw new Error("profile JSON must be text");
    return PrescribedTorqueProfile.fromJsonObject(parseUniqueJson(text));
  }
}
