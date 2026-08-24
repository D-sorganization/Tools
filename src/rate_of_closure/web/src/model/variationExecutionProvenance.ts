/** Exact producer/source provenance for one persisted variation plan. */

import { sha256Text } from "./sha256";

export const PRODUCER_PROVENANCE_SCHEMA_ID =
  "rate-of-closure/variation-plan-provenance" as const;
export const PRODUCER_PROVENANCE_SCHEMA_VERSION = 1 as const;

export interface PlanProducerProvenanceTs {
  readonly schemaId: typeof PRODUCER_PROVENANCE_SCHEMA_ID;
  readonly schemaVersion: typeof PRODUCER_PROVENANCE_SCHEMA_VERSION;
  readonly producerId: string;
  readonly producerVersion: number;
  readonly sourceRepository: string;
  readonly sourceRevision: string | null;
  readonly sourceRevisionStatus: "exact" | "unavailable";
  readonly sourceRevisionReason: string | null;
}

export interface PlanProducerProvenanceInput {
  readonly producerId: string;
  readonly producerVersion: number;
  readonly sourceRepository: string;
  readonly sourceRevision: string | null;
  readonly sourceRevisionStatus: "exact" | "unavailable";
  readonly sourceRevisionReason: string | null;
}

export interface PlanProducerProvenanceWire {
  schema_id: string;
  schema_version: number;
  producer_id: string;
  producer_version: number;
  source_repository: string;
  source_revision: string | null;
  source_revision_status: string;
  source_revision_reason: string | null;
}

const COMMIT = /^[0-9a-f]{40}$/;
const STABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/;
const FIELDS = [
  "producer_id", "producer_version", "schema_id", "schema_version",
  "source_repository", "source_revision", "source_revision_reason",
  "source_revision_status",
];

const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("provenance must be an object");
  }
  const item = value as Record<string, unknown>;
  if (JSON.stringify(Object.keys(item).sort()) !== JSON.stringify(FIELDS)) {
    throw new Error("provenance fields mismatch");
  }
  return item;
};

const text = (value: unknown, name: string): string => {
  if (typeof value !== "string") throw new Error(`${name} must be text`);
  return value;
};

const nullableText = (value: unknown, name: string): string | null =>
  value === null ? null : text(value, name);

/** Build and deeply freeze one validated producer/source record. */
export const makePlanProducerProvenance = (
  input: PlanProducerProvenanceInput,
): PlanProducerProvenanceTs => {
  if (!STABLE_ID.test(input.producerId) || !STABLE_ID.test(input.sourceRepository)) {
    throw new Error("producer and repository must be stable identifiers");
  }
  if (!Number.isSafeInteger(input.producerVersion) || input.producerVersion <= 0) {
    throw new Error("producer version must be a positive safe integer");
  }
  if (input.sourceRevisionStatus === "exact") {
    if (input.sourceRevision === null || !COMMIT.test(input.sourceRevision)) {
      throw new Error("exact source revision must be a lowercase Git commit");
    }
    if (input.sourceRevisionReason !== null) {
      throw new Error("exact source revision must not have an unavailability reason");
    }
  } else if (input.sourceRevision !== null ||
      input.sourceRevisionReason === null || input.sourceRevisionReason.trim().length < 16) {
    throw new Error("unavailable source revision requires an unavailability reason");
  }
  return Object.freeze({
    schemaId: PRODUCER_PROVENANCE_SCHEMA_ID,
    schemaVersion: PRODUCER_PROVENANCE_SCHEMA_VERSION,
    ...input,
  });
};

export const REACT_DEFAULT_PROVENANCE = makePlanProducerProvenance({
  producerId: "rate-of-closure/react",
  producerVersion: 1,
  sourceRepository: "D-sorganization/Tools",
  sourceRevision: null,
  sourceRevisionStatus: "unavailable",
  sourceRevisionReason:
    "This runtime did not receive an exact Tools source revision at build time.",
});

export const provenanceToWire = (
  value: PlanProducerProvenanceTs,
): PlanProducerProvenanceWire => ({
  schema_id: value.schemaId,
  schema_version: value.schemaVersion,
  producer_id: value.producerId,
  producer_version: value.producerVersion,
  source_repository: value.sourceRepository,
  source_revision: value.sourceRevision,
  source_revision_status: value.sourceRevisionStatus,
  source_revision_reason: value.sourceRevisionReason,
});

export const provenanceFromWire = (value: unknown): PlanProducerProvenanceTs => {
  const item = record(value);
  if (item.schema_id !== PRODUCER_PROVENANCE_SCHEMA_ID ||
      item.schema_version !== PRODUCER_PROVENANCE_SCHEMA_VERSION) {
    throw new Error("provenance schema mismatch");
  }
  return makePlanProducerProvenance({
    producerId: text(item.producer_id, "producer_id"),
    producerVersion: item.producer_version as number,
    sourceRepository: text(item.source_repository, "source_repository"),
    sourceRevision: nullableText(item.source_revision, "source_revision"),
    sourceRevisionStatus: text(
      item.source_revision_status, "source_revision_status",
    ) as "exact" | "unavailable",
    sourceRevisionReason: nullableText(
      item.source_revision_reason, "source_revision_reason",
    ),
  });
};

export const provenanceSha256 = (value: PlanProducerProvenanceTs): string =>
  sha256Text(JSON.stringify(Object.fromEntries(
    Object.entries(provenanceToWire(value)).sort(([left], [right]) =>
      left.localeCompare(right)),
  )));
