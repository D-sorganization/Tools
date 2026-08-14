/** Strict metadata validation kept separate from the workspace state adapter. */

const STABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const UTC_TIMESTAMP =
  /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$/;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Parse an exact versioned payload envelope with an object data section. */
export function versionedPayload(
  value: unknown,
  schema: string,
  versions: readonly number[],
  context: string,
): { readonly version: number; readonly data: Record<string, unknown> } {
  if (
    !isRecord(value) ||
    Object.keys(value).length !== 3 ||
    !["schema", "schema_version", "data"].every((field) => field in value)
  ) {
    throw new TypeError(`${context} has invalid fields`);
  }
  if (
    value.schema !== schema ||
    typeof value.schema_version !== "number" ||
    !versions.includes(value.schema_version) ||
    !isRecord(value.data)
  ) {
    throw new TypeError(`${context} has an unsupported schema`);
  }
  return { version: value.schema_version, data: value.data };
}

/** Reject ambiguous or partially authored whole-workspace metadata. */
export function validateWorkspaceMetadata(value: unknown): void {
  if (!isRecord(value)) throw new TypeError("metadata must be an object");
  const fields = [
    "document_id",
    "title",
    "created_at_utc",
    "modified_at_utc",
    "app_version",
    "provenance",
  ];
  if (
    Object.keys(value).length !== fields.length ||
    fields.some((field) => !(field in value))
  ) {
    throw new TypeError("metadata has invalid fields");
  }
  for (const key of [
    "document_id",
    "title",
    "created_at_utc",
    "modified_at_utc",
    "app_version",
  ] as const) {
    if (typeof value[key] !== "string" || value[key].trim().length === 0) {
      throw new TypeError(`metadata.${key} must be non-empty text`);
    }
  }
  if (!STABLE_ID.test(value.document_id as string)) {
    throw new TypeError("metadata.document_id must be a stable identifier");
  }
  const createdText = value.created_at_utc as string;
  const modifiedText = value.modified_at_utc as string;
  const created = Date.parse(createdText);
  const modified = Date.parse(modifiedText);
  if (
    !UTC_TIMESTAMP.test(createdText) ||
    !UTC_TIMESTAMP.test(modifiedText) ||
    !Number.isFinite(created) ||
    !Number.isFinite(modified) ||
    modified < created
  ) {
    throw new TypeError(
      "workspace metadata timestamps must be valid strict UTC text",
    );
  }
  if (
    !isRecord(value.provenance) ||
    Object.entries(value.provenance).some(
      ([key, entry]) => key.trim().length === 0 || typeof entry !== "string",
    )
  )
    throw new TypeError(
      "metadata.provenance must map text keys to text values",
    );
}
