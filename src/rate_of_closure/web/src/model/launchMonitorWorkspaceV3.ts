export const WORKSPACE_V3_SCHEMA = "launch-monitor-workspace/v3" as const;

export type WorkspaceV3 = Record<string, unknown> & {
  schema_id: typeof WORKSPACE_V3_SCHEMA;
  schema_version: 3;
  dataset: Record<string, unknown> & { classification: "public" | "internal" | "restricted" };
  identity_evidence: Record<string, unknown>;
  analyses: Array<Record<string, unknown>>;
  export_policy: Record<string, unknown> & { persist_rows: false };
};

export interface WorkspaceExportAuthorization {
  platform: "browser" | "desktop";
  includeBackingRows: boolean;
  restrictedDataApproved: boolean;
}

export interface WorkspaceBundle {
  files: Record<string, string>;
  manifest: {
    schema_id: "launch-monitor-workspace-export/v3";
    backing_data: { status: "available" | "unavailable"; reason: string | null };
    files: Array<{ path: string; bytes: number; sha256: string }>;
  };
}

const rowKeys = new Set(["rows", "records", "backing_data", "source_rows"]);
const hex = (value: unknown, length: number): value is string =>
  typeof value === "string" && value.length === length && /^[0-9a-f]+$/i.test(value);

function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function containsRows(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsRows);
  if (!isRecord(value)) return false;
  return Object.entries(value).some(
    ([key, item]) => rowKeys.has(key) || (key === "backing_rows" && typeof item !== "string") || containsRows(item),
  );
}

function validateIdentity(value: unknown, name: string): void {
  requireCondition(isRecord(value), `${name} identity evidence is required`);
  requireCondition(Boolean(value.column), `${name} column is required`);
  requireCondition(value.user_attested === true, `${name} identity must be user-attested`);
  requireCondition(Boolean(value.evidence), `${name} evidence is required`);
}

function validateAnalysis(value: unknown, identities: Record<string, unknown>): void {
  requireCondition(isRecord(value), "analysis entries must be objects");
  requireCondition(Boolean(value.analysis_id), "analysis_id is required");
  requireCondition(
    value.operation === "player_covariation" || value.operation === "longitudinal" || value.operation === "performance_summary",
    "unsupported operation",
  );
  if (value.operation === "player_covariation" || value.operation === "longitudinal") validateIdentity(identities.player, "player");
  if (value.operation === "longitudinal") {
    validateIdentity(identities.session, "session");
    validateIdentity(identities.order, "order");
  }
  const result = value.result;
  requireCondition(isRecord(result), "analysis result is required");
  requireCondition(!containsRows(result.payload), "result payload must not contain rows");
  requireCondition(result.status === "available" || result.status === "unavailable", "invalid result status");
  if (result.status === "available") {
    requireCondition(result.payload !== null && result.payload !== undefined, "available result payload is required");
    requireCondition(hex(result.response_sha256, 64), "response SHA-256 is required");
  } else {
    requireCondition(result.payload === null, "unavailable result payload must be null");
    requireCondition(result.response_sha256 === null, "unavailable result hash must be null");
    requireCondition(Array.isArray(result.exclusions) && result.exclusions.length > 0, "unavailable result exclusions are required");
  }
}

export function parseWorkspaceV3(value: string | unknown): WorkspaceV3 {
  const document: unknown = typeof value === "string" ? JSON.parse(value) : structuredClone(value);
  requireCondition(isRecord(document), "workspace project must be an object");
  requireCondition(document.schema_id === WORKSPACE_V3_SCHEMA, "unsupported workspace schema");
  requireCondition(document.schema_version === 3, "unsupported workspace version");
  requireCondition(!containsRows(document), "workspace projects must not contain rows");
  const allowed = new Set(["schema_id", "schema_version", "name", "dataset", "identity_evidence", "analyses", "export_policy"]);
  requireCondition(Object.keys(document).every((key) => allowed.has(key)), "unknown workspace fields");
  requireCondition(isRecord(document.dataset), "dataset metadata is required");
  requireCondition(["public", "internal", "restricted"].includes(String(document.dataset.classification)), "invalid data classification");
  requireCondition(hex(document.dataset.content_sha256, 64), "dataset SHA-256 is required");
  if (document.dataset.authority_commit !== null && document.dataset.authority_commit !== undefined) {
    requireCondition(hex(document.dataset.authority_commit, 40), "authority commit must be a full SHA");
  }
  if (document.dataset.manifest_sha256 !== null && document.dataset.manifest_sha256 !== undefined) {
    requireCondition(hex(document.dataset.manifest_sha256, 64), "manifest SHA-256 is required");
  }
  requireCondition(isRecord(document.identity_evidence), "identity evidence is required");
  requireCondition(Array.isArray(document.analyses) && document.analyses.length > 0, "at least one analysis is required");
  document.analyses.forEach((analysis) => validateAnalysis(analysis, document.identity_evidence as Record<string, unknown>));
  requireCondition(isRecord(document.export_policy), "export policy is required");
  requireCondition(document.export_policy.persist_rows === false, "saved projects must be row-free");
  return document as WorkspaceV3;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && !Number.isFinite(value)) throw new Error("canonical JSON rejects non-finite numbers");
  return JSON.stringify(value);
}

export function serializeWorkspaceV3(project: WorkspaceV3 | unknown): string {
  return canonical(parseWorkspaceV3(project));
}

const SHA256_CONSTANTS = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
] as const;

function rotateRight(value: number, amount: number): number {
  return (value >>> amount) | (value << (32 - amount));
}

function sha256(value: string): string {
  const input = Array.from(new TextEncoder().encode(value));
  const bitLength = input.length * 8;
  input.push(0x80);
  while (input.length % 64 !== 56) input.push(0);
  for (let shift = 56; shift >= 0; shift -= 8) input.push(Math.floor(bitLength / 2 ** shift) & 0xff);
  const hash = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  for (let offset = 0; offset < input.length; offset += 64) {
    const words = new Array<number>(64).fill(0);
    for (let index = 0; index < 16; index += 1) {
      const start = offset + index * 4;
      words[index] = (input[start] << 24) | (input[start + 1] << 16) | (input[start + 2] << 8) | input[start + 3];
    }
    for (let index = 16; index < 64; index += 1) {
      const x = words[index - 15];
      const y = words[index - 2];
      const small0 = rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >>> 3);
      const small1 = rotateRight(y, 17) ^ rotateRight(y, 19) ^ (y >>> 10);
      words[index] = (words[index - 16] + small0 + words[index - 7] + small1) | 0;
    }
    const state = [...hash];
    for (let index = 0; index < 64; index += 1) {
      const big1 = rotateRight(state[4], 6) ^ rotateRight(state[4], 11) ^ rotateRight(state[4], 25);
      const choice = (state[4] & state[5]) ^ (~state[4] & state[6]);
      const temporary1 = (state[7] + big1 + choice + SHA256_CONSTANTS[index] + words[index]) | 0;
      const big0 = rotateRight(state[0], 2) ^ rotateRight(state[0], 13) ^ rotateRight(state[0], 22);
      const majority = (state[0] & state[1]) ^ (state[0] & state[2]) ^ (state[1] & state[2]);
      state.unshift((temporary1 + big0 + majority) | 0);
      state[4] = (state[4] + temporary1) | 0;
      state.pop();
    }
    hash.forEach((word, index) => { hash[index] = (word + state[index]) | 0; });
  }
  return hash.map((word) => (word >>> 0).toString(16).padStart(8, "0")).join("");
}

function csv(rows: Array<Record<string, unknown>>, fields: string[]): string {
  const escape = (value: unknown) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return `${fields.join(",")}\n${rows.map((row) => fields.map((field) => escape(row[field])).join(",")).join("\n")}\n`;
}

function permission(project: WorkspaceV3, authorization: WorkspaceExportAuthorization): [boolean, string | null] {
  if (!authorization.includeBackingRows) return [false, "backing rows were not requested"];
  if (authorization.platform === "browser" && project.dataset.classification === "restricted") {
    return [false, "browser export of restricted backing rows is unavailable"];
  }
  if (project.dataset.classification === "restricted" && !authorization.restrictedDataApproved) {
    return [false, "restricted backing-row export requires explicit restricted approval"];
  }
  return [true, null];
}

export function createWorkspaceV3Bundle(
  projectValue: WorkspaceV3 | unknown,
  backingRows: Array<Record<string, unknown>>,
  authorization: WorkspaceExportAuthorization,
): WorkspaceBundle {
  const project = parseWorkspaceV3(projectValue);
  const files: Record<string, string> = {
    "project.json": `${serializeWorkspaceV3(project)}\n`,
    "results.json": `${canonical(project.analyses)}\n`,
  };
  const [allowed, reason] = permission(project, authorization);
  if (allowed && backingRows.length > 0) {
    const rowHashes = backingRows.map((row) => sha256(canonical(row)));
    files["backing_join.csv"] = csv(
      rowHashes.map((row_sha256, result_row_index) => ({ result_row_index, row_sha256 })),
      ["result_row_index", "row_sha256"],
    );
    const fields = Array.from(new Set(backingRows.flatMap((row) => Object.keys(row))));
    files["backing_rows.csv"] = csv(backingRows, fields);
  }
  const manifestFiles = Object.entries(files).map(([path, content]) => ({
    path,
    bytes: new TextEncoder().encode(content).byteLength,
    sha256: sha256(content),
  }));
  return {
    files,
    manifest: {
      schema_id: "launch-monitor-workspace-export/v3",
      backing_data: { status: allowed && backingRows.length > 0 ? "available" : "unavailable", reason },
      files: manifestFiles,
    },
  };
}
