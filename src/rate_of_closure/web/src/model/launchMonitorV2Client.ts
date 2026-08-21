/** Canonical UpstreamDrift launch-monitor analytics v2 HTTP seam. */

export interface ResidualAvailability { state: "available" | "unavailable"; reason: string; rows?: Record<string, unknown>[] }
export interface LaunchMonitorV2Response { contractVersion: "2.0.0"; payload: Record<string, unknown>; rowAlignedResiduals: ResidualAvailability }
export interface LaunchMonitorStrokesGainedResponse { status: string; count: number; mean: number | null; payload: Record<string, unknown> }
export const DATASET_JOB_CONTRACT_VERSION = "launch-monitor-dataset-job/1.0.0" as const;
export const PLAYER_COVARIATION_CONTRACT_VERSION = "launch-monitor-player-covariation/1.0.0" as const;
export const MAX_CANONICAL_INLINE_RECORDS = 20_000;
export const MAX_DATASET_JOB_PAGE_SIZE = 200;
export const CANONICAL_DATASET_METRICS = new Set([
  "club_speed", "ball_speed", "smash_factor", "launch_angle", "launch_direction",
  "spin_rate", "back_spin", "side_spin", "spin_axis", "attack_angle", "club_path",
  "face_angle", "carry_distance", "total_distance", "descent_angle", "lateral_carry",
  "flight_time",
]);

export interface CanonicalDatasetReference {
  root_id: string; repository: string; commit: string; manifest_sha256: string;
  content_sha256: string; expected_row_count: number;
}

export interface CovariationOptions {
  playerColumn: string; xColumn: string; yColumn: string;
  minSamples: number; confidenceLevel: number;
}

const object = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new RangeError(`${label} must be an object`);
  return value as Record<string, unknown>;
};

const matchedText = (value: unknown, pattern: RegExp, label: string): string => {
  if (typeof value !== "string" || !pattern.test(value)) throw new RangeError(`${label} is invalid`);
  return value;
};

export function parseCanonicalDatasetReference(value: unknown): CanonicalDatasetReference {
  const root = object(value, "dataset reference");
  const required = ["root_id", "repository", "commit", "manifest_sha256", "content_sha256", "expected_row_count"];
  if (Object.keys(root).length !== required.length || required.some((key) => !(key in root))) {
    throw new RangeError("dataset reference has missing or unknown fields");
  }
  const expected = root.expected_row_count;
  if (!Number.isSafeInteger(expected) || !(Number(expected) >= 1 && Number(expected) <= 10_000_000)) {
    throw new RangeError("expected_row_count is outside the canonical bounds");
  }
  return {
    root_id: matchedText(root.root_id, /^[a-z][a-z0-9-]{0,62}$/, "root_id"),
    repository: matchedText(root.repository, /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/, "repository"),
    commit: matchedText(root.commit, /^[0-9a-f]{40}$/, "commit"),
    manifest_sha256: matchedText(root.manifest_sha256, /^[0-9a-f]{64}$/, "manifest_sha256"),
    content_sha256: matchedText(root.content_sha256, /^[0-9a-f]{64}$/, "content_sha256"),
    expected_row_count: expected as number,
  };
}

export function buildDatasetJobRequest(
  reference: CanonicalDatasetReference,
  kind: "source_summary" | "metric_summary" | "correlation",
  metrics: string[] = [],
  groupBy: "source_id" | "monitor" | "club" | null = null,
  minimumGroupRows = 10,
) {
  parseCanonicalDatasetReference(reference);
  if (minimumGroupRows < 10) throw new RangeError("minimum_group_rows must be at least 10");
  if (metrics.length > 12 || new Set(metrics).size !== metrics.length || metrics.some((metric) => !CANONICAL_DATASET_METRICS.has(metric))) {
    throw new RangeError("metrics must be unique canonical dataset metrics");
  }
  if (kind === "source_summary" && (metrics.length || groupBy !== null)) throw new RangeError("source_summary does not accept options");
  if (kind === "metric_summary" && !metrics.length) throw new RangeError("metric_summary requires metrics");
  if (kind === "correlation" && metrics.length < 2) throw new RangeError("correlation requires two metrics");
  return { contract_version: DATASET_JOB_CONTRACT_VERSION, dataset: reference,
    operation: { kind, metrics, group_by: groupBy, minimum_group_rows: minimumGroupRows } };
}

export function validateDatasetJobStatus(value: unknown): Record<string, unknown> {
  const root = object(value, "dataset job status");
  const keys = ["contract_version", "job_id", "status", "submitted_at_utc", "completed_at_utc", "input_row_count", "result_item_count", "unavailable"];
  if (Object.keys(root).length !== keys.length || keys.some((key) => !(key in root))) throw new RangeError("dataset job status has missing or unknown fields");
  if (root.contract_version !== DATASET_JOB_CONTRACT_VERSION) throw new RangeError("Unsupported dataset job contract");
  matchedText(root.job_id, /^[0-9a-f]{32}$/, "job_id");
  if (!["queued", "running", "completed", "unavailable", "failed"].includes(String(root.status))) throw new RangeError("dataset job status is invalid");
  for (const key of ["input_row_count", "result_item_count"]) {
    if (!Number.isSafeInteger(root[key]) || Number(root[key]) < 0) throw new RangeError(`dataset job ${key} is invalid`);
  }
  return root;
}

export function validateDatasetJobPage(value: unknown): Record<string, unknown> {
  const root = object(value, "dataset job page");
  const keys = ["contract_version", "job_id", "offset", "limit", "total_items", "next_offset", "items"];
  if (Object.keys(root).length !== keys.length || keys.some((key) => !(key in root))) throw new RangeError("dataset job page has missing or unknown fields");
  if (root.contract_version !== DATASET_JOB_CONTRACT_VERSION) throw new RangeError("Unsupported dataset job contract");
  matchedText(root.job_id, /^[0-9a-f]{32}$/, "job_id");
  for (const key of ["offset", "total_items"]) {
    if (!Number.isSafeInteger(root[key]) || Number(root[key]) < 0) throw new RangeError(`dataset job page ${key} is invalid`);
  }
  if (root.next_offset !== null && (!Number.isSafeInteger(root.next_offset) || Number(root.next_offset) < 0)) throw new RangeError("dataset job page next_offset is invalid");
  if (!Number.isSafeInteger(root.limit) || Number(root.limit) < 1 || Number(root.limit) > MAX_DATASET_JOB_PAGE_SIZE) throw new RangeError("dataset job page limit is invalid");
  if (!Array.isArray(root.items) || root.items.length > Number(root.limit)) throw new RangeError("dataset job page items are invalid");
  const privateKeys = new Set(["shot_id", "source_row", "row_index"]);
  const aggregateKeysets = [
    ["source_id", "row_count", "vendor_key", "redistribution_status", "license_spdx", "backing_repository", "backing_commit", "backing_object_digests"],
    ["group_by", "group", "metric", "n", "mean", "standard_deviation", "minimum", "maximum"],
    ["group_by", "group", "left_metric", "right_metric", "n", "correlation"],
  ].map((keys) => [...keys].sort().join("\u001f"));
  for (const item of root.items) {
    const aggregate = object(item, "dataset job aggregate");
    if (Object.keys(aggregate).some((key) => privateKeys.has(key))) throw new RangeError("dataset job pages cannot expose private rows");
    if (!aggregateKeysets.includes(Object.keys(aggregate).sort().join("\u001f"))) throw new RangeError("dataset job page item does not match an aggregate schema");
  }
  return root;
}

export function buildPlayerCovariationPayload(records: Record<string, unknown>[], options: CovariationOptions) {
  if (records.length < 1 || records.length > MAX_CANONICAL_INLINE_RECORDS) throw new RangeError("canonical player covariation accepts at most 20,000 rows");
  const columns = [options.playerColumn, options.xColumn, options.yColumn];
  if (columns.some((value) => !value.trim()) || new Set(columns).size !== 3) throw new RangeError("player, x, and y columns must be distinct and non-empty");
  if (!Number.isSafeInteger(options.minSamples) || options.minSamples < 4 || !(options.confidenceLevel > 0.5 && options.confidenceLevel < 1)) throw new RangeError("canonical covariation options are invalid");
  return { records, request: { x_column: options.xColumn, y_column: options.yColumn,
    player_column: options.playerColumn, min_samples: options.minSamples,
    confidence_level: options.confidenceLevel }, context: { player_identity: {
      trust_level: "explicit_user_attested", identifier_column: options.playerColumn,
      evidence: `Dataset owner attested ${options.playerColumn} in this client session.`,
    } } };
}

export function validatePlayerCovariationResponse(value: unknown): Record<string, unknown> {
  const root = object(value, "player covariation response");
  if (root.contract_version !== PLAYER_COVARIATION_CONTRACT_VERSION) throw new RangeError("Unsupported player covariation contract");
  const common = ["analysis_kind", "contract_version", "status", "request", "lineage", "player_identity", "vendor_provenance", "claims", "warnings"];
  const selected = [...common, "pooled", "within_player", "between_player", "per_player", "meta_analysis", "missingness", "units", "availability", "uncertainty", "definitions"];
  const scan = [...common, "pair_count", "available_pair_count", "unavailable_pair_count", "ranking", "method_description"];
  const expected = root.analysis_kind === "selected_pair" ? selected : scan;
  if (Object.keys(root).length !== expected.length || expected.some((key) => !(key in root))) throw new RangeError("player covariation response is missing required fields");
  if (!["selected_pair", "pair_scan"].includes(String(root.analysis_kind))) throw new RangeError("player covariation analysis kind is invalid");
  if (!Array.isArray(object(root.lineage, "lineage").backing_records)) throw new RangeError("player covariation backing lineage is invalid");
  const trust = object(root.player_identity, "player identity").trust_level;
  if (!["explicit_user_attested", "pseudonymous_stable", "verified_external"].includes(String(trust))) throw new RangeError("player covariation requires trusted identity evidence");
  const claims = object(root.claims, "claims");
  if (claims.device_emulation !== false || claims.device_certification !== false || claims.causal_inference !== false) throw new RangeError("player covariation response makes an unsupported claim");
  return root;
}

export function validateLaunchMonitorV2Response(value: unknown): LaunchMonitorV2Response {
  const root = object(value, "Upstream v2 response");
  if (root.contract_version !== "2.0.0") throw new RangeError("Unsupported Upstream contract version");
  const required = ["status", "analysis", "units", "lineage", "missingness", "availability", "uncertainty", "player_identity", "vendor_provenance", "claims", "warnings"];
  if (required.some((key) => !(key in root))) throw new RangeError("Upstream v2 response is missing required fields");
  const claims = object(root.claims, "claims");
  if (claims.device_emulation !== false || claims.device_certification !== false) throw new RangeError("Unsupported device emulation or certification claim");
  const lineage = object(root.lineage, "lineage");
  if (!Array.isArray(lineage.backing_records)) throw new RangeError("Backing-record lineage is invalid");
  const analysis = root.analysis && typeof root.analysis === "object" && !Array.isArray(root.analysis) ? root.analysis as Record<string, unknown> : {};
  const residualRows = analysis.row_aligned_residuals;
  const residuals: ResidualAvailability = Array.isArray(residualRows) && residualRows.length === lineage.backing_records.length
    ? { state: "available", reason: "v2 row-aligned residuals match backing records", rows: residualRows.map((row) => object(row, "residual row")) }
    : { state: "unavailable", reason: "The canonical v2 response does not provide row-aligned residuals matching backing records." };
  return { contractVersion: "2.0.0", payload: root, rowAlignedResiduals: residuals };
}

export function validateLaunchMonitorStrokesGainedResponse(value: unknown): LaunchMonitorStrokesGainedResponse {
  const root = object(value, "Upstream scoring response");
  if (root.contract_version !== "launch-monitor-strokes-gained-analysis/1.0.0") throw new RangeError("Unsupported Upstream scoring contract");
  const required = ["status", "metric_name", "unit", "value_summary", "baseline", "formula", "units", "availability", "uncertainty", "row_results", "excluded_rows", "exclusions", "group_summaries", "longitudinal_summaries", "analysis_context", "dataset_fingerprint_sha256", "claims", "warnings", "limitations"];
  if (required.some((key) => !(key in root))) throw new RangeError("Upstream scoring response is missing required fields");
  if (root.metric_name !== "source_backed_strokes_gained") throw new RangeError("Upstream scoring metric is invalid");
  const claims = object(root.claims, "scoring claims");
  if (claims.is_strokes_gained !== true || claims.source_backed !== true) throw new RangeError("Upstream scoring response is not source-backed strokes gained");
  if (claims.device_emulation !== false || claims.device_certification !== false || claims.causal_inference !== false) throw new RangeError("Upstream scoring response makes an unsupported claim");
  const summary = object(root.value_summary, "value summary");
  if (!Number.isInteger(summary.count) || (summary.mean !== null && typeof summary.mean !== "number")) throw new RangeError("Upstream scoring summary is invalid");
  return { status: String(root.status), count: summary.count as number, mean: summary.mean as number | null, payload: root };
}

export function createLaunchMonitorV2Client(baseUrl: string) {
  const root = baseUrl.replace(/\/$/, "");
  return async (payload: Record<string, unknown>): Promise<LaunchMonitorV2Response> => {
    const response = await fetch(`${root}/tools/launch-monitor-analytics/v2/analyze`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(`Upstream v2 analysis failed (${response.status})`);
    return validateLaunchMonitorV2Response(await response.json());
  };
}

export function createLaunchMonitorStrokesGainedClient(baseUrl: string) {
  const root = baseUrl.replace(/\/$/, "");
  return async (payload: Record<string, unknown>): Promise<LaunchMonitorStrokesGainedResponse> => {
    const response = await fetch(`${root}/tools/launch-monitor-analytics/v2/strokes-gained`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(`Upstream strokes-gained analysis failed (${response.status})`);
    return validateLaunchMonitorStrokesGainedResponse(await response.json());
  };
}

const requestJson = async (url: string, init?: RequestInit): Promise<unknown> => {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`Upstream launch-monitor request failed (${response.status})`);
  return response.json();
};

export function createCanonicalLaunchMonitorClient(baseUrl: string) {
  let authority: URL;
  try { authority = new URL(baseUrl); } catch { throw new RangeError("Canonical authority must be an HTTP(S) URL"); }
  if (!["http:", "https:"].includes(authority.protocol) || !authority.host) throw new RangeError("Canonical authority must be an HTTP(S) URL");
  const root = baseUrl.replace(/\/$/, "");
  const endpoint = `${root}/tools/launch-monitor-analytics/v2`;
  return {
    submitDatasetJob: async (payload: Record<string, unknown>) => validateDatasetJobStatus(await requestJson(`${endpoint}/dataset-jobs`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    })),
    datasetJobStatus: async (jobId: string) => {
      matchedText(jobId, /^[0-9a-f]{32}$/, "job_id");
      return validateDatasetJobStatus(await requestJson(`${endpoint}/dataset-jobs/${jobId}`));
    },
    datasetJobResults: async (jobId: string, offset = 0, limit = 100) => {
      matchedText(jobId, /^[0-9a-f]{32}$/, "job_id");
      return validateDatasetJobPage(await requestJson(`${endpoint}/dataset-jobs/${jobId}/results?offset=${offset}&limit=${limit}`));
    },
    playerCovariation: async (payload: Record<string, unknown>) => validatePlayerCovariationResponse(await requestJson(`${endpoint}/player-covariation`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    })),
  };
}
