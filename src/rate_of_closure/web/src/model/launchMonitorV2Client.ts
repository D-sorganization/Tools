/** Canonical UpstreamDrift launch-monitor analytics v2 HTTP seam. */

export interface ResidualAvailability { state: "available" | "unavailable"; reason: string; rows?: Record<string, unknown>[] }
export interface LaunchMonitorV2Response { contractVersion: "2.0.0"; payload: Record<string, unknown>; rowAlignedResiduals: ResidualAvailability }
export interface LaunchMonitorStrokesGainedResponse { status: string; count: number; mean: number | null; payload: Record<string, unknown> }

const object = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new RangeError(`${label} must be an object`);
  return value as Record<string, unknown>;
};

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
