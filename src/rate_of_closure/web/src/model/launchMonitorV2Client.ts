/** Canonical UpstreamDrift launch-monitor analytics v2 HTTP seam. */

export interface ResidualAvailability { state: "available" | "unavailable"; reason: string; rows?: Record<string, unknown>[] }
export interface LaunchMonitorV2Response { contractVersion: "2.0.0"; payload: Record<string, unknown>; rowAlignedResiduals: ResidualAvailability }

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

export function createLaunchMonitorV2Client(baseUrl: string) {
  const root = baseUrl.replace(/\/$/, "");
  return async (payload: Record<string, unknown>): Promise<LaunchMonitorV2Response> => {
    const response = await fetch(`${root}/tools/launch-monitor-analytics/v2/analyze`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(`Upstream v2 analysis failed (${response.status})`);
    return validateLaunchMonitorV2Response(await response.json());
  };
}
