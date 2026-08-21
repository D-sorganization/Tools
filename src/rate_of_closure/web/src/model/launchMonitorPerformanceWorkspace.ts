/** Row-free v3 persistence adapter for descriptive performance analytics. */

import { sha256Text } from "./launchMonitorFingerprint";
import { parseWorkspaceV3, type WorkspaceV3 } from "./launchMonitorWorkspaceV3";

interface PerformanceWorkspaceInput {
  sourceName: string;
  datasetSha256: string;
  rowCount: number;
  settings: Record<string, unknown>;
  results: Record<string, unknown>;
}

interface LoadedPerformanceWorkspace {
  importedFrom: "v3" | "v1-compatibility";
  settings: Record<string, unknown>;
  results: Record<string, unknown>;
}

function aggregateOnly(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(aggregateOnly);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).filter(([key]) =>
    !["points", "values", "rows", "records", "backing_data", "backing_rows"].includes(key),
  ).map(([key, item]) => [key, aggregateOnly(item)]));
}

export function createPerformanceWorkspaceV3(input: PerformanceWorkspaceInput): WorkspaceV3 {
  const payload = aggregateOnly(input.results) as Record<string, unknown>;
  const available = Object.values(payload).some((value) => value !== null);
  return parseWorkspaceV3({
    schema_id: "launch-monitor-workspace/v3", schema_version: 3,
    name: `${input.sourceName} performance analysis`,
    dataset: {
      source_name: input.sourceName, repository: "local-user-data", revision: "unversioned",
      relative_path: input.sourceName, content_sha256: input.datasetSha256,
      row_count: input.rowCount, classification: "restricted",
      authority_commit: null, manifest_sha256: null,
    },
    identity_evidence: {},
    analyses: [{
      analysis_id: "performance-summary", operation: "performance_summary", settings: input.settings,
      result: {
        status: available ? "available" : "unavailable", authority: "offline-compatibility-v1",
        authority_commit: null, response_sha256: available ? sha256Text(JSON.stringify(payload)) : null,
        payload: available ? payload : null,
        units: { carry: "yd", lateral: "yd", target_error: "yd" },
        formulas: ["directional dispersion and session aggregates; see result fields"],
        exclusions: ["Per-shot and per-session points remain outside the saved project."],
      },
      backing_join: {
        algorithm: "sha256-canonical-json-v1", row_count: input.rowCount, sha256: null,
        status: "available-on-authorized-export", reason: null,
      },
    }],
    export_policy: {
      persist_rows: false, backing_rows: "explicit-restricted-approval",
      reason: "Restricted rows remain outside saved projects and browser persistence.",
    },
  });
}

export function loadPerformanceWorkspace(text: string, expectedSha256: string): LoadedPerformanceWorkspace {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new RangeError("Saved analysis must be an object");
  const wire = value as Record<string, unknown>;
  if (wire.schema_id === "launch-monitor-workspace/v3") {
    const workspace = parseWorkspaceV3(wire);
    if (workspace.dataset.content_sha256 !== expectedSha256) throw new RangeError("Saved analysis references a different dataset");
    const analysis = workspace.analyses.find((item) => item.operation === "performance_summary");
    if (!analysis) throw new RangeError("Saved v3 analysis has no performance summary");
    const result = analysis.result as Record<string, unknown>;
    return {
      importedFrom: "v3", settings: analysis.settings as Record<string, unknown>,
      results: (result.payload as Record<string, unknown> | null) ?? {},
    };
  }
  if (wire.datasetSha256 !== expectedSha256) throw new RangeError("Saved analysis references a different dataset");
  if (!wire.settings || typeof wire.settings !== "object") throw new RangeError("Saved analysis settings are unavailable");
  return {
    importedFrom: "v1-compatibility", settings: wire.settings as Record<string, unknown>,
    results: {
      dispersion: wire.dispersion ?? null, proxy: wire.proxy ?? null,
      strokes: wire.strokes ?? null, trend: wire.trend ?? null,
    },
  };
}
