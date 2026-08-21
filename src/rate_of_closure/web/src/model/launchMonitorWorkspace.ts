/** Reference-only project and authoritative-backend seam for player analytics. */

import type { LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { sha256Text } from "./launchMonitorFingerprint";
import { parseCanonicalDatasetReference, type CanonicalDatasetReference } from "./launchMonitorV2Client";
import {
  createWorkspaceV3Bundle,
  parseWorkspaceV3,
  serializeWorkspaceV3,
  type WorkspaceV3,
} from "./launchMonitorWorkspaceV3";

export const LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION = "2.0.0" as const;

export interface DatasetReference {
  sourceName: string;
  repository: string;
  revision: string;
  relativePath: string;
  sha256: string;
  rowCount: number;
}

export interface LaunchMonitorProject {
  contractVersion: typeof LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION;
  name: string;
  dataset: DatasetReference;
  playerIdentity: { column: string; userAttested: boolean };
  selection: { x: string; y: string; minSamples: number; confidenceLevel: number };
  canonicalDataset?: CanonicalDatasetReference;
}

export interface PlayerCovariationRequest {
  contract_version: typeof LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION;
  operation: "player_covariation";
  dataset: {
    source_name: string; repository: string; revision: string;
    relative_path: string; sha256: string; row_count: number;
  };
  player_identity: { column: string; user_attested: true };
  variables: { x: string; y: string };
  options: { min_samples: number; confidence_level: number };
}

export type PlayerAnalyticsBackend = (
  request: PlayerCovariationRequest,
) => Promise<Record<string, unknown>>;

const requireText = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !value.trim()) throw new RangeError(`${label} must be non-empty`);
  return value;
};

function validateProject(project: LaunchMonitorProject): void {
  if (project.contractVersion !== LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION) {
    throw new RangeError(`Unsupported project contract: ${String(project.contractVersion)}`);
  }
  requireText(project.name, "Project name");
  const identity = requireText(project.playerIdentity.column, "Player identity column");
  if (!project.playerIdentity.userAttested) {
    throw new RangeError("Player identity must be explicitly user-attested");
  }
  const x = requireText(project.selection.x, "X variable");
  const y = requireText(project.selection.y, "Y variable");
  if (x === y || identity === x || identity === y) {
    throw new RangeError("Identity, X, and Y columns must be different");
  }
  const dataset = project.dataset;
  [dataset.sourceName, dataset.repository, dataset.revision, dataset.relativePath]
    .forEach((value) => requireText(value, "Dataset reference field"));
  if (!/^[a-f0-9]{64}$/i.test(dataset.sha256)) throw new RangeError("Dataset SHA-256 is invalid");
  if (!Number.isSafeInteger(dataset.rowCount) || dataset.rowCount < 0) {
    throw new RangeError("Dataset row count is invalid");
  }
  if (!Number.isSafeInteger(project.selection.minSamples) || project.selection.minSamples < 3) {
    throw new RangeError("Minimum samples must be at least three");
  }
  if (!(project.selection.confidenceLevel > 0.5 && project.selection.confidenceLevel < 1)) {
    throw new RangeError("Confidence level must be between 0.5 and 1");
  }
  if (project.canonicalDataset) parseCanonicalDatasetReference(project.canonicalDataset);
}

export function buildPlayerCovariationRequest(project: LaunchMonitorProject): PlayerCovariationRequest {
  validateProject(project);
  const { dataset, playerIdentity, selection } = project;
  return {
    contract_version: LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION,
    operation: "player_covariation",
    dataset: {
      source_name: dataset.sourceName,
      repository: dataset.repository,
      revision: dataset.revision,
      relative_path: dataset.relativePath,
      sha256: dataset.sha256,
      row_count: dataset.rowCount,
    },
    player_identity: { column: playerIdentity.column, user_attested: true },
    variables: { x: selection.x, y: selection.y },
    options: { min_samples: selection.minSamples, confidence_level: selection.confidenceLevel },
  };
}

function rowFreeResult(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(rowFreeResult);
  if (!value || typeof value !== "object") return value;
  const forbidden = new Set(["rows", "records", "backing_data", "backing_rows", "per_player"]);
  return Object.fromEntries(Object.entries(value).filter(([key, item]) =>
    !forbidden.has(key) && !(Array.isArray(item) && item.length > 0 && typeof item[0] === "object"),
  ).map(([key, item]) => [key, rowFreeResult(item)]));
}

function workspaceV3(project: LaunchMonitorProject, result?: Record<string, unknown>): WorkspaceV3 {
  validateProject(project);
  const canonical = project.canonicalDataset;
  const payload = result === undefined ? null : rowFreeResult(result);
  return parseWorkspaceV3({
    schema_id: "launch-monitor-workspace/v3",
    schema_version: 3,
    name: project.name,
    dataset: {
      source_name: project.dataset.sourceName,
      repository: project.dataset.repository,
      revision: project.dataset.revision,
      relative_path: project.dataset.relativePath,
      content_sha256: project.dataset.sha256,
      row_count: project.dataset.rowCount,
      classification: "restricted",
      authority_root_id: canonical?.root_id ?? null,
      authority_repository: canonical?.repository ?? null,
      authority_commit: canonical?.commit ?? null,
      manifest_sha256: canonical?.manifest_sha256 ?? null,
      authority_content_sha256: canonical?.content_sha256 ?? null,
      authority_row_count: canonical?.expected_row_count ?? null,
    },
    identity_evidence: { player: {
      column: project.playerIdentity.column,
      user_attested: project.playerIdentity.userAttested,
      evidence: "Dataset owner explicitly attested this player identifier.",
    } },
    analyses: [{
      analysis_id: "player-covariation",
      operation: "player_covariation",
      settings: {
        x_column: project.selection.x,
        y_column: project.selection.y,
        method: "pearson",
        minimum_samples: project.selection.minSamples,
        confidence_level: project.selection.confidenceLevel,
      },
      result: {
        status: payload === null ? "unavailable" : "available",
        authority: canonical ? "upstream-v2" : "offline-compatibility-v1",
        authority_commit: canonical?.commit ?? null,
        response_sha256: payload === null ? null : sha256Text(JSON.stringify(payload)),
        payload,
        units: { [project.selection.x]: "source-unit-unavailable", [project.selection.y]: "source-unit-unavailable" },
        formulas: ["pairwise-complete player covariation"],
        exclusions: ["Row-aligned records are retained outside the saved project."],
      },
      backing_join: {
        algorithm: "sha256-canonical-json-v1", row_count: project.dataset.rowCount,
        sha256: null, status: "available-on-authorized-export", reason: null,
      },
    }],
    export_policy: {
      persist_rows: false,
      backing_rows: "explicit-restricted-approval",
      reason: "Restricted rows remain outside saved projects and browser persistence.",
    },
  });
}

export function serializeLaunchMonitorProject(project: LaunchMonitorProject): string {
  return `${serializeWorkspaceV3(workspaceV3(project))}\n`;
}

export function parseLaunchMonitorProject(text: string): LaunchMonitorProject {
  const candidate: unknown = JSON.parse(text);
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
    throw new RangeError("Project must be a JSON object");
  }
  const wire = candidate as Record<string, unknown>;
  if (wire.schema_id === "launch-monitor-workspace/v3") {
    const saved = parseWorkspaceV3(wire);
    const dataset = saved.dataset as Record<string, unknown>;
    const identity = (saved.identity_evidence as Record<string, Record<string, unknown>>).player;
    const analysis = saved.analyses.find((item) => item.operation === "player_covariation");
    if (!analysis) throw new RangeError("Saved v3 project has no player covariation analysis");
    const settings = analysis.settings as Record<string, unknown>;
    const canonical = dataset.authority_root_id ? {
      root_id: String(dataset.authority_root_id), repository: String(dataset.authority_repository),
      commit: String(dataset.authority_commit), manifest_sha256: String(dataset.manifest_sha256),
      content_sha256: String(dataset.authority_content_sha256), expected_row_count: Number(dataset.authority_row_count),
    } : undefined;
    const project: LaunchMonitorProject = {
      contractVersion: LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION,
      name: String(saved.name),
      dataset: {
        sourceName: String(dataset.source_name), repository: String(dataset.repository),
        revision: String(dataset.revision), relativePath: String(dataset.relative_path),
        sha256: String(dataset.content_sha256), rowCount: Number(dataset.row_count),
      },
      playerIdentity: { column: String(identity.column), userAttested: identity.user_attested === true },
      selection: {
        x: String(settings.x_column), y: String(settings.y_column),
        minSamples: Number(settings.minimum_samples), confidenceLevel: Number(settings.confidence_level),
      },
      ...(canonical ? { canonicalDataset: parseCanonicalDatasetReference(canonical) } : {}),
    };
    validateProject(project);
    return project;
  }
  const project = candidate as LaunchMonitorProject & { rows?: unknown };
  if ("rows" in project) throw new RangeError("Saved projects cannot embed dataset rows");
  validateProject(project);
  return project;
}

export function parseLaunchMonitorProjectVersioned(text: string): {
  project: LaunchMonitorProject;
  importedFrom: "v3" | "v2-compatibility";
} {
  const wire: unknown = JSON.parse(text);
  const importedFrom = wire && typeof wire === "object" && !Array.isArray(wire)
    && (wire as Record<string, unknown>).schema_id === "launch-monitor-workspace/v3"
    ? "v3" : "v2-compatibility";
  return { project: parseLaunchMonitorProject(text), importedFrom };
}

export async function runPlayerCovariation(
  backend: PlayerAnalyticsBackend,
  project: LaunchMonitorProject,
): Promise<Record<string, unknown>> {
  return backend(buildPlayerCovariationRequest(project));
}

export function fingerprintLaunchMonitorRows(rows: LaunchMonitorRow[]): string {
  return sha256Text(JSON.stringify(rows));
}

export async function createAnalysisExportBundle(
  project: LaunchMonitorProject,
  result: Record<string, unknown>,
  backingRows: LaunchMonitorRow[],
) {
  return createWorkspaceV3Bundle(workspaceV3(project, result), backingRows, {
    platform: "browser", includeBackingRows: true, restrictedDataApproved: false,
  });
}
