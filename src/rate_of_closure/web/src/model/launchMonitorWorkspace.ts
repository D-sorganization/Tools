/** Reference-only project and authoritative-backend seam for player analytics. */

import type { LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { sha256Text } from "./launchMonitorFingerprint";

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

export function serializeLaunchMonitorProject(project: LaunchMonitorProject): string {
  validateProject(project);
  return `${JSON.stringify(project, null, 2)}\n`;
}

export function parseLaunchMonitorProject(text: string): LaunchMonitorProject {
  const candidate: unknown = JSON.parse(text);
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
    throw new RangeError("Project must be a JSON object");
  }
  const project = candidate as LaunchMonitorProject & { rows?: unknown };
  if ("rows" in project) throw new RangeError("Saved projects cannot embed dataset rows");
  validateProject(project);
  return project;
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

const csvCell = (value: unknown): string => {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

const rowsToCsv = (rows: LaunchMonitorRow[]): string => {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  // ⚡ Bolt Optimization: Replace chained array .map().join() with a single-pass loop
  // to eliminate intermediate array allocations and reduce GC pressure for large dataset exports
  let csv = columns.map(csvCell).join(",") + "\n";
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    let rowString = "";
    for (let j = 0; j < columns.length; j++) {
      if (j > 0) rowString += ",";
      rowString += csvCell(row[columns[j]]);
    }
    csv += rowString + "\n";
  }
  return csv;
};

export async function createAnalysisExportBundle(
  project: LaunchMonitorProject,
  result: Record<string, unknown>,
  backingRows: LaunchMonitorRow[],
) {
  const files = {
    "project.json": serializeLaunchMonitorProject(project),
    "result.json": `${JSON.stringify(result, null, 2)}\n`,
    "backing_rows.csv": rowsToCsv(backingRows),
  };
  const entries = Object.entries(files).map(([name, content]) => [
    name, { sha256: sha256Text(content), bytes: new TextEncoder().encode(content).byteLength },
  ] as const);
  return {
    files,
    manifest: {
      contractVersion: LAUNCH_MONITOR_WORKSPACE_CONTRACT_VERSION,
      purpose: "explicit full analysis export including backing rows",
      files: Object.fromEntries(entries),
    },
  };
}
