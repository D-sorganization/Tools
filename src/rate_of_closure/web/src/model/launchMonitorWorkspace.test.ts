import { describe, expect, it, vi } from "vitest";

import {
  buildPlayerCovariationRequest,
  createAnalysisExportBundle,
  parseLaunchMonitorProject,
  parseLaunchMonitorProjectVersioned,
  runPlayerCovariation,
  serializeLaunchMonitorProject,
  type LaunchMonitorProject,
} from "./launchMonitorWorkspace";

const project: LaunchMonitorProject = {
  contractVersion: "2.0.0",
  name: "Face and path",
  dataset: {
    sourceName: "private-corpus",
    repository: "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
    revision: "97f3ecf",
    relativePath: "data/authority/database/shot_corpus_parquet",
    sha256: "a".repeat(64),
    rowCount: 261666,
  },
  playerIdentity: { column: "player_id", userAttested: true },
  selection: { x: "face_angle", y: "club_path", minSamples: 10, confidenceLevel: 0.95 },
};

describe("launch monitor workspace v2", () => {
  it("requires explicit user-attested identity and never infers it", () => {
    expect(() => buildPlayerCovariationRequest({
      ...project,
      playerIdentity: { column: "player_id", userAttested: false },
    })).toThrow(/attest/i);
    expect(() => buildPlayerCovariationRequest({
      ...project,
      playerIdentity: { column: "", userAttested: true },
    })).toThrow(/column/i);
  });

  it("builds a reference-only backend request", () => {
    const request = buildPlayerCovariationRequest(project);
    expect(request).toMatchObject({
      contract_version: "2.0.0",
      operation: "player_covariation",
      player_identity: { column: "player_id", user_attested: true },
      variables: { x: "face_angle", y: "club_path" },
    });
    expect(request).not.toHaveProperty("records");
  });

  it("writes v3 and imports it without embedding private rows", () => {
    const serialized = serializeLaunchMonitorProject(project);
    expect(serialized).not.toContain('"rows"');
    expect(JSON.parse(serialized).schema_id).toBe("launch-monitor-workspace/v3");
    expect(parseLaunchMonitorProject(serialized)).toEqual(project);
  });

  it("persists an authorized corpus reference without a private path", () => {
    const canonical = { ...project, canonicalDataset: {
      root_id: "launch-monitor-authority",
      repository: "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
      commit: "7".repeat(40), manifest_sha256: "8".repeat(64),
      content_sha256: "9".repeat(64), expected_row_count: 261666,
    } };
    const serialized = serializeLaunchMonitorProject(canonical);
    expect(parseLaunchMonitorProject(serialized)).toEqual(canonical);
    expect(serialized).not.toContain("privatePath");
  });

  it("delegates computation to the injected authoritative backend", async () => {
    const backend = vi.fn().mockResolvedValue({ contract_version: "2.0.0", result: { ok: true } });
    await expect(runPlayerCovariation(backend, project)).resolves.toEqual({
      contract_version: "2.0.0", result: { ok: true },
    });
    expect(backend).toHaveBeenCalledWith(buildPlayerCovariationRequest(project));
  });

  it("fails closed for restricted backing rows in browser exports", async () => {
    const bundle = await createAnalysisExportBundle(project, { ok: true }, [
      { shot_id: "s1", player_id: "p1", face_angle: 1, club_path: 0.5 },
    ]);
    expect(bundle.files["backing_rows.csv"]).toBeUndefined();
    expect(bundle.files["backing_join.csv"]).toBeUndefined();
    expect(bundle.manifest.backing_data.status).toBe("unavailable");
    expect(bundle.manifest.backing_data.reason).toMatch(/browser.*restricted/i);
  });

  it("keeps labelled v2 import compatibility", () => {
    const imported = parseLaunchMonitorProjectVersioned(`${JSON.stringify(project)}\n`);
    expect(imported.project).toEqual(project);
    expect(imported.importedFrom).toBe("v2-compatibility");
  });
});
