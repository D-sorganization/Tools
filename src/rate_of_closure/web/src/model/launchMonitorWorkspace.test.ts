import { describe, expect, it, vi } from "vitest";

import {
  buildPlayerCovariationRequest,
  createAnalysisExportBundle,
  parseLaunchMonitorProject,
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

  it("round trips a project without embedding private rows", () => {
    const serialized = serializeLaunchMonitorProject(project);
    expect(serialized).not.toContain('"rows"');
    expect(parseLaunchMonitorProject(serialized)).toEqual(project);
  });

  it("delegates computation to the injected authoritative backend", async () => {
    const backend = vi.fn().mockResolvedValue({ contract_version: "2.0.0", result: { ok: true } });
    await expect(runPlayerCovariation(backend, project)).resolves.toEqual({
      contract_version: "2.0.0", result: { ok: true },
    });
    expect(backend).toHaveBeenCalledWith(buildPlayerCovariationRequest(project));
  });

  it("creates a complete explicit export while the saved project stays reference-only", async () => {
    const bundle = await createAnalysisExportBundle(project, { ok: true }, [
      { shot_id: "s1", player_id: "p1", face_angle: 1, club_path: 0.5 },
    ]);
    expect(Object.keys(bundle.files).sort()).toEqual([
      "backing_rows.csv", "project.json", "result.json",
    ]);
    expect(bundle.manifest.files["backing_rows.csv"].sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(bundle.files["backing_rows.csv"]).toContain("shot_id,player_id,face_angle,club_path");
  });
});
