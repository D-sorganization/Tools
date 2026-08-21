import { describe, expect, it } from "vitest";

import golden from "./__fixtures__/launch_monitor_workspace_v3_golden.json";
import {
  createWorkspaceV3Bundle,
  parseWorkspaceV3,
  serializeWorkspaceV3,
} from "./launchMonitorWorkspaceV3";

describe("launch monitor workspace v3", () => {
  it("round trips the shared row-free golden", () => {
    const project = parseWorkspaceV3(golden);
    const serialized = serializeWorkspaceV3(project);
    expect(JSON.parse(serialized)).toEqual(golden);
    expect(serialized).not.toContain('"rows"');
  });

  it("rejects embedded rows and unattested identity", () => {
    expect(() => parseWorkspaceV3({ ...golden, rows: [{ player_id: "p1" }] }))
      .toThrow(/unknown|rows/i);
    expect(() => parseWorkspaceV3({
      ...golden,
      identity_evidence: {
        ...golden.identity_evidence,
        player: { ...golden.identity_evidence.player, user_attested: false },
      },
    })).toThrow(/attested/i);
  });

  it("fails closed for restricted browser backing exports", () => {
    const bundle = createWorkspaceV3Bundle(parseWorkspaceV3(golden), [
      { player_id: "p1", face_angle: 1 },
    ], { includeBackingRows: true, restrictedDataApproved: true, platform: "browser" });
    expect(bundle.files["backing_rows.csv"]).toBeUndefined();
    expect(bundle.files["backing_join.csv"]).toBeUndefined();
    expect(bundle.manifest.backing_data.status).toBe("unavailable");
    expect(bundle.manifest.backing_data.reason).toMatch(/browser.*restricted/i);
  });

  it("creates deterministic joins for explicitly approved desktop exports", () => {
    const rows = [{ player_id: "p1", face_angle: 1 }, { player_id: "p1", face_angle: 2 }];
    const authorization = {
      includeBackingRows: true, restrictedDataApproved: true, platform: "desktop" as const,
    };
    const first = createWorkspaceV3Bundle(parseWorkspaceV3(golden), rows, authorization);
    const second = createWorkspaceV3Bundle(parseWorkspaceV3(golden), rows, authorization);
    const expectedRowHash = [
      "44bcce6d01c15405", // pragma: allowlist secret
      "60681daf1db90869", // pragma: allowlist secret
      "e9d4b15cd59d43bc", // pragma: allowlist secret
      "f99ab4ea694ccf1d", // pragma: allowlist secret
    ].join("");
    expect(first.files["backing_join.csv"]).toBe(second.files["backing_join.csv"]);
    expect(first.files["backing_join.csv"]).toContain(expectedRowHash);
    expect(first.files["backing_rows.csv"]).toContain("player_id,face_angle");
  });
});
