import { describe, expect, it } from "vitest";

import {
  createPerformanceWorkspaceV3,
  loadPerformanceWorkspace,
} from "./launchMonitorPerformanceWorkspace";

const settings = { carry: "carry", lateral: "lateral", carryUnit: "yd", lateralUnit: "yd", target: 150 };

describe("performance workspace v3", () => {
  it("persists settings and aggregate results without row-shaped points", () => {
    const workspace = createPerformanceWorkspaceV3({
      sourceName: "test.csv", datasetSha256: "a".repeat(64), rowCount: 2,
      settings, results: { dispersion: { rmsYards: 4, points: [{ sourceIndex: 0 }] } },
    });
    const encoded = JSON.stringify(workspace);
    expect(workspace.schema_id).toBe("launch-monitor-workspace/v3");
    expect(encoded).not.toContain('"points"');
    expect(encoded).not.toContain('"rows"');
    expect(loadPerformanceWorkspace(encoded, "a".repeat(64))).toMatchObject({
      importedFrom: "v3", settings,
    });
  });

  it("keeps labelled v1 import compatibility and rejects a crossed fingerprint", () => {
    const legacy = JSON.stringify({ contractVersion: "launch-monitor-performance/1.0", datasetSha256: "b".repeat(64), settings });
    expect(loadPerformanceWorkspace(legacy, "b".repeat(64)).importedFrom).toBe("v1-compatibility");
    expect(() => loadPerformanceWorkspace(legacy, "c".repeat(64))).toThrow(/different dataset/i);
  });
});
