import { describe, expect, it } from "vitest";

import { buildTrainingManifest, canonicalTrainingManifestJson, inferPortableModel, parseCapabilityManifest,
  parsePortableModel, validateTrainingGroups } from "./neuralLabContract";
import { sha256Text } from "./launchMonitorFingerprint";

const sha = "a".repeat(64);

describe("neural lab v2 contract", () => {
  it("derives unavailable vendors and blockers from a capability manifest", () => {
    const manifest = parseCapabilityManifest({ schema: "launch-monitor-capability-manifest/v1", policy_sha256: sha, vendors: [
      { vendor_key: "trackman", rows: 11699, strict_model_input_rows: 9298,
        allowed_operations: { vendor_training: false }, approved_split_groups: [],
        current_surrogate_artifact_status: "retired_non_group_safe", training_blockers: { no_approved_repeating_split_group: 5 } },
    ] });
    expect(manifest.vendors[0]).toMatchObject({ rowCount: 11699, strictRowCount: 9298, state: "unavailable" });
  });

  it.each(["shot_id", "source_row_number", "row_index"])("rejects forbidden split %s", (split) => {
    expect(() => validateTrainingGroups([{ [split]: "a" }, { [split]: "b" }, { [split]: "c" }], split, true))
      .toThrow(/split group/i);
  });

  it("requires an approved split with three groups and a repeat", () => {
    const rows = [{ player: "a", x: 1, y: 2 }, { player: "a", x: 2, y: 3 },
      { player: "b", x: 3, y: 4 }, { player: "c", x: 4, y: 5 }];
    expect(() => validateTrainingGroups(rows, "player", false)).toThrow(/policy-approved/i);
    const request = buildTrainingManifest({ datasetId: "custom", repository: "private/repo",
      commit: "b".repeat(40), datasetPath: "data.csv", sha256: sha, rowCount: 4 }, rows,
    { vendor: "Custom", features: ["x"], targets: ["y"], splitGroup: "player", splitGroupPolicyApproved: true });
    expect(JSON.stringify(request)).not.toContain('"rows"');
    expect(request.split).toMatchObject({ distinctGroups: 3, repeatedGroups: 1 });
  });

  it("validates manifest hashes, infers consistently, and warns OOD", () => {
    const trainingManifest = { schema: "launch-monitor-neural-training/v2", dataset: { sha256: sha },
      features: ["speed"], targets: ["carry"], split: { column: "player", policy_approved: true } };
    const model = parsePortableModel({ schema: "launch-monitor-neural-bundle/v2", model_id: "test",
      vendor: "TrackMan-Comparable", training_manifest: trainingManifest,
      training_manifest_sha256: sha256Text(canonicalTrainingManifestJson(trainingManifest)), dataset_sha256: sha,
      features: [{ name: "speed", unit: "mph", mean: 100, scale: 10, min: 80, max: 120 }],
      targets: [{ name: "carry", unit: "yd", mean: 250, scale: 20 }],
      layers: [{ activation: "linear", weights: [[2]], bias: [0] }], model_card: {}, metrics: [],
      residuals: { state: "unavailable", reason: "not exported" } });
    expect(inferPortableModel(model, { speed: 130 })).toEqual({ values: { carry: 370 },
      warnings: ["speed is outside training range [80, 120] mph."] });
  });
});
