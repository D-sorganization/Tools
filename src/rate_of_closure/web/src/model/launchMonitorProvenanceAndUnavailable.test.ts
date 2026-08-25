import { describe, expect, it } from "vitest";

import goldenV3 from "./__fixtures__/launch_monitor_workspace_v3_golden.json";
import {
  createWorkspaceV3Bundle,
  parseWorkspaceV3,
  serializeWorkspaceV3,
} from "./launchMonitorWorkspaceV3";
import {
  buildDatasetJobRequest,
  parseCanonicalDatasetReference,
  validateDatasetJobPage,
  validateDatasetJobStatus,
  validateLaunchMonitorStrokesGainedResponse,
  validatePlayerCovariationResponse,
} from "./launchMonitorV2Client";

describe("launch monitor provenance and unavailable states", () => {
  it("enforces exact SHA-256 and commit lengths in canonical dataset references", () => {
    const validRef = {
      root_id: "launch-monitor-authority",
      repository: "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
      commit: "d469b8a427418fa00e99b0ad488e4310b067697d",
      manifest_sha256: "b45fd9100e6786d32dce229224ed901f02c20ef5c44962769faf6cc94700c299",
      content_sha256: "7bedf88ba473c947db2d4d078a73ee0ccd3512ffa182b751ea0a23298d1ab10c",
      expected_row_count: 261666,
    };
    const parsed = parseCanonicalDatasetReference(validRef);
    expect(parsed.expected_row_count).toBe(261666);
    expect(parsed.commit).toBe("d469b8a427418fa00e99b0ad488e4310b067697d");

    expect(() => parseCanonicalDatasetReference({ ...validRef, commit: "short" })).toThrow(/commit/i);
    expect(() => parseCanonicalDatasetReference({ ...validRef, manifest_sha256: "short" })).toThrow(/manifest_sha256/i);
  });

  it("buildDatasetJobRequest enforces minimum group rows and canonical metrics", () => {
    const validRef = parseCanonicalDatasetReference({
      root_id: "launch-monitor-authority",
      repository: "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
      commit: "d469b8a427418fa00e99b0ad488e4310b067697d",
      manifest_sha256: "b45fd9100e6786d32dce229224ed901f02c20ef5c44962769faf6cc94700c299",
      content_sha256: "7bedf88ba473c947db2d4d078a73ee0ccd3512ffa182b751ea0a23298d1ab10c",
      expected_row_count: 261666,
    });

    const job = buildDatasetJobRequest(validRef, "source_summary");
    expect(job.contract_version).toBe("launch-monitor-dataset-job/1.0.0");
    expect(job.operation.kind).toBe("source_summary");

    expect(() => buildDatasetJobRequest(validRef, "metric_summary", ["invalid_metric"])).toThrow(/canonical/i);
    expect(() => buildDatasetJobRequest(validRef, "metric_summary", ["ball_speed"], null, 5)).toThrow(/minimum_group_rows/i);
  });

  it("rejects dataset job pages containing private rows", () => {
    const validPage = {
      contract_version: "launch-monitor-dataset-job/1.0.0",
      job_id: "a".repeat(32),
      offset: 0,
      limit: 50,
      total_items: 1,
      next_offset: null,
      items: [
        {
          source_id: "source-1",
          row_count: 100,
          vendor_key: "trackman",
          redistribution_status: "restricted",
          license_spdx: "NOASSERTION",
          backing_repository: "owner/repo",
          backing_commit: "b".repeat(40),
          backing_object_digests: [],
        },
      ],
    };
    expect(validateDatasetJobPage(validPage)).toBeDefined();

    const leakyPage = {
      ...validPage,
      items: [{ ...validPage.items[0], row_index: 10 }],
    };
    expect(() => validateDatasetJobPage(leakyPage)).toThrow(/private rows/i);
  });

  it("validates workspace v3 round-trip persistence and export authorization boundaries", () => {
    const parsed = parseWorkspaceV3(goldenV3);
    const serialized = serializeWorkspaceV3(parsed);
    expect(JSON.parse(serialized)).toEqual(goldenV3);

    // Restricted browser export fails closed
    const bundle = createWorkspaceV3Bundle(parsed, [{ player_id: "p1", face_angle: 1 }], {
      includeBackingRows: true,
      restrictedDataApproved: false,
      platform: "browser",
    });
    expect(bundle.manifest.backing_data.status).toBe("unavailable");
    expect(bundle.files["backing_rows.csv"]).toBeUndefined();
  });
});
