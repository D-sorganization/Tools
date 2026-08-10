import { describe, expect, it } from "vitest";

import resultFixture from "./__fixtures__/ground_reference_pipeline_golden_v1.json";
import workspaceFixture from "./__fixtures__/ground_playback_workspace_golden_v2.json?raw";
import {
  GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  groundWorkspaceToJson,
  type GroundPlaybackWorkspace,
} from "./groundPlaybackWorkspace";
import {
  GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
  GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
  groundWorkspaceFromVersionedJson,
  loadGroundWorkspaceVersionedJson,
  groundWorkspaceV2FromJson,
  groundWorkspaceV2ToJson,
  type GroundPlaybackWorkspaceV2,
} from "./groundPlaybackWorkspaceV2";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";

const WORKSPACE_SHA256 =
  "28b94af16a05315a9d1067bda894a3817dce5849a9562e1ffef7d0d8caecd654";

const comparisonResult = () => {
  const payload = structuredClone(resultFixture.result);
  payload.request_id = "comparison-run";
  payload.provenance.input_sha256 = "b".repeat(64);
  payload.trajectory.forEach((point) => {
    point.time_s += 0.2;
  });
  payload.events.forEach((event) => {
    event.time_s += 0.2;
  });
  payload.termination.time_s += 0.2;
  return parseFlightToGroundResultRecord(payload);
};

const workspaceV2 = (): GroundPlaybackWorkspaceV2 => ({
  schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
  result: parseFlightToGroundResultRecord(resultFixture.result),
  comparison: { result: comparisonResult(), visible: false },
  playback: { timeS: 1.60466094435, speed: 2, loop: true },
  view: { yawDeg: -37.5, pitchDeg: 18, zoom: 1.75 },
});

const sha256 = async (text: string): Promise<string> => {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
};

describe("ground playback workspace v2", () => {
  it("round trips a hidden comparison at union time", () => {
    const encoded = groundWorkspaceV2ToJson(workspaceV2());
    const restored = groundWorkspaceV2FromJson(encoded);

    expect(restored).toEqual(workspaceV2());
    expect(groundWorkspaceV2ToJson(restored)).toBe(encoded);
    expect(JSON.parse(encoded)).toMatchObject({
      schema_version: GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
      comparison: { visible: false },
      playback: { time_s: 1.60466094435 },
    });
  });

  it("migrates v1 one way while leaving the v1 API unchanged", () => {
    const legacy: GroundPlaybackWorkspace = {
      schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA,
      result: parseFlightToGroundResultRecord(resultFixture.result),
      playback: { timeS: 1.205, speed: 1, loop: false },
      view: { yawDeg: 0, pitchDeg: 22, zoom: 1 },
    };
    const legacyText = groundWorkspaceToJson(legacy);

    const migrated = groundWorkspaceFromVersionedJson(legacyText);
    const load = loadGroundWorkspaceVersionedJson(legacyText);

    expect(JSON.parse(legacyText).schema_version).toBe(
      GROUND_PLAYBACK_WORKSPACE_SCHEMA,
    );
    expect(migrated).toEqual({
      ...legacy,
      schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
      comparison: null,
    });
    expect(JSON.parse(groundWorkspaceV2ToJson(migrated)).comparison).toBeNull();
    expect(load).toEqual({
      workspace: migrated,
      sourceSchemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA,
      migratedFromV1: true,
    });
    expect(() => groundWorkspaceV2FromJson(legacyText)).toThrow(
      /expected.*schema v2/i,
    );
  });

  it("rejects exact-field, duplicate, state, and document-bound violations", () => {
    const encoded = groundWorkspaceV2ToJson(workspaceV2());
    const unknown = JSON.parse(encoded);
    unknown.comparison.extra = true;
    expect(() => groundWorkspaceV2FromJson(JSON.stringify(unknown))).toThrow(
      /comparison fields do not match v2/i,
    );
    const wrongVisible = JSON.parse(encoded);
    wrongVisible.comparison.visible = 1;
    expect(() =>
      groundWorkspaceV2FromJson(JSON.stringify(wrongVisible)),
    ).toThrow(/comparison visible must be a boolean/i);
    const invalidTime = JSON.parse(encoded);
    invalidTime.playback.time_s = 99;
    expect(() =>
      groundWorkspaceV2FromJson(JSON.stringify(invalidTime)),
    ).toThrow(/union timeline/i);
    const duplicate = encoded.replace(
      `"schema_version":"${GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2}"`,
      `"schema_version":"${GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2}","schema_version":"duplicate"`,
    );
    expect(() => groundWorkspaceV2FromJson(duplicate)).toThrow(
      /duplicate JSON field/i,
    );
    expect(() => groundWorkspaceV2FromJson(encoded, { maxBytes: 10 })).toThrow(
      /size limit/i,
    );
    expect(() =>
      groundWorkspaceV2ToJson(workspaceV2(), { maxBytes: 10 }),
    ).toThrow(/size limit/i);
    expect(() =>
      groundWorkspaceV2FromJson(encoded, { maxPointsPerResult: 1 }),
    ).toThrow(/per-result point limit/i);
    expect(() =>
      groundWorkspaceV2FromJson(encoded, {
        maxPointsPerResult: 100,
        maxCombinedPoints: 3,
      }),
    ).toThrow(/combined point limit/i);
    expect(GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2).toBe(11 * 1024 * 1024);
  });

  it("matches the shared canonical golden bytes and SHA", async () => {
    expect(
      groundWorkspaceV2ToJson(groundWorkspaceV2FromJson(workspaceFixture)),
    ).toBe(workspaceFixture);
    expect(await sha256(workspaceFixture)).toBe(WORKSPACE_SHA256);
    expect(
      loadGroundWorkspaceVersionedJson(workspaceFixture).migratedFromV1,
    ).toBe(false);
  });
});
