import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_reference_pipeline_golden_v1.json";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import {
  GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  groundEventCsv,
  groundResultJson,
  groundTrajectoryCsv,
  groundWorkspaceFromJson,
  groundWorkspaceToJson,
  type GroundPlaybackWorkspace,
} from "./groundPlaybackWorkspace";

const workspace = (): GroundPlaybackWorkspace => ({
  schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  result: parseFlightToGroundResultRecord(fixture.result),
  playback: { timeS: 1.205, speed: 2, loop: true },
  view: { yawDeg: -37.5, pitchDeg: 18, zoom: 1.75 },
});

describe("ground playback workspace", () => {
  it("round trips a strict versioned deterministic document", () => {
    const encoded = groundWorkspaceToJson(workspace());
    const restored = groundWorkspaceFromJson(encoded);

    expect(groundWorkspaceToJson(restored)).toBe(encoded);
    expect(restored).toEqual(workspace());
    expect(JSON.parse(encoded).schema_version).toBe(GROUND_PLAYBACK_WORKSPACE_SCHEMA);
    expect(JSON.parse(encoded).result).toEqual(fixture.result);
    expect(JSON.parse(encoded).playback).not.toHaveProperty("playing");
  });

  it("rejects duplicates, unknown fields, and invalid state", () => {
    const encoded = groundWorkspaceToJson(workspace());
    expect(() => groundWorkspaceFromJson(encoded.replace(
      `"schema_version":"${GROUND_PLAYBACK_WORKSPACE_SCHEMA}"`,
      `"schema_version":"${GROUND_PLAYBACK_WORKSPACE_SCHEMA}","schema_version":"duplicate"`,
    ))).toThrow(/duplicate JSON field/i);

    const unknown = { ...JSON.parse(encoded), unexpected: true };
    expect(() => groundWorkspaceFromJson(JSON.stringify(unknown))).toThrow(/fields do not match/i);

    const invalidTime = JSON.parse(encoded);
    invalidTime.playback.time_s = 99;
    expect(() => groundWorkspaceFromJson(JSON.stringify(invalidTime)))
      .toThrow(/within the result timeline/i);

    const invalidSpeed = JSON.parse(encoded);
    invalidSpeed.playback.speed = 3;
    expect(() => groundWorkspaceFromJson(JSON.stringify(invalidSpeed)))
      .toThrow(/supported playback speed/i);
  });

  it("exports lossless deterministic result and evidence CSV", () => {
    const result = workspace().result;
    expect(groundResultJson(result)).toBe(groundResultJson(
      parseFlightToGroundResultRecord(JSON.parse(groundResultJson(result))),
    ));

    const trajectory = groundTrajectoryCsv(result);
    expect(trajectory.endsWith("\n")).toBe(true);
    expect(trajectory).not.toContain("\r");
    const trajectoryRows = trajectory.trimEnd().split("\n");
    expect(trajectoryRows[0]).toBe(
      "sample_index,time_s,phase,frame,position_x_m,position_y_m,position_z_m," +
      "velocity_x_m_s,velocity_y_m_s,velocity_z_m_s," +
      "angular_velocity_x_rad_s,angular_velocity_y_rad_s,angular_velocity_z_rad_s",
    );
    expect(trajectoryRows).toHaveLength(result.trajectory.length + 1);
    expect(trajectoryRows[1]).toContain(',0,0.02135,0,0.976,0.02,0,0,0,-2.81030444965');

    const events = groundEventCsv(result);
    expect(events.endsWith("\n")).toBe(true);
    expect(events).not.toContain("\r");
    const eventRows = events.trimEnd().split("\n");
    expect(eventRows[0]).toMatch(/^sequence,event_type,time_s,frame,position_x_m/);
    expect(eventRows).toHaveLength(result.events.length + 1);
    expect(eventRows[1]).toContain(',0,0.02135,0,1,-0.1,0,0.976,0.02,0');
  });
});
