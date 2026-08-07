import { describe, expect, it } from "vitest";

import {
  DRIVER_TEE_HEIGHT_M,
  type BallSetup,
} from "./ballSetup";
import {
  ballSetupFromSimulationDocument,
  createSimulationRunCsv,
  createSimulationRunDocument,
  exportBallSetupMetadata,
  loadBallSetupPreference,
  saveBallSetupPreference,
  spatialTargetFromSimulationDocument,
} from "./ballSetupPersistence";
import { runSimulation, type SimulationInput } from "./simulation";
import {
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "./spatialTarget";
import { spatialTargetToJson } from "./spatialTargetSerialization";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "./targets";

const tee: BallSetup = { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M };
const landingTarget = spatialTargetFromRegion(DEFAULT_TARGET);
const aerialTarget = createSpatialTarget({
  label: "Apex gate",
  kind: "aerial_waypoint",
  point: targetPointFromFrame([140, 24, -3], "app"),
  tolerance: sphereTolerance(4),
  elevationSource: "absolute",
});

describe("ball setup persistence", () => {
  it("exports canonical snake-case setup plus unit and reference metadata", () => {
    const input: SimulationInput = {
      sourceKind: "manual",
      clubheadSpeedMph: 100,
      omegaDps: [0, 0, 0],
      loftDeg: 10.5,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 1.5,
      ballSetup: tee,
    };
    const run = runSimulation(input);
    const document = createSimulationRunDocument(input, run, null, aerialTarget);
    expect(document.format).toBe("rate_of_closure.simulation_run.web/4");
    expect(document.parameters).toMatchObject({
      ball_setup: { support_mode: "tee", tee_height_m: DRIVER_TEE_HEIGHT_M },
    });
    expect(document.parameters.ballSetup).toBeUndefined();
    expect(document.ballSetupMetadata).toMatchObject({
      tee_height_unit: "m",
      height_reference: "ground_plane_to_ball_bottom",
      ball_center_m: [0, 0.059435, 0],
    });
    expect(document.series.clubScrewMotion).toMatchObject({
      frame: "app/world",
      units: "SI",
    });
    expect(document.series.clubScrewMotion.rows).toHaveLength(document.series.swing.length);
    const canonicalTarget = JSON.parse(spatialTargetToJson(aerialTarget));
    expect(document.spatial_target).toEqual(canonicalTarget);
    expect(document.solver_manifest).toMatchObject({
      schema: "swing_sim.solver_manifest",
      schema_version: 1,
      target: canonicalTarget,
    });
    expect(document.variation_manifest).toMatchObject({ target: canonicalTarget });

    const csv = createSimulationRunCsv(run, aerialTarget);
    expect(csv).toContain("target_schema,target_schema_version,target_label,target_kind");
    expect(csv).toContain("swing_sim.spatial_target,1,Apex gate,aerial_waypoint");
    expect(csv).toContain("140,24,-3");
    expect(csv).toContain("target_frame,target_source_frame,target_units");
    expect(csv).toContain(",app,app,m,absolute,");

    const missInput = { ...input, contactMode: "fixed_ball_contact" as const };
    const missCsv = createSimulationRunCsv(runSimulation(missInput), aerialTarget);
    expect(missCsv).toContain("swing_sim.spatial_target,1,Apex gate,aerial_waypoint");
  });

  it("round-trips the setup, override policy, and reference metadata", () => {
    const storage = new Map<string, string>();
    const adapter: Storage = {
      get length() { return storage.size; },
      clear: () => storage.clear(),
      getItem: (key) => storage.get(key) ?? null,
      key: (index) => [...storage.keys()][index] ?? null,
      removeItem: (key) => storage.delete(key),
      setItem: (key, value) => storage.set(key, value),
    };
    saveBallSetupPreference({ setup: tee, userOverridden: true }, adapter);
    expect(loadBallSetupPreference(adapter)).toEqual({
      setup: tee,
      userOverridden: true,
      warning: null,
    });
    expect(exportBallSetupMetadata(tee)).toMatchObject({
      support_mode: "tee",
      tee_height_m: DRIVER_TEE_HEIGHT_M,
      tee_height_unit: "m",
      height_reference: "ground_plane_to_ball_bottom",
    });
  });

  it("migrates old simulation exports to backward-compatible Ground behavior", () => {
    expect(ballSetupFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/2",
      parameters: { sourceKind: "manual" },
    })).toEqual({ supportMode: "ground", teeHeightM: 0 });
    expect(ballSetupFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/3",
      parameters: { ball_setup: { support_mode: "tee", tee_height_m: DRIVER_TEE_HEIGHT_M } },
    })).toEqual(tee);
    expect(ballSetupFromSimulationDocument({
      format: "rate_of_closure.simulation_run/2",
      parameters: { ball_setup: { support_mode: "tee", tee_height_m: 0.04 } },
    })).toEqual({ supportMode: "tee", teeHeightM: 0.04 });
    expect(() => ballSetupFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/99",
    })).toThrow(/unsupported.*version 99/i);

    expect(spatialTargetFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/3",
      parameters: { target: { ...DEFAULT_TARGET, distanceM: 205 } },
    })).toMatchObject({
      label: "Migrated Green Target",
      kind: "landing_area",
      point: { appCoordinatesM: [205, 0, 0] },
      groundSource: "legacy.course_surface/default",
    });
    expect(spatialTargetFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/3",
      parameters: {},
    })).toEqual(landingTarget);
  });

  it("round-trips canonical targets and rejects malformed current documents", () => {
    const document = {
      format: "rate_of_closure.simulation_run.web/4",
      spatial_target: JSON.parse(spatialTargetToJson(aerialTarget)),
    };
    expect(spatialTargetFromSimulationDocument(document)).toEqual(aerialTarget);
    expect(() => spatialTargetFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/4",
    })).toThrow(/requires spatial_target/i);
    expect(() => spatialTargetFromSimulationDocument({
      ...document,
      spatial_target: { ...document.spatial_target, units: "yd" },
    })).toThrow(/units must be 'm'/i);
    expect(() => ballSetupFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/4",
      spatial_target: document.spatial_target,
      parameters: {},
    })).toThrow(/requires ball_setup/i);
  });

  it("neutralizes formula-leading target text in CSV while preserving numerics", () => {
    const hostileTarget = createSpatialTarget({
      ...landingTarget,
      label: "=HYPERLINK(\"https://example.invalid\")",
      groundSource: "@malicious-source",
    });
    const input: SimulationInput = {
      sourceKind: "manual", clubheadSpeedMph: 100, omegaDps: [0, 0, 0],
      loftDeg: 10.5, impactOffsetToeMm: 0, impactOffsetHighMm: 0,
      planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
      impactTimeS: null, swingDurationS: 1.5, ballSetup: tee,
    };
    const csv = createSimulationRunCsv(runSimulation(input), hostileTarget);
    expect(csv).toContain("'=" + "HYPERLINK");
    expect(csv).toContain("'@malicious-source");
    expect(csv).toContain(",230,0,0,app,app,m,");
    expect(csv).not.toContain("'230");
  });

  it("reports corrupt preferences without applying unsafe geometry", () => {
    const storage = { getItem: () => "{bad", setItem: () => undefined } as unknown as Storage;
    const loaded = loadBallSetupPreference(storage);
    expect(loaded.setup).toEqual({ supportMode: "ground", teeHeightM: 0 });
    expect(loaded.warning).toMatch(/could not be loaded/i);
  });

  it("reports unavailable persistent storage without crashing the editor", () => {
    const storage = {
      getItem: () => null,
      setItem: () => { throw new Error("quota exceeded"); },
    } as unknown as Storage;
    expect(saveBallSetupPreference({ setup: tee, userOverridden: true }, storage))
      .toMatch(/could not be saved.*quota exceeded/i);

    const unreadable = {
      getItem: () => { throw new Error("access denied"); },
      setItem: () => undefined,
    } as unknown as Storage;
    expect(loadBallSetupPreference(unreadable, tee)).toEqual({
      setup: tee,
      userOverridden: false,
      warning: "Saved ball setup could not be read: access denied",
    });
  });
});
