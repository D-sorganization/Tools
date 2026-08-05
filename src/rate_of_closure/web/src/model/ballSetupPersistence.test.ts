import { describe, expect, it } from "vitest";

import {
  DRIVER_TEE_HEIGHT_M,
  type BallSetup,
} from "./ballSetup";
import {
  ballSetupFromSimulationDocument,
  createSimulationRunDocument,
  exportBallSetupMetadata,
  loadBallSetupPreference,
  saveBallSetupPreference,
} from "./ballSetupPersistence";
import { runSimulation, type SimulationInput } from "./simulation";

const tee: BallSetup = { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M };

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
    const document = createSimulationRunDocument(input, runSimulation(input));
    expect(document.format).toBe("rate_of_closure.simulation_run.web/3");
    expect(document.parameters).toMatchObject({
      ball_setup: { support_mode: "tee", tee_height_m: DRIVER_TEE_HEIGHT_M },
    });
    expect(document.parameters.ballSetup).toBeUndefined();
    expect(document.ballSetupMetadata).toMatchObject({
      tee_height_unit: "m",
      height_reference: "ground_plane_to_ball_bottom",
      ball_center_m: [0, 0.059435, 0],
    });
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
