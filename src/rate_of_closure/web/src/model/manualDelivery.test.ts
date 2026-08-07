import { describe, expect, it } from "vitest";

import {
  DEFAULT_MANUAL_DELIVERY,
  manualDeliveryFromSimulationDocument,
  resolveManualDelivery,
  validateDeliveredDynamicLoft,
} from "./manualDelivery";
import {
  ballSetupFromSimulationDocument,
  createSimulationRunDocument,
  spatialTargetFromSimulationDocument,
} from "./ballSetupPersistence";
import { runSimulation, type SimulationInput } from "./simulation";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "./targets";

describe("manual delivery contract", () => {
  it("preserves the legacy target-line, level, tracked-reference defaults", () => {
    expect(resolveManualDelivery({})).toEqual(DEFAULT_MANUAL_DELIVERY);
  });

  it.each([
    ["manualAttackAngleDeg", 90],
    ["manualClubPathDeg", -90],
    ["manualForwardShaftLeanDeg", 61],
    ["manualAttackAngleDeg", Number.NaN],
  ])("rejects an invalid %s", (key, value) => {
    expect(() => resolveManualDelivery({ [key]: value })).toThrow(key);
  });

  it("rejects an unknown shaft-axis datum", () => {
    expect(() => resolveManualDelivery({ shaftAxisDatum: "invented" as never }))
      .toThrow(/shaftAxisDatum/);
  });

  it.each([
    [46, -43, 89],
    [-29, 60, -89],
  ])("accepts delivered dynamic loft boundary %s - %s = %s", (loft, lean, expected) => {
    expect(validateDeliveredDynamicLoft(loft, lean)).toBe(expected);
  });

  it.each([
    [46, -44, 90],
    [-30, 60, -90],
    [Number.NaN, 0, Number.NaN],
  ])("rejects out-of-contract delivered dynamic loft %s - %s", (loft, lean) => {
    expect(() => validateDeliveredDynamicLoft(loft, lean))
      .toThrow(/delivered dynamic loft.*\[-89, 89\]/i);
  });

  it("imports new run parameters and migrates older documents to defaults", () => {
    expect(manualDeliveryFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/5",
      parameters: {
        manual_delivery: {
          attack_angle_deg: -7,
          club_path_deg: 4,
          forward_shaft_lean_deg: 12,
          shaft_axis_datum: "generated_hosel",
        },
      },
    })).toEqual({
      manualAttackAngleDeg: -7,
      manualClubPathDeg: 4,
      manualForwardShaftLeanDeg: 12,
      shaftAxisDatum: "generated_hosel",
    });
    expect(manualDeliveryFromSimulationDocument({
      format: "rate_of_closure.simulation_run.web/3",
      parameters: { sourceKind: "manual" },
    })).toEqual(DEFAULT_MANUAL_DELIVERY);
  });

  it("exports the resolved manual delivery fields with a simulation run", () => {
    const input: SimulationInput = {
      sourceKind: "manual",
      clubheadSpeedMph: 30,
      omegaDps: [0, 0, 0],
      loftDeg: 46,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 1.5,
      ballSetup: { supportMode: "ground", teeHeightM: 0 },
      manualAttackAngleDeg: -10,
      manualClubPathDeg: 6,
      manualForwardShaftLeanDeg: 15,
      shaftAxisDatum: "generated_hosel",
    };
    const document = createSimulationRunDocument(
      input,
      runSimulation(input),
      null,
      spatialTargetFromRegion(DEFAULT_TARGET),
    );
    expect(document.parameters).toMatchObject({
      manual_delivery: {
        attack_angle_deg: -10,
        club_path_deg: 6,
        forward_shaft_lean_deg: 15,
        shaft_axis_datum: "generated_hosel",
      },
    });
    expect(ballSetupFromSimulationDocument(document)).toEqual(input.ballSetup);
    expect(spatialTargetFromSimulationDocument(document))
      .toEqual(spatialTargetFromRegion(DEFAULT_TARGET));
    expect(manualDeliveryFromSimulationDocument(document)).toEqual({
      manualAttackAngleDeg: -10,
      manualClubPathDeg: 6,
      manualForwardShaftLeanDeg: 15,
      shaftAxisDatum: "generated_hosel",
    });
    expect(() => ballSetupFromSimulationDocument({
      ...document,
      format: "rate_of_closure.simulation_run.web/6",
    })).toThrow(/unsupported.*version 6/i);
    const legacyDefaultInput = {
      ...input,
      manualAttackAngleDeg: undefined,
      manualClubPathDeg: undefined,
      manualForwardShaftLeanDeg: undefined,
      shaftAxisDatum: undefined,
    };
    const defaultDocument = createSimulationRunDocument(
      legacyDefaultInput,
      runSimulation(legacyDefaultInput),
      null,
      spatialTargetFromRegion(DEFAULT_TARGET),
    );
    expect(defaultDocument.parameters.manual_delivery).toEqual({
      attack_angle_deg: 0,
      club_path_deg: 0,
      forward_shaft_lean_deg: 0,
      shaft_axis_datum: "tracked_reference",
    });
  });
});
