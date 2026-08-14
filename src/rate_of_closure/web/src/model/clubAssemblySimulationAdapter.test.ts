import { describe, expect, it, vi } from "vitest";

import bindingFixture from "../../../../../tests/rate_of_closure/fixtures/club_assembly_binding_driver_10_5.json";
import { getClub } from "./club";
import {
  parseClubAssemblyBinding,
  type ClubAssemblyBindingDigestRuntime,
} from "./clubAssemblyBinding";
import { adaptClubAssemblyForImpact } from "./clubAssemblySimulationAdapter";
import { runSimulation, type SimulationInput } from "./simulation";

const DRIVER = "Driver 10.5\u00b0";

function digestRuntime(): ClubAssemblyBindingDigestRuntime {
  return {
    sha256Hex: vi.fn(async (payload: ArrayBuffer) => {
      const decoded = new TextDecoder().decode(payload);
      return decoded.includes("driver-qualified-2026-08")
        ? bindingFixture.assembly_identity.sha256
        : bindingFixture.selected_spec_identity.sha256;
    }),
  };
}

async function binding() {
  return parseClubAssemblyBinding(
    getClub(DRIVER),
    JSON.stringify(bindingFixture),
    digestRuntime(),
  );
}

function input(): SimulationInput {
  return {
    sourceKind: "manual",
    clubheadSpeedMph: 120,
    omegaDps: [0, 0, 0],
    loftDeg: 10.5,
    impactOffsetToeMm: 5,
    impactOffsetHighMm: 3,
    planeYawDeg: 0,
    planeSideTiltDeg: -45,
    planeForwardTiltDeg: 0,
    impactTimeS: null,
    swingDurationS: 1.5,
    club: getClub(DRIVER),
  };
}

describe("ClubAssembly simulation adapter", () => {
  it("keeps head tensor and all CG/assembly properties unavailable in scalar web impact", async () => {
    const adapted = adaptClubAssemblyForImpact(
      getClub(DRIVER),
      await binding(),
    );

    expect(adapted.headMassKg).toBeCloseTo(0.2, 14);
    expect(adapted.headInertiaTensorAppKgM2).toBeNull();
    expect(adapted.headInertia).toMatchObject({
      status: "unavailable",
      consumed: false,
    });
    expect(adapted.headInertia.reason).toMatch(/scalar-MOI-only/);
    expect(adapted.headCenterOfMass.reason).toMatch(/does not accept/);
    expect(adapted.assemblyMassProperties.reason).toMatch(
      /must not substitute/,
    );
  });

  it("rejects a binding retained against another selected spec", async () => {
    const validated = await binding();
    expect(() =>
      adaptClubAssemblyForImpact(
        { ...getClub(DRIVER), loftDeg: 11 },
        validated,
      ),
    ).toThrow(/selected ClubSpec identity/);
  });

  it("records no-impact non-consumption without fabricating flight", async () => {
    const simulationInput = input();
    simulationInput.assemblyBinding = await binding();
    simulationInput.assemblyClubSpec = getClub(DRIVER);
    simulationInput.contactMode = "fixed_ball_contact";
    simulationInput.ballSetup = { supportMode: "tee", teeHeightM: 0.1 };

    const run = runSimulation(simulationInput);

    expect(run.impactOutcome.status).toBe("miss");
    expect(run.launch).toBeNull();
    expect(run.clubAssemblyUsage.headInertia).toMatchObject({
      status: "not_used",
      consumed: false,
    });
    expect(run.clubAssemblyUsage.headInertia.reason).toMatch(
      /no club-ball impact/,
    );
  });
});
