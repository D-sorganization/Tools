import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/launch_monitor_registry_golden_v1.json";
import {
  COMPARABILITY_REASON,
  CONVENTION_IDS,
  PARAMETER_IDS,
  compareDefinitions,
  conventionRegistry,
  migrateConventionRegistry,
  shiftPointVelocity,
  stableConventionJson,
  transformVector,
} from "./launchMonitorConventions";

describe("launch-monitor convention registry", () => {
  it("contains each foundation parameter for all three conventions", () => {
    const registry = conventionRegistry();
    expect(registry.schemaVersion).toBe("launch-monitor-conventions/v1");
    for (const convention of CONVENTION_IDS) {
      expect(registry.definitions.filter((item) => item.conventionId === convention)
        .map((item) => item.parameterId).sort()).toEqual([...PARAMETER_IDS].sort());
    }
    expect(Object.isFrozen(registry.definitions)).toBe(true);
  });

  it("matches the shared Python/TypeScript golden contract", async () => {
    const registry = conventionRegistry();
    expect(registry.definitions).toHaveLength(fixture.definition_count);
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(stableConventionJson(registry)),
    );
    expect(Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""))
      .toBe(fixture.canonical_json_sha256);
    for (const golden of fixture.cases) {
      const definition = registry.definition(golden.convention_id, golden.parameter_id);
      expect(definition.referencePoint).toBe(golden.reference_point);
      expect(definition.eventTime).toBe(golden.event_time);
      expect(definition.quantityStatus).toBe(golden.quantity_status);
    }
  });

  it("reports reference-point and event-time incompatibility", () => {
    const registry = conventionRegistry();
    const trackman = registry.definition("trackman_comparable", "club_path");
    const foresight = registry.definition("foresight_comparable", "club_path");
    expect(compareDefinitions(trackman, foresight)).toEqual({
      comparable: false,
      reasons: [COMPARABILITY_REASON.referencePoint, COMPARABILITY_REASON.eventTime],
    });
  });

  it("reports geometry-contract incompatibility", () => {
    const definition = conventionRegistry().definition("app_native", "spin_loft");
    const planar = { ...definition, geometryContract: "planar_loft_difference" };
    expect(compareDefinitions(definition, planar).reasons)
      .toEqual([COMPARABILITY_REASON.geometry]);
  });

  it("reports sign-rule incompatibility", () => {
    const definition = conventionRegistry().definition("app_native", "launch_direction");
    const unspecified = { ...definition, signRule: "unspecified" as const };
    expect(compareDefinitions(definition, unspecified).reasons)
      .toEqual([COMPARABILITY_REASON.signRule]);
  });

  it("does not invent an absolute Foresight launch-direction sign", () => {
    const definition = conventionRegistry()
      .definition("foresight_comparable", "launch_direction");
    expect(definition.signRule).toBe("unspecified");
  });

  it("performs exact point and proper-frame transforms", () => {
    expect(shiftPointVelocity([50, 1, 0], [0, 20, 0], [0.04, 0, 0.02]))
      .toEqual(expect.arrayContaining([
        expect.closeTo(50.4, 12), expect.closeTo(1, 12), expect.closeTo(-0.8, 12),
      ]));
    expect(transformVector([1, 2, 3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
      .toEqual([3, 2, -1]);
    expect(() => transformVector([1, 0, 0], [[2, 0, 0], [2, 0, 0], [2, 0, 0]]))
      .toThrow(/proper orthonormal/);
  });

  it("serializes deterministically and migrates the explicit v0 field rename", () => {
    const registry = conventionRegistry();
    expect(stableConventionJson(registry)).toBe(stableConventionJson(registry));
    const legacy = JSON.parse(stableConventionJson(registry));
    legacy.schema_version = "launch-monitor-conventions/v0";
    legacy.definitions[0].vendor = legacy.definitions[0].convention_id;
    delete legacy.definitions[0].convention_id;
    expect(stableConventionJson(migrateConventionRegistry(legacy)))
      .toBe(stableConventionJson(registry));
  });
});
