import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_regional_execution_golden_v1.json";
import { parseGroundRegionalExecutionResult } from "./groundRegionalExecution";
import {
  readRegionalExecutionEvidenceFile,
  regionalExecutionReadback,
} from "./regionalExecutionReadback";

describe("regional execution evidence readback", () => {
  const result = parseGroundRegionalExecutionResult(fixture.representable.result);

  it("reports Python-produced evidence bound to the exact visible plan", () => {
    const readback = regionalExecutionReadback(result, result.regional_plan);

    expect(readback.status).toBe("partial");
    expect(readback.planId).toBe("regional-execution-plan-001");
    expect(readback.terminationReason).toBe("time_limit");
    expect(readback.groundTimeS).toBeCloseTo(1.155);
    expect(readback.completed).toBe(false);
    expect(readback.transitionCount).toBe(1);
    expect(readback.carryDistanceM).toBeCloseTo(0);
    expect(readback.bounceAirDistanceM).toBeCloseTo(0.04);
    expect(readback.skidDistanceM).toBeCloseTo(0);
    expect(readback.rollDistanceM).toBeCloseTo(0.25374857896);
    expect(readback.surfacePathDistanceM).toBeCloseTo(0.25374857896);
    expect(readback.totalDistanceM).toBeCloseTo(0.2937485791);
    expect(readback.finalDownrangeM).toBeCloseTo(0.2937485791);
    expect(readback.finalOfflineM).toBeCloseTo(0);
    expect(readback.bounceCount).toBe(1);
    expect(readback.surfaceProviderId).toBe("tools.planar-surface");
    expect(readback.surfaceProviderVersion).toBe("1.0.0");
    expect(readback.calibrationId).toBe("literature-default-2026-08");
    expect(readback.calibrationKind).toBe("literature");
    expect(readback.calibrationSource).toBe("documented literature basis");
    expect(readback.calibrationConfidence).toBeCloseTo(0.6);
    expect(readback.observedPhases).toEqual(["impact", "skid", "roll"]);
    expect(readback.unitSystem).toBe("SI");
    expect(readback.events).toHaveLength(4);
    expect(readback.events[0]).toMatchObject({
      sequence: 0,
      eventType: "first_contact",
      timeS: 1.005,
      positionM: [0, 0.02135, 0],
      velocityBeforeMps: [2, -0.1, 0],
      angularVelocityAfterRadS: [0, 0, -93.67681498829],
    });
    expect(readback.transitions).toEqual([expect.objectContaining({
      eventSequence: 3,
      fromRegionId: null,
      toRegionId: "rough-band",
      fromSurfaceId: "firm-fairway",
      toSurfaceId: "regional-rough",
    })]);
    expect(readback.warnings[readback.warnings.length - 1]).toEqual({
      code: "CENSORED_ENDPOINT",
      severity: "warning",
      message: "Distance totals describe only the observed endpoint and are not projected final-rest metrics.",
    });
  });

  it("does not fabricate metrics for null-result cancellation evidence", () => {
    const cancelled = parseGroundRegionalExecutionResult(fixture.cancelled.result);
    const readback = regionalExecutionReadback(cancelled, cancelled.regional_plan);

    expect(readback.failureReason).toBe("cancelled");
    expect(readback.groundTimeS).toBeNull();
    expect(readback.completed).toBeNull();
    expect(readback.carryDistanceM).toBeNull();
    expect(readback.bounceCount).toBeNull();
    expect(readback.calibrationKind).toBeNull();
    expect(readback.calibrationId).toBeNull();
    expect(readback.calibrationSource).toBeNull();
    expect(readback.calibrationConfidence).toBeNull();
    expect(readback.observedPhases).toEqual([]);
    expect(readback.events).toEqual([]);
    expect(readback.transitions).toEqual([]);
    expect(readback.warnings).toEqual([]);
  });

  it("rejects evidence for a different plan and oversize browser files", async () => {
    expect(() => regionalExecutionReadback(result, {
      ...result.regional_plan,
      request_id: "different-plan",
    })).toThrow(/does not match the current regional plan/);

    await expect(readRegionalExecutionEvidenceFile({
      name: "oversize.json",
      size: 8_388_609,
      text: async () => "{}",
    }, result.regional_plan)).rejects.toThrow(/maximum wire size/);
  });

  it("strictly parses a bounded browser file before presenting it", async () => {
    const loaded = await readRegionalExecutionEvidenceFile({
      name: "execution.json",
      size: JSON.stringify(fixture.representable.result).length,
      text: async () => JSON.stringify(fixture.representable.result),
    }, result.regional_plan);

    expect(loaded.result.status).toBe("partial");
    expect(loaded.readback.executorSourceRevision).toBe("ground-regional-execution-v1");
  });
});
