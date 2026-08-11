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
    expect(readback.transitionCount).toBe(1);
    expect(readback.skidDistanceM).toBeCloseTo(0);
    expect(readback.rollDistanceM).toBeCloseTo(0.25374857896);
    expect(readback.totalDistanceM).toBeCloseTo(0.2937485791);
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
