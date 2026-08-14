import { describe, expect, it } from "vitest";

import { geometricVariability, type SwingTraceRowTs } from "./variationGeometry";

describe("geometric variability scale budget", () => {
  it("prepares 500 complete 240-sample traces within the interactive budget", () => {
    const trials = 500;
    const samples = 240;
    const timesS = Array.from({ length: samples }, (_, index) => index / (samples - 1));
    const traces: SwingTraceRowTs[] = Array.from({ length: trials }, (_, trialIndex) => {
      const offset = -0.02 + 0.04 * trialIndex / (trials - 1);
      return {
        trialIndex,
        status: "evaluated_hit",
        timesS,
        points: timesS.map((time) => [
          Math.sin(time) + offset,
          Math.cos(time) - 0.4 * offset,
          time + 0.2 * offset,
        ]),
      };
    });

    const started = performance.now();
    const result = geometricVariability(traces, 0.005);
    const elapsedMs = performance.now() - started;

    expect(elapsedMs).toBeLessThan(2000);
    expect(result.sampleTimesS).toHaveLength(samples);
    expect(result.validTrialCount.every((count) => count === trials)).toBe(true);
    expect(result.rmsRadiusM.every(Number.isFinite)).toBe(true);
  });
});
