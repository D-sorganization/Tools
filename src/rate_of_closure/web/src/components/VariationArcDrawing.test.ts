import { describe, expect, it, vi } from "vitest";

import type { ConfidenceEllipsoidMeshTs } from "../model/confidenceEllipsoidMesh";
import type { GeometricVariabilityTs, SwingTraceRowTs } from "../model/variationGeometry";
import { drawVariationArcScene } from "./VariationArcDrawing";

function capturingContext(paths: Array<Array<[number, number]>>): CanvasRenderingContext2D {
  let alpha = 1;
  let current: Array<[number, number]> = [];
  return {
    clearRect: () => undefined,
    fillRect: () => undefined,
    fillText: () => undefined,
    beginPath: () => { current = []; },
    moveTo: (x: number, y: number) => { current.push([x, y]); },
    lineTo: (x: number, y: number) => { current.push([x, y]); },
    closePath: () => undefined,
    stroke: () => undefined,
    fill: () => { if (alpha === 0.16) paths.push([...current]); },
    get globalAlpha() { return alpha; },
    set globalAlpha(value: number) { alpha = value; },
  } as unknown as CanvasRenderingContext2D;
}

const variability: GeometricVariabilityTs = {
  sampleTimesS: [], validTrialCount: [], meanPositionsM: [], rmsRadiusM: [],
  principalSigmaM: [], principalAxes: [], principalFrames: [],
  confidenceSemiAxisLengthsM: [], metric: "confidence-ellipsoid-volume",
  authorityUnit: "m^3", displayUnit: "mm³", confidenceLevel: 0.95,
  interpretation: "gaussian-position-content-region", metricValues: [],
  displayValues: [], adequacy: [],
  adequacyCounts: {
    estimable: 0, "rank-deficient": 0, "insufficient-samples": 0,
    "invalid-covariance": 0,
  },
  unavailableCount: 0, quietMask: [], quietIntervals: [],
  criteria: {
    metric: "confidence-ellipsoid-volume", maxValue: 1,
    confidenceLevel: 0.95, minDurationS: 0, minSamples: 1,
  },
  coordinateFrame: "app_frame:x_target,y_up,z_right",
  alignmentBasis: "common_simulation_time_s",
};

describe("variation arc confidence surface projection", () => {
  it("maps x horizontally, y vertically, z in depth and includes mesh in bounds", () => {
    const paths: Array<Array<[number, number]>> = [];
    const trace: SwingTraceRowTs = {
      trialIndex: 0, points: [[0, 0, 0]], timesS: [0], status: "evaluated_hit",
    };
    const mesh: ConfidenceEllipsoidMeshTs = {
      coordinateFrame: "app_frame:x_target,y_up,z_right",
      interpretation: "gaussian-position-content-region", sampleIndices: [0],
      verticesM: [[10, 0, 0], [0, 10, 0], [0, 0, 10]], triangles: [[0, 1, 2]],
      verticesPerEllipsoid: 3, trianglesPerEllipsoid: 1,
    };

    drawVariationArcScene(
      capturingContext(paths), 100, 100, [trace], variability,
      { yaw: 0, pitch: 0, zoom: 1 }, 1, null, mesh,
    );

    expect(paths).toHaveLength(1);
    expect(paths[0][0][0]).toBeCloseTo(74.2487, 3);
    expect(paths[0][0][1]).toBeCloseTo(74.2487, 3);
    expect(paths[0][1][0]).toBeCloseTo(25.7513, 3);
    expect(paths[0][1][1]).toBeCloseTo(25.7513, 3);
    expect(paths[0][2][0]).toBeCloseTo(25.7513, 3);
    expect(paths[0][2][1]).toBeCloseTo(74.2487, 3);
    expect(paths[0].flat().every((value) => value >= 0 && value <= 100)).toBe(true);
  });

  it("streams maximum-scale render bounds without materializing raw points", () => {
    const paths: Array<Array<[number, number]>> = [];
    const points: SwingTraceRowTs["points"] = Array.from(
      { length: 1_501 },
      (_, index) => [index / 1_500, Math.sin(index / 100), Math.cos(index / 100)],
    );
    const timesS = Array.from({ length: 1_501 }, (_, index) => index / 1_500);
    const traces: SwingTraceRowTs[] = Array.from({ length: 500 }, (_, trialIndex) => ({
      trialIndex, points, timesS, status: "evaluated_hit",
    }));
    const mesh: ConfidenceEllipsoidMeshTs = {
      coordinateFrame: "app_frame:x_target,y_up,z_right",
      interpretation: "gaussian-position-content-region", sampleIndices: [0],
      verticesM: [[-2, 0, 0], [2, 0, 0], [0, 2, 0]], triangles: [[0, 1, 2]],
      verticesPerEllipsoid: 3, trianglesPerEllipsoid: 1,
    };

    const started = performance.now();
    const flatMapSpy = vi.spyOn(Array.prototype, "flatMap");
    try {
      expect(() => drawVariationArcScene(
        capturingContext(paths), 400, 300, traces, variability,
        { yaw: 0.2, pitch: -0.1, zoom: 1 }, 25, null, mesh,
      )).not.toThrow();
      expect(flatMapSpy).not.toHaveBeenCalled();
    } finally {
      flatMapSpy.mockRestore();
    }
    const elapsedMs = performance.now() - started;

    expect(elapsedMs).toBeLessThan(2_000);
    expect(paths).toHaveLength(1);
    expect(paths[0].every(([x, y]) => (
      Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= 400 && y >= 0 && y <= 300
    ))).toBe(true);
  });
});
