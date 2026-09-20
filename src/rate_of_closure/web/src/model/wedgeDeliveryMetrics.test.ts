import { describe, expect, it } from "vitest";

import { getClub, type ClubSpec } from "./club";
import { DEFAULT_SCENARIO, solve } from "./impact";
import { runSimulation, type SimulationInput, type SimulationRunTs } from "./simulation";
import {
  computeLinearVelocityWaterfall,
  computeWedgeDeliveryMetrics,
  wedgeDeliveryMetricsForRun,
} from "./wedgeDeliveryMetrics";
import { representativeWedgeForClub } from "./wedgeGroundGeometry";

const mockClub = getClub("Sand Wedge");
const mockScenario = { ...DEFAULT_SCENARIO, clubheadSpeedMph: 45, lieAngleDeg: 64 };
const mockInput: SimulationInput = {
  sourceKind: "manual",
  clubheadSpeedMph: 45,
  omegaDps: solve(mockScenario).omegaDps,
  loftDeg: 56,
  impactOffsetToeMm: 0,
  impactOffsetHighMm: 0,
  planeYawDeg: 0,
  planeSideTiltDeg: -45,
  planeForwardTiltDeg: 0,
  impactTimeS: 0.03,
  swingDurationS: 1.5,
};
const mockRun: SimulationRunTs = runSimulation(mockInput);

describe("wedgeDeliveryMetrics", () => {
  it("enforces exact linear addition: v_contact = v_axis + v_shaft + v_other", () => {
    const vRef = [15, -2.5, 0.5] as const;
    const omega = [5, 18, -3] as const;
    const shaftPoint = [0, 0, 0] as const;
    const lieRad = (64 * Math.PI) / 180;
    const shaftUnit = [0, Math.sin(lieRad), -Math.cos(lieRad)] as const;
    const contactPt = [0.02, 0.01, -0.005] as const;

    const wf = computeLinearVelocityWaterfall(
      [...vRef],
      [...omega],
      [...shaftPoint],
      [...shaftUnit],
      [...contactPt],
    );

    const sumX = wf.baseAxis.downrangeMps + wf.shaftRotation.downrangeMps + wf.otherRotation.downrangeMps;
    const sumY = wf.baseAxis.verticalMps + wf.shaftRotation.verticalMps + wf.otherRotation.verticalMps;
    const sumZ = wf.baseAxis.lateralMps + wf.shaftRotation.lateralMps + wf.otherRotation.lateralMps;

    expect(sumX).toBeCloseTo(wf.totalContact.downrangeMps, 10);
    expect(sumY).toBeCloseTo(wf.totalContact.verticalMps, 10);
    expect(sumZ).toBeCloseTo(wf.totalContact.lateralMps, 10);
  });

  it("yields zero shaft velocity when rotation is zero", () => {
    const wf = computeLinearVelocityWaterfall(
      [15, -2, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0.9, -0.4],
      [0.02, 0.01, 0],
    );

    expect(wf.shaftRotation.totalSpeedMps).toBeCloseTo(0, 10);
    expect(wf.otherRotation.totalSpeedMps).toBeCloseTo(0, 10);
    expect(wf.shaftCounterfactualAoaDeltaDeg).toBeCloseTo(0, 10);
    expect(wf.totalAoaDeg).toBeCloseTo(wf.withoutShaftAoaDeg!, 10);
  });

  it("satisfies Shapley attribution summing to total swing delta", () => {
    const wf = computeLinearVelocityWaterfall(
      [15, -2, 0],
      [8, 20, 5],
      [0, 0, 0],
      [0, 0.898, -0.438],
      [0.02, 0.01, 0],
    );

    expect(wf.totalAoaDeg).not.toBeNull();
    expect(wf.shaftShapleyAoaDeg).not.toBeNull();
    expect(wf.otherShapleyAoaDeg).not.toBeNull();

    const baseAoa = (Math.atan2(wf.baseAxis.verticalMps, wf.baseAxis.downrangeMps) * 180) / Math.PI;
    const totalDelta = wf.totalAoaDeg! - baseAoa;
    const shapleySum = wf.shaftShapleyAoaDeg! + wf.otherShapleyAoaDeg!;
    expect(shapleySum).toBeCloseTo(totalDelta, 8);
  });

  it("computes comprehensive delivery metrics and explainers for wedge run", () => {
    const wedge = representativeWedgeForClub(mockClub)!;
    expect(wedge).not.toBeNull();

    const metrics = computeWedgeDeliveryMetrics(
      wedge,
      { position: [0, 0.02, 0], rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] },
      { velocity: [15, -2, 0], angularVelocity: [0, 20, 0] },
    );

    expect(metrics.dynamicLoftDeg).toBeGreaterThan(0);
    expect(metrics.dynamicLieDeg).toBeGreaterThan(0);
    expect(metrics.leDownrangeRateMps).toBeCloseTo(15, 1);
    expect(metrics.le3dAngularRateDps).toBeGreaterThan(0);
    expect(metrics.explainers.length).toBeGreaterThan(10);

    const keys = new Set(metrics.explainers.map((e) => e.key));
    expect(keys.has("total_aoa")).toBe(true);
    expect(keys.has("shaft_counterfactual_aoa_delta")).toBe(true);
    expect(keys.has("le_vertical_rate")).toBe(true);
    expect(keys.has("waterfall_axis")).toBe(true);
    expect(keys.has("waterfall_shaft")).toBe(true);
  });

  it("integrates with wedgeDeliveryMetricsForRun", () => {
    const metrics = wedgeDeliveryMetricsForRun(mockRun, mockScenario, mockClub);
    expect(metrics).not.toBeNull();
    expect(metrics!.waterfall.totalContact.totalSpeedMps).toBeGreaterThan(0);

    // Non-wedge club returns null
    const driverClub: ClubSpec = { ...mockClub, clubType: "Wood" };
    expect(wedgeDeliveryMetricsForRun(mockRun, mockScenario, driverClub)).toBeNull();
  });
});
