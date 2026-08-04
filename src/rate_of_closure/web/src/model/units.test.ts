/**
 * Parity tests for units and the common literature closure metrics —
 * pinned against tests/rate_of_closure/test_units.py.
 */

import { describe, expect, it } from "vitest";

import { METRIC_EXPLANATIONS } from "./derivation";
import { closureMetrics, DEFAULT_SCENARIO, solve } from "./impact";
import {
  FIELD_GUIDANCE,
  fromCanonical,
  QUANTITY_UNITS,
  toCanonical,
  type Quantity,
} from "./units";

describe("unit conversions — parity with pytest", () => {
  it("round-trips every unit", () => {
    for (const quantity of Object.keys(QUANTITY_UNITS) as Quantity[]) {
      for (const unit of Object.keys(QUANTITY_UNITS[quantity])) {
        const out = fromCanonical(
          quantity,
          unit,
          toCanonical(quantity, unit, 123.456),
        );
        expect(out).toBeCloseTo(123.456, 9);
      }
    }
  });

  it("pins known conversions", () => {
    expect(toCanonical("speed", "m/s", 53.645)).toBeCloseTo(120.0, 1);
    expect(toCanonical("rotation", "rpm", 350)).toBeCloseTo(2100.0, 6);
    expect(toCanonical("length", "in", 1)).toBeCloseTo(25.4, 9);
  });

  it("every scenario field has guidance with a source", () => {
    for (const key of Object.keys(DEFAULT_SCENARIO)) {
      expect(FIELD_GUIDANCE[key], key).toContain("Suggested range");
      expect(FIELD_GUIDANCE[key], key).toContain("Source:");
    }
  });
});

describe("closure metrics — parity with pytest", () => {
  it("restates the solved delivery", () => {
    const result = solve(DEFAULT_SCENARIO);
    const metrics = closureMetrics(DEFAULT_SCENARIO);
    expect(metrics.ccvDps).toBeCloseTo(result.closureRateDps, 9);
    expect(metrics.closureDegPerInch).toBeCloseTo(
      metrics.closureDegPerFt / 12,
      9,
    );
    expect(metrics.closureDegPerMs).toBeCloseTo(metrics.ccvDps / 1000, 9);
  });

  it("d / R_ISA reproduces the path gap", () => {
    const result = solve(DEFAULT_SCENARIO);
    const metrics = closureMetrics(DEFAULT_SCENARIO);
    const gapDeg = ((0.04 / metrics.rIsaM) * 180) / Math.PI;
    expect(gapDeg / Math.abs(result.pathDeviationDeg)).toBeCloseTo(1.0, 1);
  });

  it("time to square is about half a millisecond at tour rates", () => {
    const metrics = closureMetrics(DEFAULT_SCENARIO);
    expect(metrics.timeToSquareFrom1DegOpenMs).toBeGreaterThan(0.3);
    expect(metrics.timeToSquareFrom1DegOpenMs).toBeLessThan(0.8);
  });

  it("non-closing face reports infinite ratios", () => {
    const metrics = closureMetrics({
      ...DEFAULT_SCENARIO,
      omegaPlaneDps: 0,
      omegaShaftDps: 0,
    });
    expect(metrics.rIsaM).toBe(Infinity);
    expect(metrics.timeToSquareFrom1DegOpenMs).toBe(Infinity);
    expect(metrics.toeHeelSpeedDeltaMph).toBeCloseTo(0, 9);
  });

  it("every metric has a substantial, brand-neutral explanation", () => {
    const metrics = closureMetrics(DEFAULT_SCENARIO);
    for (const key of Object.keys(metrics)) {
      expect(METRIC_EXPLANATIONS[key], key).toBeDefined();
      expect(METRIC_EXPLANATIONS[key].length, key).toBeGreaterThan(80);
      expect(METRIC_EXPLANATIONS[key], key).not.toContain("TrackMan");
    }
  });
});
