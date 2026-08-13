import { describe, expect, it } from "vitest";

import {
  analyzeLaunchMonitorData,
  parseLaunchMonitorFile,
  sha256Text,
  type LaunchMonitorRow,
} from "./launchMonitorAnalysis";

const rows = (): LaunchMonitorRow[] =>
  Array.from({ length: 80 }, (_, index) => {
    const clubSpeed = 35 + index * 0.2;
    const attackAngle = -3 + (index % 8) * 0.5;
    return {
      shot_id: `shot-${index}`,
      session_id: index < 40 ? "a" : "b",
      monitor_vendor: index % 2 ? "FlightScope" : "TrackMan",
      club_speed: clubSpeed,
      attack_angle: attackAngle,
      ball_speed: 1.48 * clubSpeed + 0.04 * attackAngle,
    };
  });

describe("launch monitor flexible analysis", () => {
  it("uses standards-conformant SHA-256 fingerprints", () => {
    expect(sha256Text("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", // pragma: allowlist secret
    );
  });

  it("reports traceable correlation and multivariable OLS results", () => {
    const result = analyzeLaunchMonitorData(rows(), {
      outcome: "ball_speed",
      predictors: ["club_speed", "attack_angle"],
      analysisMode: "comprehensive",
      correlationMethod: "pearson",
      missingPolicy: "pairwise",
      confidenceLevel: 0.95,
      minSamples: 10,
    });

    expect(result.contractVersion).toBe("1.0.0");
    expect(result.dataset.rowCount).toBe(80);
    expect(result.dataset.monitorVendors).toEqual(["FlightScope", "TrackMan"]);
    expect(result.dataset.fingerprintSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.regression?.rSquared).toBeGreaterThan(0.999);
    expect(result.regression?.coefficients.club_speed.estimate).toBeCloseTo(
      1.48,
      8,
    );
    expect(result.correlations.every((item) => item.sampleCount === 80)).toBe(
      true,
    );
  });

  it("keeps pairwise counts and grouped results explicit", () => {
    const data = rows();
    data[0].attack_angle = null;
    data[1].club_speed = null;
    const result = analyzeLaunchMonitorData(data, {
      outcome: "ball_speed",
      predictors: ["club_speed", "attack_angle"],
      analysisMode: "correlation",
      correlationMethod: "spearman",
      missingPolicy: "pairwise",
      groupBy: "monitor_vendor",
      confidenceLevel: 0.95,
      minSamples: 10,
    });

    expect(
      Object.fromEntries(
        result.correlations.map((item) => [item.predictor, item.sampleCount]),
      ),
    ).toEqual({ club_speed: 79, attack_angle: 79 });
    expect(result.groups.map((group) => group.groupValue)).toEqual([
      "FlightScope",
      "TrackMan",
    ]);
  });

  it("blocks aggregate regression and pooled vendor-specific source fields", () => {
    expect(() =>
      analyzeLaunchMonitorData(
        rows().map((row) => ({
          ...row,
          observation_kind: "aggregate",
        })),
        {
          outcome: "ball_speed",
          predictors: ["club_speed"],
          analysisMode: "regression",
          correlationMethod: "pearson",
          missingPolicy: "listwise",
          confidenceLevel: 0.95,
          minSamples: 10,
          allowAggregate: true,
        },
      ),
    ).toThrow(/Aggregate observations cannot enter regression/);

    expect(() =>
      analyzeLaunchMonitorData(
        rows().map((row, index) => ({
          ...row,
          "source::temperature": 20 + index * 0.01,
        })),
        {
          outcome: "ball_speed",
          predictors: ["source::temperature"],
          analysisMode: "correlation",
          correlationMethod: "pearson",
          missingPolicy: "pairwise",
          confidenceLevel: 0.95,
          minSamples: 10,
        },
      ),
    ).toThrow(/source fields.*multiple monitors/);
  });

  it("parses quoted CSV and JSON without dropping source columns", () => {
    const csv = 'Shot,Vendor,Comment,Speed\r\n1,TrackMan,"wind, left",70\r\n';
    expect(parseLaunchMonitorFile("shots.csv", csv)).toEqual([
      { Shot: 1, Vendor: "TrackMan", Comment: "wind, left", Speed: 70 },
    ]);
    expect(
      parseLaunchMonitorFile("shots.json", JSON.stringify([{ custom: 4.2 }])),
    ).toEqual([{ custom: 4.2 }]);
  });

  it("uses strict decimal cells and rejects malformed retained record shapes", () => {
    expect(parseLaunchMonitorFile(
      "shots.csv", "x,y\n0x10,1\n0b10,2\n١٢,3\n1.5e2,4\n",
    )).toEqual([
      { x: "0x10", y: 1 }, { x: "0b10", y: 2 },
      { x: "١٢", y: 3 }, { x: 150, y: 4 },
    ]);
    expect(() => parseLaunchMonitorFile("shots.csv", "x,y\n1,2,3\n"))
      .toThrow(/match the header width/);
    expect(() => parseLaunchMonitorFile("shots.json", '[{"x":{"nested":1}}]'))
      .toThrow(/portable finite scalars/);
    expect(() => parseLaunchMonitorFile("shots.json", '[{"x":1e1000}]'))
      .toThrow(/portable finite scalars/);
    expect(() => parseLaunchMonitorFile("shots.json", '[{"x":9007199254740992}]'))
      .toThrow(/portable finite scalars/);
    expect(() => parseLaunchMonitorFile("shots.json", '[{"x":1e20}]'))
      .toThrow(/portable finite scalars/);
    expect(() => parseLaunchMonitorFile("shots.json", '[{"":1,"y":2}]'))
      .toThrow(/field names must be non-empty/);
    expect(() => parseLaunchMonitorFile("shots.txt", "x,y\n1,2\n"))
      .toThrow(/supports CSV and JSON/);
  });

  it("shares CSV coercion and blank-line policy with the Python reader", () => {
    expect(parseLaunchMonitorFile(
      "shots.csv", "x,y,label\n\n1,2, alpha \n1.5e2,3,\n0x10,4,hex\n",
    )).toEqual([
      { x: 1, y: 2, label: "alpha" },
      { x: 150, y: 3, label: null },
      { x: "0x10", y: 4, label: "hex" },
    ]);
  });
});
