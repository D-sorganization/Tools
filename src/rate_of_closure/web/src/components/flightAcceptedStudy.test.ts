import { describe, expect, it } from "vitest";

import { buildAcceptedFlightStudy, type FlightStudyContext } from "./flightAcceptedStudy";
import {
  compareWind, directLaunch, exploreFlight, type FlightExplorationTs,
} from "../model/flightExplorer";
import { BALL_POSITION } from "../model/simulation";
import { meteorologicalWind } from "../model/wind";

const context: FlightStudyContext = {
  entryMode: "direct", ballSpeedMph: 100, launchAngleDeg: 10,
  launchDirectionDeg: 0, spinRpm: 2000, spinAxisTiltDeg: 0,
  directionConvention: "app_native", windScenario: null, model: "waterloo_penner",
  kernelRevision: "web-rk4-10ms-sampled-v1",
};

function exploration(): FlightExplorationTs {
  const speedMps = 100 / 2.236936292054402;
  const launchRad = 10 * Math.PI / 180;
  const launchVelocity: [number, number, number] = [
    speedMps * Math.cos(launchRad), speedMps * Math.sin(launchRad), 0,
  ];
  return {
    points: [
      { time: 0, position: [...BALL_POSITION], velocity: launchVelocity },
      { time: 0.5, position: [1, BALL_POSITION[1] + 1, 0.25], velocity: [2, 0, 0.5] },
      { time: 1, position: [2, BALL_POSITION[1], 0.5], velocity: [1, -1, 0] },
    ],
    metrics: {
      ballSpeedMph: 100, launchAngleDeg: 10, launchDirectionDeg: 0,
      launchAzimuthDeg: 0, spinRpm: 2000, carryM: Math.hypot(2, 0.5), maxHeightM: 1,
      flightTimeS: 1, landingAngleDeg: 45, lateralM: 0.5,
    },
    execution: {
      model: "waterloo_penner", kernelRevision: "web-rk4-10ms-sampled-v1",
      windScenario: null,
      launch: directLaunch({
        ballSpeedMph: 100, launchAngleDeg: 10, launchDirectionDeg: 0,
        spinRpm: 2000, spinAxisTiltDeg: 0,
      }),
    },
  };
}

describe("accepted flight study", () => {
  it("deep-snapshots complete validated evidence before publication", () => {
    const input = exploration();
    const accepted = buildAcceptedFlightStudy(1, context, input, null);
    input.points[2].position[0] = 99;
    input.metrics.carryM = 99;
    expect(accepted.plan.rawSample(2).downrangeM).toBe(2);
    expect(accepted.exploration.metrics.carryM).toBe(Math.hypot(2, 0.5));
    expect(accepted.contextLabel).toContain("kernel web-rk4-10ms-sampled-v1");
  });

  it.each(["carryM", "maxHeightM", "flightTimeS", "lateralM"] as const)(
    "rejects summary/raw disagreement for %s",
    (field) => {
      const input = exploration();
      input.metrics[field] += field === "maxHeightM" ? -2 : 1;
      expect(() => buildAcceptedFlightStudy(1, context, input, null)).toThrow(
        /disagrees|below sampled/,
      );
    },
  );

  it("rejects malformed points before any accepted bundle exists", () => {
    const input = exploration();
    input.points[1].position[1] = Number.NaN;
    expect(() => buildAcceptedFlightStudy(1, context, input, null)).toThrow(/finite/);
  });

  it("binds canonical origin, landing floor, and all retained heights", () => {
    const translated = exploration();
    translated.points.forEach((point) => {
      point.position[0] += 10;
      point.position[1] += 2;
      point.position[2] += 3;
    });
    expect(() => buildAcceptedFlightStudy(1, context, translated, null)).toThrow(
      /launch downrange position/,
    );
    const airborne = exploration();
    airborne.points[airborne.points.length - 1].position[1] += 1;
    expect(() => buildAcceptedFlightStudy(1, context, airborne, null)).toThrow(
      /landing height/,
    );
    const belowGround = exploration();
    belowGround.points[1].position[1] = -1;
    expect(() => buildAcceptedFlightStudy(1, context, belowGround, null)).toThrow(
      /ground plane/,
    );
  });

  it("rejects implausible hidden apex evidence and wind mislabeled as calm", () => {
    const huge = exploration();
    huge.metrics.maxHeightM = 9_999;
    expect(() => buildAcceptedFlightStudy(1, context, huge, null)).toThrow(
      /sampled-apex allowance/,
    );
    const scenario = meteorologicalWind(4, 90);
    const windy = exploreFlight({
      ...directLaunch({
        ballSpeedMph: 100, launchAngleDeg: 10, launchDirectionDeg: 0,
        spinRpm: 2000, spinAxisTiltDeg: 0,
      }),
      windScenario: scenario,
    });
    expect(() => buildAcceptedFlightStudy(1, context, windy, null)).toThrow(/provenance/);
  });

  it("rejects an undefined top-level execution wind declaration", () => {
    const input = exploration();
    input.execution = { ...input.execution, windScenario: undefined } as unknown as
      FlightExplorationTs["execution"];
    expect(() => buildAcceptedFlightStudy(1, context, input, null)).toThrow(
      /provenance is incomplete/,
    );
  });

  it("rejects a different spin-axis launch under the same displayed summary", () => {
    const input = exploration();
    input.execution = {
      ...input.execution,
      launch: directLaunch({
        ballSpeedMph: 100, launchAngleDeg: 10, launchDirectionDeg: 0,
        spinRpm: 2000, spinAxisTiltDeg: 10,
      }),
    };
    expect(() => buildAcceptedFlightStudy(1, context, input, null)).toThrow(/fingerprint/);
  });

  it.each(["launchAzimuthDeg", "landingAngleDeg"] as const)(
    "rejects forged derived %s before publication", (field) => {
      const input = exploration();
      input.metrics[field] += 5;
      expect(() => buildAcceptedFlightStudy(1, context, input, null)).toThrow();
    },
  );

  it("accepts a real lateral production flight and a cohesive crosswind comparison", () => {
    const lateralContext = { ...context, launchDirectionDeg: 3, spinAxisTiltDeg: 5 };
    const launch = directLaunch({
      ballSpeedMph: 100, launchAngleDeg: 10, launchDirectionDeg: 3,
      spinRpm: 2000, spinAxisTiltDeg: 5,
    });
    const calm = exploreFlight(launch);
    expect(buildAcceptedFlightStudy(2, lateralContext, calm, null).plan.rawCount).toBeGreaterThan(2);
    const scenario = meteorologicalWind(4, 90);
    const comparison = compareWind(launch, scenario);
    const windyContext = { ...lateralContext, windScenario: scenario };
    expect(buildAcceptedFlightStudy(3, windyContext, comparison.wind, comparison).comparison)
      .not.toBeNull();

    const forged = { ...comparison, deltas: { ...comparison.deltas, carryM: 999 } };
    expect(() => buildAcceptedFlightStudy(4, windyContext, comparison.wind, forged)).toThrow();
    expect(() => buildAcceptedFlightStudy(4, { ...windyContext, ballSpeedMph: Number.NaN },
      comparison.wind, comparison)).toThrow();
  });
});
