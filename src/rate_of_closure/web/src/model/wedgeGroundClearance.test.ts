import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import { DEFAULT_SCENARIO, solve } from "./impact";
import { runSimulation, type SimulationInput } from "./simulation";
import { wedgeGroundClearance } from "./wedgeGroundClearance";

const scenario = {
  ...DEFAULT_SCENARIO,
  clubheadSpeedMph: 30,
  lieAngleDeg: 64,
  omegaPlaneDps: 0,
  omegaShaftDps: 1307,
  comToFaceMm: 20,
};

const input: SimulationInput = {
  sourceKind: "manual",
  clubheadSpeedMph: scenario.clubheadSpeedMph,
  omegaDps: solve(scenario).omegaDps,
  loftDeg: 46,
  impactOffsetToeMm: 0,
  impactOffsetHighMm: 0,
  planeYawDeg: 0,
  planeSideTiltDeg: -45,
  planeForwardTiltDeg: 0,
  impactTimeS: 0.03,
  swingDurationS: 1.5,
};

describe("wedge ground-clearance parity", () => {
  it("matches the Python representative pitching-wedge fixture", () => {
    const result = wedgeGroundClearance(
      runSimulation(input), scenario, getClub("Pitching Wedge"),
    );

    expect(result?.format).toBe("upstreamdrift.wedge-ground-clearance/v1");
    expect(result?.sequence).toBe("ball_first");
    expect(result?.firstGroundContact?.feature).toBe("leading_edge_toe");
    expect(result?.firstGroundContact?.timeS).toBeCloseTo(0.032713516604961065, 8);
    expect(result?.metrics.leadingEdgeClearanceAtBallM).toBeCloseTo(0.0010738823680660724, 12);
    expect(result?.metrics.groundAfterBallTimeMarginS).toBeCloseTo(0.0027135166049610665, 8);
    expect(result?.metrics.pathProjectedEffectiveBounceDegAtBall).toBeCloseTo(10.015034183017084, 10);
    expect(result?.metrics.referenceAoaDegAtBall).toBeCloseTo(-1.5459393262843542, 10);
    expect(result?.envelope).toHaveLength(481);
  });

  it("preserves miss semantics without fabricating ball metrics", () => {
    const run = { ...runSimulation(input), impactTimeS: null };
    const result = wedgeGroundClearance(run, scenario, getClub("Pitching Wedge"));

    expect(result?.sequence).toBe("ground_only_miss");
    expect(result?.ballContactTimeS).toBeNull();
    expect(result?.metrics.leadingEdgeClearanceAtBallM).toBeNull();
  });

  it("makes no wedge-ground claim for a driver", () => {
    expect(wedgeGroundClearance(
      runSimulation(input), scenario, getClub("Driver 10.5°"),
    )).toBeNull();
  });
});
