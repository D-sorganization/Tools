import { describe, expect, it, vi } from "vitest";

import type { WedgeGroundClearancePayloadTs } from "../model/wedgeGroundClearance";
import { drawWedgeGroundOverlay } from "./wedgeGroundOverlay";

const payload: WedgeGroundClearancePayloadTs = {
  format: "upstreamdrift.wedge-ground-clearance/v1",
  frameId: "ground_frame:x_target,y_up,z_right",
  units: { angle: "deg", angularVelocity: "rad/s", length: "m", time: "s", velocity: "m/s" },
  sequence: "ball_first", ballContactTimeS: 0.03,
  firstGroundContact: { timeS: 0.033, feature: "leading_edge_toe", worldPointM: [1, 0, 0],
    normalVelocityMps: -1, tangentialVelocityMps: [1, 0, 0],
    poseHeadToGround: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
  metrics: { bounceUtilizationMarginDeg: 8, deliveredBounceDegAtBall: 10,
    groundAfterBallTimeMarginS: 0.003, leadingEdgeClearanceAtBallM: 0.001,
    minimumPreBallClearanceM: 0.001, pathProjectedEffectiveBounceDegAtBall: 10,
    referenceAoaDegAtBall: -2, soleEntryMarginM: 0.005 },
  lowPoint: { feature: "leading_edge_toe", timeS: 0.04, worldPointM: [1.1, -0.01, 0] },
  envelope: [
    { timeS: 0.03, minimumClearanceM: 0.001, feature: "leading_edge_toe", worldPointM: [0.9, 0.001, 0] },
    { timeS: 0.04, minimumClearanceM: -0.01, feature: "leading_edge_toe", worldPointM: [1.1, -0.01, 0] },
  ], geometryBasis: "test", provenance: "test", limitations: "test",
};

describe("wedge ground overlay", () => {
  it("labels ball, ground, envelope, and current clearance", () => {
    const fillText = vi.fn();
    const context = new Proxy({ fillText } as unknown as CanvasRenderingContext2D, {
      get: (target, property) => property in target ? target[property as keyof typeof target] : vi.fn(),
      set: () => true,
    });
    drawWedgeGroundOverlay(context, payload, 0.03, ([x, y]) => [x * 10, y * 10]);

    expect(fillText.mock.calls.flat()).toEqual(expect.arrayContaining([
      "Ball Contact", "Ground Contact", "Wedge Sole Envelope", "1.0 mm clearance",
    ]));
  });
});
