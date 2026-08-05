import { describe, expect, it } from "vitest";

import { inPlaneGravity } from "./simulation";
import { simulateTriplePendulum } from "./triplePendulum";

describe("triple pendulum parity", () => {
  it("matches the canonical Python RK4 state at 500 ms", () => {
    const gravity = inPlaneGravity(0, (-45 * Math.PI) / 180, 0);
    const state = simulateTriplePendulum(gravity, 0.001, 500)[500];
    const expected = [
      -0.05269620677267223,
      -0.2570049704796624,
      -1.2559403929062531,
      3.695122645710546,
      5.69035508536069,
      4.157255701636642,
    ];
    state.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 9));
  });
});
