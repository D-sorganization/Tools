import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { VariationDatasetTs } from "../model/variation";
import { LandingCanvas } from "./VariationLanding";

const dataset = (): VariationDatasetTs => ({
  plan: {
    mode: "launch",
    baseVariables: {},
    noise: [{
      variableKey: "swing_sim.launch.ball_speed_mph",
      distribution: "normal",
      scale: 1,
      lower: null,
      upper: null,
    }],
    nRuns: 4,
    seed: 1,
    flightModel: "waterloo_penner",
  },
  inputNames: ["swing_sim.launch.ball_speed_mph"],
  inputs: [[150], [151], [152], [153]],
  outputNames: ["carry_m", "lateral_m"],
  outputs: [[100, 5], [110, null], [120, Number.POSITIVE_INFINITY], [null, 8]],
  success: [true, true, true, true],
});

describe("LandingCanvas missing-data treatment", () => {
  beforeEach(() => {
    const context: unknown = new Proxy(function () {} as object, {
      get: () => () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });

  it("reports only paired finite carry/lateral rows as plotted landings", () => {
    render(<LandingCanvas dataset={dataset()} />);

    expect(screen.getByRole("status")).toHaveTextContent("Evaluated landings: 1/4");
  });
});
