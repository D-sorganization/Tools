import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { VariationDatasetTs } from "../model/variation";
import { CATEGORY_LAUNCH } from "../model/variation";
import { VariationResults } from "./VariationResults";

const { scatterSelections } = vi.hoisted(() => ({
  scatterSelections: [] as Array<number | null>,
}));

vi.mock("./VariationScatter", () => ({
  VariationScatter: ({ selectedTrialIndex }: { selectedTrialIndex: number | null }) => {
    scatterSelections.push(selectedTrialIndex);
    return <output aria-label="Observed linked scatter selection">
      {selectedTrialIndex ?? "none"}
    </output>;
  },
}));

beforeEach(() => {
  scatterSelections.length = 0;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const dataset = (nRuns: number): VariationDatasetTs => ({
  plan: {
    mode: "launch",
    baseVariables: {},
    noise: [{
      variableKey: `${CATEGORY_LAUNCH}.ball_speed_mph`,
      distribution: "normal",
      scale: 1,
      lower: null,
      upper: null,
    }],
    nRuns,
    seed: 7,
    flightModel: "waterloo_penner",
  },
  inputNames: [`${CATEGORY_LAUNCH}.ball_speed_mph`],
  inputs: Array.from({ length: nRuns }, (_, index) => [150 + index]),
  outputNames: ["carry_m", "lateral_m", "apex_m", "flight_time_s"],
  outputs: Array.from({ length: nRuns }, (_, index) => [
    220 + index,
    index,
    30 + index,
    5 + index / 10,
  ]),
  success: Array.from({ length: nRuns }, () => true),
});

describe("VariationResults linked trial lifecycle", () => {
  it("clears a selected trial when a smaller rerun replaces the result", () => {
    const view = render(
      <VariationResults
        dataset={dataset(3)}
        sensitivity={null}
        distanceUnit="m"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Select matrix trial 2" }));
    expect(screen.getByRole("status", { name: "Observed linked scatter selection" }))
      .toHaveTextContent("1");
    const renderCount = scatterSelections.length;

    view.rerender(
      <VariationResults
        dataset={dataset(2)}
        sensitivity={null}
        distanceUnit="m"
      />,
    );

    expect(scatterSelections.slice(renderCount)).toEqual([null]);
    expect(screen.getByRole("status", { name: "Observed linked scatter selection" }))
      .toHaveTextContent("none");
  });
});
