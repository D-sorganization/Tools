import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CATEGORY_SWING, type VariationPlanTs } from "../model/variation";
import { runSwingVariation } from "../model/variationSwingEnsemble";
import { VariationArcOverlay } from "./VariationArcOverlay";

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VariationArcOverlay dispersion controls", () => {
  it("enables confidence only for volume and explains adequacy semantics", () => {
    const yaw = `${CATEGORY_SWING}.yaw_deg`;
    const plan: VariationPlanTs = {
      mode: "swing",
      baseVariables: { [yaw]: 0 },
      noise: [{
        variableKey: yaw,
        distribution: "uniform",
        scale: 0.2,
        lower: null,
        upper: null,
      }],
      nRuns: 5,
      seed: 41,
      flightModel: "waterloo_penner",
    };
    render(<VariationArcOverlay
      ensemble={runSwingVariation(plan)}
      selectedTrialIndex={null}
      onSelectedTrialChange={() => undefined}
    />);

    const confidence = screen.getByLabelText("Dispersion confidence percent");
    expect(confidence).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Dispersion metric"), {
      target: { value: "confidence-ellipsoid-volume" },
    });
    expect(confidence).toBeEnabled();
    expect(screen.getByText(/Gaussian position-content region/)).toBeInTheDocument();
    expect(screen.getByText(/not a confidence region for the mean/)).toBeInTheDocument();
    expect(screen.getByText(/Sparse 2σ principal-axis glyphs/)).toBeInTheDocument();

    fireEvent.change(confidence, { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Minimum quiet duration seconds"), {
      target: { value: "-1" },
    });
    fireEvent.change(screen.getByLabelText("Minimum quiet samples"), {
      target: { value: "1.5" },
    });
    expect(confidence).toHaveValue(95);
    expect(screen.getByLabelText("Minimum quiet duration seconds")).toHaveValue(0);
    expect(screen.getByLabelText("Minimum quiet samples")).toHaveValue(1);
  });
});
