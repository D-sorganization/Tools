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
    const surfaces = screen.getByLabelText("Show confidence ellipsoid surfaces");
    expect(confidence).toBeDisabled();
    expect(surfaces).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Dispersion metric"), {
      target: { value: "confidence-ellipsoid-volume" },
    });
    expect(confidence).toBeEnabled();
    expect(surfaces).toBeEnabled();
    fireEvent.click(surfaces);
    expect(screen.getByLabelText("Arc visualization legend")).toHaveTextContent(
      /Gaussian position-content ellipsoid \(not mean CI\)/,
    );
    expect(screen.getByText(/Gaussian position-content region/)).toBeInTheDocument();
    expect(screen.getByText(/not a confidence region for the mean/)).toBeInTheDocument();
    expect(screen.getByText(/Sparse yellow 2σ principal-axis glyphs/)).toBeInTheDocument();
    expect(screen.getByText(/selection criteria are retained in the plot definition/))
      .toBeInTheDocument();
    expect(screen.queryByText(/adequacy and ranked intervals are retained in the plot definition/))
      .not.toBeInTheDocument();

    const camera = screen.getByLabelText("Arc camera state");
    const initialYaw = camera.getAttribute("data-yaw-deg");
    const canvas = screen.getByRole("img", { name: /Interactive all-trial swing arcs/ });
    fireEvent.keyDown(canvas, { key: "ArrowRight" });
    expect(camera).not.toHaveAttribute("data-yaw-deg", initialYaw);
    fireEvent.click(screen.getByRole("button", { name: "Reset View" }));
    expect(camera).toHaveAttribute("data-yaw-deg", initialYaw);
    expect(surfaces).toBeChecked();

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

  it("retains localized source identity beside confidence-surface controls", () => {
    const shoulder = `${CATEGORY_SWING}.shoulder_commanded_torque_offset_nm`;
    const plan: VariationPlanTs = {
      mode: "swing",
      baseVariables: { [shoulder]: 3 },
      noise: [{
        variableKey: shoulder,
        specId: "integration.shoulder",
        distribution: "uniform",
        scale: 0.1,
        lower: null,
        upper: null,
        timeWindowS: [0.001, 0.003],
        pointIds: ["joint.shoulder"],
      }],
      nRuns: 2,
      seed: 41,
      flightModel: "waterloo_penner",
    };
    render(<VariationArcOverlay
      ensemble={runSwingVariation(plan)}
      selectedTrialIndex={null}
      onSelectedTrialChange={() => undefined}
    />);

    expect(screen.getByRole("option", {
      name: /integration\.shoulder.*joint\.shoulder.*\[0\.001, 0\.003\) s.*N\*m/,
    })).toHaveValue(shoulder);
    expect(screen.getByLabelText("Show confidence ellipsoid surfaces")).toBeDisabled();
  });
});
