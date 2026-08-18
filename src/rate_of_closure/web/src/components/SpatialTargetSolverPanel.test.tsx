import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "../model/spatialTarget";
import { SpatialTargetSolverPanel } from "./SpatialTargetSolverPanel";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";

afterEach(cleanup);

describe("SpatialTargetSolverPanel", () => {
  it("fails closed for an aerial target instead of solving a zero-elevation projection", () => {
    const target = createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([140, 24, -3], "app"),
      tolerance: sphereTolerance(4),
      elevationSource: "absolute",
    });
    render(<SpatialTargetSolverPanel onApply={() => undefined}
      spatialTarget={target} />);
    expect(screen.queryByRole("button", { name: "Optimize to Target" }))
      .not.toBeInTheDocument();
    expect(screen.getByRole("status", {
      name: "Solver spatial target compatibility",
    })).toHaveTextContent(/elevation was not coerced to zero/i);
  });

  it("marks target-derived results stale and disables Apply after target incompatibility", () => {
    const landing = spatialTargetFromRegion(DEFAULT_TARGET);
    const aerial = createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([140, 24, -3], "app"),
      tolerance: sphereTolerance(4),
      elevationSource: "absolute",
    });
    const view = render(<SpatialTargetSolverPanel onApply={() => undefined}
      spatialTarget={landing} />);
    fireEvent.click(screen.getByRole("button", { name: "Optimize to Target" }));
    expect(screen.getByRole("button", { name: "Apply to Scenario" })).toBeEnabled();

    view.rerender(<SpatialTargetSolverPanel onApply={() => undefined}
      spatialTarget={aerial} />);
    expect(screen.getByRole("button", { name: "Apply to Scenario" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(/result is stale/i);
    expect(screen.queryByRole("button", { name: "Optimize to Target" }))
      .not.toBeInTheDocument();
  });

  it("marks a solved result stale when a goal changes", () => {
    render(<SpatialTargetSolverPanel onApply={() => undefined}
      spatialTarget={spatialTargetFromRegion(DEFAULT_TARGET)} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Solver" }));
    expect(screen.getByRole("button", { name: "Apply to Scenario" })).toBeEnabled();
    fireEvent.click(screen.getByTitle("Enable the Ball Speed goal"));
    expect(screen.getByRole("button", { name: "Apply to Scenario" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(/goals, variables, or the canonical target changed/i);
  });
});
