import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { FlightExplorerPanel } from "./FlightExplorerPanel";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";

function FlightExplorerHarness() {
  const [target, setTarget] = useState(() => spatialTargetFromRegion(DEFAULT_TARGET));
  return <FlightExplorerPanel spatialTarget={target} onSpatialTargetChange={setTarget} />;
}

beforeAll(() => {
  const ctx: unknown = new Proxy(function () {} as object, {
    get: (_target, prop) =>
      prop === "measureText" ? () => ({ width: 0 }) : () => ctx,
    set: () => true,
    apply: () => ctx,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    ctx as CanvasRenderingContext2D,
  );
});

describe("FlightExplorerPanel input editing", () => {
  it("applies one canonical spatial target to its summary and flight plots", () => {
    render(<FlightExplorerHarness />);

    const downrange = screen.getByLabelText("Target downrange m");
    fireEvent.change(downrange, { target: { value: "180" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));

    expect(screen.getByRole("status", { name: "Current spatial target" }))
      .toHaveTextContent("180.0 m downrange");
    expect(screen.getByLabelText("Flight top-down view (lateral vs carry)"))
      .toHaveAttribute("aria-description", expect.stringContaining("180.0 m downrange"));
    expect(screen.getByLabelText("Flight side profile (height vs carry)"))
      .toHaveAttribute("aria-description", expect.stringContaining("180.0 m downrange"));
    expect(screen.getByRole("status", { name: "Spatial target assessment" }))
      .toHaveTextContent("Run Flight to evaluate this target");

    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.getByRole("status", { name: "Spatial target assessment" }))
      .toHaveTextContent(/Target (accepted|missed)/);
  });

  it("uses launch-monitor terminology and exposes a working definition", () => {
    render(<FlightExplorerHarness />);

    expect(screen.queryByText("Launch Azimuth")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Launch Direction")).toBeInTheDocument();
    expect(screen.getByLabelText("Launch Direction Convention")).toHaveValue(
      "app_native",
    );
    const foresight = screen.getByRole("option", {
      name: "Foresight-Comparable (Sign Unavailable)",
    });
    expect(foresight).toBeDisabled();
    expect(foresight).toHaveAttribute(
      "title",
      expect.stringMatching(/public sign convention/i),
    );
    expect(screen.getByTestId("direction-sign-example")).toHaveTextContent(
      "0° = straight · + = right of the target line",
    );
    fireEvent.click(screen.getByLabelText("Explain Launch Direction"));
    expect(screen.getByText(/positive values start right of the target line/i)).toBeVisible();
  });

  it("accepts and preserves a negative spin-axis tilt", () => {
    render(<FlightExplorerHarness />);
    const input = screen.getByLabelText("Spin-Axis Tilt") as HTMLInputElement;

    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "-" } });
    expect(input.value).toBe("-");
    fireEvent.change(input, { target: { value: "-12.5" } });
    fireEvent.blur(input);

    expect(input.value).toBe("-12.5");
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("runs a visible common-input no-wind versus selected-wind comparison", () => {
    render(<FlightExplorerHarness />);

    expect(screen.getByLabelText(/Wind 10.0 miles per hour from 0.0 degrees/))
      .toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", {
      name: "Compare No Wind and Selected Wind",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));

    expect(screen.getByLabelText("Wind effect deltas")).toHaveTextContent(
      "Selected Wind Minus No Wind",
    );
    expect(screen.getByLabelText("Flight side profile (height vs carry)"))
      .toBeInTheDocument();
  });
});
