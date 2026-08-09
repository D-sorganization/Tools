import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { runCapabilityOptimization, type CapabilityRunOutput } from "../model/capabilityRun";
import type { CapabilityRunner } from "../model/capabilityWorkerClient";
import { CapabilityOptimizationPanel } from "./CapabilityOptimizationPanel";

describe("CapabilityOptimizationPanel", () => {
  it("runs a bounded workflow and exposes alternatives plus raw diagnostics", async () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const runner: CapabilityRunner = (document, onProgress) => ({
      promise: Promise.resolve(runCapabilityOptimization(document, onProgress)),
      cancel: vi.fn(),
    });
    render(<CapabilityOptimizationPanel runner={runner} />);
    fireEvent.change(screen.getByLabelText("Candidate budget"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("Trials per candidate"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("Alternatives retained"), { target: { value: "1" } });

    fireEvent.click(screen.getByRole("button", { name: "Run optimization" }));

    expect(await screen.findByText(/Attempted 2; complete/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Ranked capability alternatives" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Capability raw observation rows" })).toBeInTheDocument();
    expect(screen.getByText(/Paired finite 2\/2/)).toBeInTheDocument();
    const axis = screen.getByLabelText("Horizontal axis");
    const labels = [...axis.querySelectorAll("option")].map(({ textContent }) => textContent);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it("shows boundary validation errors without dispatching a worker", () => {
    const runner = vi.fn() as unknown as CapabilityRunner;
    render(<CapabilityOptimizationPanel runner={runner} />);
    fireEvent.change(screen.getByLabelText("Ball speed center"), { target: { value: "0" } });

    fireEvent.click(screen.getByRole("button", { name: "Run optimization" }));

    expect(screen.getByRole("alert")).toHaveTextContent("ballSpeedMps");
    expect(runner).not.toHaveBeenCalled();
  });

  it("exposes integration settings that are persisted and run", () => {
    render(<CapabilityOptimizationPanel />);

    expect(screen.getByLabelText("Maximum flight time")).toHaveValue(10);
    expect(screen.getByLabelText("Trajectory sample interval")).toHaveValue(0.01);
  });

  it("allows clearing a numeric field before entering a negative value", () => {
    const runner = vi.fn<CapabilityRunner>(() => ({
      promise: new Promise<CapabilityRunOutput>(() => undefined),
      cancel: vi.fn(),
    }));
    render(<CapabilityOptimizationPanel runner={runner} />);
    const tilt = screen.getByLabelText("Fixed spin-axis tilt (+ fade/right)");

    fireEvent.change(tilt, { target: { value: "" } });
    expect(tilt).toHaveValue(null);
    fireEvent.change(tilt, { target: { value: "-" } });
    expect(tilt).toHaveValue(null);
    fireEvent.change(tilt, { target: { value: "-3.5" } });
    fireEvent.blur(tilt);
    fireEvent.click(screen.getByRole("button", { name: "Run optimization" }));

    expect(tilt).toHaveValue(-3.5);
    expect(runner.mock.calls[0][0].evaluatorConfig.spinDefaults[0].spinAxisTiltDeg)
      .toBe(-3.5);
  });
});
