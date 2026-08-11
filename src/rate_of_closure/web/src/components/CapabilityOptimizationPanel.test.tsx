import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { runCapabilityOptimization, type CapabilityRunOutput } from "../model/capabilityRun";
import type { CapabilityRunner } from "../model/capabilityWorkerClient";
import { CapabilityOptimizationPanel } from "./CapabilityOptimizationPanel";
import { defaultCapabilityWorkflowInputs } from "../model/capabilityWorkflow";

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

    expect(screen.getByLabelText("Maximum flight time")).toHaveValue("10");
    expect(screen.getByLabelText("Trajectory sample interval")).toHaveValue("0.01");
  });

  it("allows normal keyboard entry of a negative decimal", async () => {
    const user = userEvent.setup();
    const runner = vi.fn<CapabilityRunner>(() => ({
      promise: new Promise<CapabilityRunOutput>(() => undefined),
      cancel: vi.fn(),
    }));
    render(<CapabilityOptimizationPanel runner={runner} />);
    const tilt = screen.getByLabelText("Fixed spin-axis tilt (+ fade/right)");

    expect(tilt).toHaveAttribute("type", "text");
    expect(tilt).toHaveAttribute("inputmode", "decimal");
    await user.clear(tilt);
    await user.type(tilt, "-3.5");
    await user.tab();
    await user.click(screen.getByRole("button", { name: "Run optimization" }));

    expect(tilt).toHaveValue("-3.5");
    expect(runner.mock.calls[0][0].evaluatorConfig.spinDefaults[0].spinAxisTiltDeg)
      .toBe(-3.5);
  });

  it("uses the workspace input authority and invalidates stale output on replacement", async () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const runner: CapabilityRunner = (document, onProgress) => ({
      promise: Promise.resolve(runCapabilityOptimization(document, onProgress)),
      cancel: vi.fn(),
    });
    const initial = {
      ...defaultCapabilityWorkflowInputs(),
      candidateBudget: 1,
      ensembleSize: 1,
      alternativesCount: 1,
    };
    const changed = { ...initial, targetDistanceM: 199 };
    const onInputsChange = vi.fn();
    const { rerender } = render(
      <CapabilityOptimizationPanel runner={runner} inputs={initial}
        onInputsChange={onInputsChange} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Run optimization" }));
    expect(await screen.findByRole("table", {
      name: "Ranked capability alternatives",
    })).toBeInTheDocument();

    rerender(<CapabilityOptimizationPanel runner={runner} inputs={changed}
      onInputsChange={onInputsChange} />);

    expect(screen.queryByRole("table", {
      name: "Ranked capability alternatives",
    })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Target distance"), {
      target: { value: "201" },
    });
    expect(onInputsChange).toHaveBeenCalled();
  });
});
