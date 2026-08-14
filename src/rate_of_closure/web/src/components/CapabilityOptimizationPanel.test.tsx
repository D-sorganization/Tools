import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { runCapabilityOptimization, type CapabilityRunOutput } from "../model/capabilityRun";
import type { CapabilityRunner } from "../model/capabilityWorkerClient";
import { CapabilityOptimizationPanel } from "./CapabilityOptimizationPanel";
import {
  buildCapabilityWorkflow,
  capabilityWorkflowFromJson,
  capabilityWorkflowToJson,
  defaultCapabilityWorkflowInputs,
  type CapabilityWorkflowDocument,
} from "../model/capabilityWorkflow";

const customWorkflow = (): CapabilityWorkflowDocument => {
  const payload = JSON.parse(capabilityWorkflowToJson(
    buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
  ));
  payload.profile.provenance = "measured/session-42";
  payload.profile.confidence = 0.71;
  payload.profile.clubs[0].provenance = "fit/driver-42";
  payload.profile.clubs[0].confidence = 0.63;
  payload.profile.clubs[0].matrix = [
    [1, 0.2, 0], [0.2, 1, 0.1], [0, 0.1, 1],
  ];
  payload.profile.clubs[0].parameters[0].bias = 0.4;
  payload.request.problem_id = "custom-problem-42";
  payload.request.cvar_alpha = 0.83;
  payload.request.minimum_success_fraction = 0.64;
  payload.request.target.kind = "fairway";
  payload.request.target.band_half_length_m = 21;
  payload.request.target.half_width_m = 8;
  payload.evaluator_config.spin_defaults[0].provenance = "measured/spin-42";
  return capabilityWorkflowFromJson(JSON.stringify(payload));
};

const noncanonicalWorkflow = (kind: "mph" | "covariance") => {
  const payload = JSON.parse(capabilityWorkflowToJson(
    buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
  ));
  if (kind === "mph") payload.profile.clubs[0].parameters[0].unit = "mph";
  else payload.profile.clubs[0].matrix_kind = "covariance";
  return capabilityWorkflowFromJson(JSON.stringify(payload));
};

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
    const initial = buildCapabilityWorkflow({
      ...defaultCapabilityWorkflowInputs(), candidateBudget: 1,
      ensembleSize: 1, alternativesCount: 1,
    });
    const changed = buildCapabilityWorkflow({
      ...defaultCapabilityWorkflowInputs(), candidateBudget: 1,
      ensembleSize: 1, alternativesCount: 1, targetDistanceM: 199,
    });
    const onWorkflowChange = vi.fn();
    const { rerender } = render(
      <CapabilityOptimizationPanel runner={runner} workflow={initial}
        onWorkflowChange={onWorkflowChange} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Run optimization" }));
    expect(await screen.findByRole("table", {
      name: "Ranked capability alternatives",
    })).toBeInTheDocument();

    rerender(<CapabilityOptimizationPanel runner={runner} workflow={changed}
      onWorkflowChange={onWorkflowChange} />);

    expect(screen.queryByRole("table", {
      name: "Ranked capability alternatives",
    })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Target distance"), {
      target: { value: "201" },
    });
    expect(onWorkflowChange).toHaveBeenCalled();
  });

  it("overlays edits without normalizing accepted non-editable workflow fields", () => {
    function Harness() {
      const [workflow, setWorkflow] = useState(customWorkflow);
      return <>
        <CapabilityOptimizationPanel workflow={workflow}
          onWorkflowChange={setWorkflow} />
        <output aria-label="Persisted capability workflow">
          {capabilityWorkflowToJson(workflow)}
        </output>
      </>;
    }
    render(<Harness />);

    fireEvent.change(screen.getByLabelText("Target distance"), {
      target: { value: "201" },
    });

    const persisted = JSON.parse(
      screen.getByLabelText("Persisted capability workflow").textContent ?? "",
    );
    expect(persisted.request.target.distance_m).toBe(201);
    expect(persisted.request.target).toMatchObject({
      kind: "fairway", band_half_length_m: 21, half_width_m: 8,
    });
    expect(persisted.request).toMatchObject({
      problem_id: "custom-problem-42", cvar_alpha: 0.83,
      minimum_success_fraction: 0.64,
    });
    expect(persisted.profile).toMatchObject({
      provenance: "measured/session-42", confidence: 0.71,
    });
    expect(persisted.profile.clubs[0]).toMatchObject({
      provenance: "fit/driver-42", confidence: 0.63,
      matrix: [[1, 0.2, 0], [0.2, 1, 0.1], [0, 0.1, 1]],
    });
    expect(persisted.profile.clubs[0].parameters[0].bias).toBe(0.4);
    expect(persisted.evaluator_config.spin_defaults[0].provenance)
      .toBe("measured/spin-42");
  });

  it.each([
    ["mph", /unit/i], ["covariance", /correlation/i],
  ] as const)("rejects a noncanonical %s workflow before panel apply", (kind, message) => {
    expect(() => render(
      <CapabilityOptimizationPanel workflow={noncanonicalWorkflow(kind)}
        onWorkflowChange={vi.fn()} />,
    )).toThrow(message);
  });
});
