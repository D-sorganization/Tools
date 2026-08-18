import { cleanup, fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { getClub } from "../model/club";
import { runSimulation, type SimulationInput } from "../model/simulation";
import {
  installSimulationPanelTestEnvironment,
  renderSimulationPanel,
} from "./simulationPanelTestSupport";

beforeAll(installSimulationPanelTestEnvironment);

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe("Simulation impact-time execution authority", () => {
  it("executes the exact pointer and keyboard commits", () => {
    const inputs: SimulationInput[] = [];
    renderSimulationPanel(getClub("Driver 10.5°"), undefined, (input) => {
      inputs.push(input);
      return runSimulation(input);
    });
    const slider = screen.getByRole("slider", { name: "Impact Time" });
    const initialCalls = inputs.length;

    fireEvent.change(slider, { target: { value: "40" } });
    fireEvent.pointerUp(slider);
    expect(inputs).toHaveLength(initialCalls + 1);
    expect(inputs[inputs.length - 1]?.impactTimeS).toBe(0.04);

    fireEvent.change(slider, { target: { value: "35" } });
    fireEvent.keyUp(slider, { key: "ArrowLeft" });
    expect(inputs).toHaveLength(initialCalls + 2);
    expect(inputs[inputs.length - 1]?.impactTimeS).toBe(0.035);
    fireEvent.keyUp(slider, { key: "Tab" });
    expect(inputs).toHaveLength(initialCalls + 2);
  });

  it("executes Auto tau and retains the prior scene on failure", () => {
    let fail = false;
    const inputs: SimulationInput[] = [];
    const executor = vi.fn((input: SimulationInput) => {
      inputs.push(input);
      if (fail) throw new Error("planted simulation failure");
      return runSimulation(input);
    });
    renderSimulationPanel(getClub("Driver 10.5°"), undefined, executor);
    const slider = screen.getByRole("slider", { name: "Impact Time" });
    fireEvent.change(slider, { target: { value: "30" } });
    fireEvent.pointerUp(slider);
    expect(inputs[inputs.length - 1]?.impactTimeS).toBe(0.03);

    fireEvent.click(screen.getByRole("button", { name: "Auto τ" }));
    expect(inputs[inputs.length - 1]?.impactTimeS).toBeNull();
    expect(screen.getByText("Completed — impact and flight available"))
      .toBeInTheDocument();

    const canvas = screen.getByLabelText(
      "Simulation scene with selectable screw-axis motion glyph",
    );
    fail = true;
    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    expect(screen.getByText(/Run failed: planted simulation failure.*prior.*retained/i))
      .toBeInTheDocument();
    expect(canvas).toBeInTheDocument();
  });

  it("keeps a first execution failure empty and bounds its status", () => {
    renderSimulationPanel(getClub("Driver 10.5°"), undefined, () => {
      throw new Error("x".repeat(600));
    });

    const status = screen.getByText(/no accepted simulation available/i);
    expect(status).toHaveTextContent(`Run failed: ${"x".repeat(512)};`);
    expect(status).not.toHaveTextContent("x".repeat(513));
  });
});
