import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SimulationStatusHeader } from "./SimulationStatusHeader";

describe("SimulationStatusHeader", () => {
  it("keeps model selection, capability guidance, and run status together", () => {
    const onSourceKindChange = vi.fn();
    render(
      <SimulationStatusHeader
        sourceKind="manual"
        onSourceKindChange={onSourceKindChange}
        status="Not run"
        warning={false}
      />,
    );

    expect(screen.getByRole("heading", { name: "Swing-to-Impact Simulation" }))
      .toBeInTheDocument();
    expect(screen.getByLabelText("Swing Source")).toHaveValue("manual");
    expect(screen.getByText(/specified constant-twist delivery/i))
      .toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Simulation run status" }))
      .toHaveTextContent("Not run");

    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    expect(onSourceKindChange).toHaveBeenCalledWith("double_pendulum");
  });

  it("distinguishes an error from an ordinary stale warning", () => {
    const { rerender } = render(
      <SimulationStatusHeader
        sourceKind="double_pendulum"
        onSourceKindChange={() => undefined}
        status="Inputs changed — run required"
        warning
      />,
    );
    const status = screen.getByRole("status", { name: "Simulation run status" });
    expect(status.className).toContain("amber");

    rerender(
      <SimulationStatusHeader
        sourceKind="double_pendulum"
        onSourceKindChange={() => undefined}
        status="Run failed: invalid state"
        warning
      />,
    );
    expect(status.className).toContain("red");
  });

  it("supports selecting the movement optimizer swing source", () => {
    const onSourceKindChange = vi.fn();
    render(
      <SimulationStatusHeader
        sourceKind="movement_optimizer"
        onSourceKindChange={onSourceKindChange}
        status="Completed — impact and flight available"
        warning={false}
      />,
    );

    expect(screen.getByLabelText("Swing Source")).toHaveValue("movement_optimizer");
    expect(screen.getByText(/reproducible swing delivery generated from golfer anthropometry/i))
      .toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "manual" },
    });
    expect(onSourceKindChange).toHaveBeenCalledWith("manual");
  });
});
