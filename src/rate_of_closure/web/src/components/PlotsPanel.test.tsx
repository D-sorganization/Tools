import { StrictMode } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_SCENARIO } from "../model/impact";
import { runSimulation, type SimulationInput } from "../model/simulation";
import { PlotsPanel } from "./PlotsPanel";

describe("PlotsPanel view workspace", () => {
  beforeEach(() => {
    const context: unknown = new Proxy(function () {} as object, {
      get: (_target, property) =>
        property === "measureText" ? () => ({ width: 40 }) : () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });

  afterEach(() => vi.restoreAllMocks());

  it("renders every managed plot in a distinct canvas", () => {
    render(<PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5} />);

    fireEvent.change(screen.getByRole("combobox", { name: "Built-in plot" }), {
      target: { value: "swing_time_series" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    expect(screen.getByLabelText("Closure Sweep plot")).toBeInTheDocument();
    expect(screen.getByLabelText("Swing Time Series plot")).toBeInTheDocument();
    expect(screen.getAllByRole("img", { name: /plot$/ })).toHaveLength(2);
  });

  it("keeps zoom, autoscale, and legend placement independent per plot", () => {
    render(<PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5} />);
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    const cards = screen.getAllByRole("group", { name: /plot controls$/ });
    expect(cards).toHaveLength(2);
    fireEvent.click(within(cards[0]).getByRole("button", { name: "Zoom In" }));
    expect(within(cards[0]).getByText("125%" )).toBeInTheDocument();
    expect(within(cards[1]).getByText("100%" )).toBeInTheDocument();

    fireEvent.change(within(cards[0]).getByRole("combobox", { name: "Legend position" }), {
      target: { value: "hidden" },
    });
    expect(within(cards[0]).getByRole("combobox", { name: "Legend position" }))
      .toHaveValue("hidden");
    expect(within(cards[1]).getByRole("combobox", { name: "Legend position" }))
      .toHaveValue("outside_right");

    fireEvent.click(within(cards[0]).getByRole("button", { name: "Auto Fit" }));
    expect(within(cards[0]).getByText("100%" )).toBeInTheDocument();
  });

  it("supports pointer-wheel zoom directly on a plot", () => {
    render(<PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5} />);
    const canvas = screen.getByLabelText("Closure Sweep plot");
    fireEvent.wheel(canvas, { deltaY: -100 });
    expect(screen.getByText("125%" )).toBeInTheDocument();
  });

  it("computes only the newly added plot and never recomputes on selection", () => {
    const executor = vi.fn((input: SimulationInput) => runSimulation(input));
    render(
      <PlotsPanel
        scenario={DEFAULT_SCENARIO}
        loftDeg={10.5}
        executeSimulation={executor}
      />,
    );
    expect(executor).toHaveBeenCalledTimes(42);

    fireEvent.change(screen.getByRole("combobox", { name: "Built-in plot" }), {
      target: { value: "swing_time_series" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(executor).toHaveBeenCalledTimes(42);
    fireEvent.click(screen.getByRole("button", { name: "Closure Sweep" }));
    expect(executor).toHaveBeenCalledTimes(42);
    fireEvent.keyDown(screen.getByLabelText("Closure Sweep plot"), { key: "Home" });
    expect(executor).toHaveBeenCalledTimes(42);
  });

  it("does not duplicate scientific computation under Strict Mode", () => {
    const executor = vi.fn((input: SimulationInput) => runSimulation(input));
    render(
      <StrictMode>
        <PlotsPanel
          scenario={DEFAULT_SCENARIO}
          loftDeg={10.5}
          executeSimulation={executor}
        />
      </StrictMode>,
    );
    expect(executor).toHaveBeenCalledTimes(42);
  });

  it("atomically retains prior data and selection when recomputation fails", () => {
    const accepted = vi.fn((input: SimulationInput) => runSimulation(input));
    const failed = vi.fn(() => {
      throw new Error("planted plot authority failure");
    });
    const { rerender } = render(
      <PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5}
        executeSimulation={accepted} />,
    );
    const canvas = screen.getByLabelText("Closure Sweep plot");
    fireEvent.keyDown(canvas, { key: "Home" });
    expect(screen.getByText(/source point 1\//)).toBeInTheDocument();

    rerender(<PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5}
      executeSimulation={failed} />);
    expect(screen.getByLabelText("Closure Sweep plot")).toBe(canvas);
    expect(screen.getByRole("alert")).toHaveTextContent(
      "prior accepted plot retained",
    );
    expect(screen.getByText(/source point 1\//)).toBeInTheDocument();
    expect(failed).toHaveBeenCalledTimes(1);

    rerender(<PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5}
      executeSimulation={accepted} />);
    expect(screen.getByText(/No exact point selected/)).toBeInTheDocument();
  });

  it("rejects a ninth plot before any scientific computation", () => {
    const executor = vi.fn((input: SimulationInput) => runSimulation(input));
    render(
      <PlotsPanel
        scenario={DEFAULT_SCENARIO}
        loftDeg={10.5}
        executeSimulation={executor}
      />,
    );
    fireEvent.change(screen.getByRole("combobox", { name: "Built-in plot" }), {
      target: { value: "swing_time_series" },
    });
    for (let index = 0; index < 8; index += 1) {
      fireEvent.click(screen.getByRole("button", { name: "Add" }));
    }

    expect(screen.getAllByRole("img", { name: /plot$/ })).toHaveLength(8);
    expect(screen.getByRole("alert")).toHaveTextContent("at most 8 managed plots");
    expect(executor).toHaveBeenCalledTimes(42);
  });
});
