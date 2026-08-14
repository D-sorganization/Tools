import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_SCENARIO } from "../model/impact";
import { PlotsPanel } from "./PlotsPanel";
import { resolvePlotLayout } from "./plotLayout";

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

  it("reserves a non-overlapping outside legend rail at constrained width", () => {
    const layout = resolvePlotLayout(520, 320, "outside_right");

    expect(layout.plotRight).toBeLessThanOrEqual(layout.legendX);
    expect(layout.legendX).toBeGreaterThan(0);
    expect(layout.legendX).toBeLessThan(520);
    expect(() => resolvePlotLayout(200, 320, "outside_right")).toThrow(RangeError);
    expect(() => resolvePlotLayout(Number.NaN, 320, "hidden")).toThrow(RangeError);
  });
});
