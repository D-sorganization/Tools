import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PlotData, PlotSpec } from "../model/plotspec";
import { PlotCanvasCard } from "./PlotCanvasCard";

const spec = (kind: PlotSpec["kind"]): PlotSpec => ({
  kind, x_key: "swing.time_s", y_keys: kind === "histogram" ? [] : ["swing.speed_mps"],
  series_key: null, title: "Evidence", x_log: false, y_log: false,
  x_start: null, x_stop: null, x_count: 25,
});

const lineData = (): PlotData => ({
  spec: spec("line"), x: [0, 1, 2],
  series: [
    { label: "Speed", values: [10, 11, 12] },
    { label: "Rate", values: [20, 21, 22] },
  ],
  xLabel: "Time [s]", yLabel: "Value",
});

describe("PlotCanvasCard exact evidence", () => {
  beforeEach(() => {
    const context: unknown = new Proxy(function () {} as object, {
      get: (_target, property) => property === "measureText"
        ? () => ({ width: 40 }) : () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });
  afterEach(() => vi.restoreAllMocks());

  it("navigates exact series points and clears synchronously on replacement", () => {
    const props = { label: "Study", selected: true, onSelect: vi.fn(), onCanvas: vi.fn() };
    const { rerender } = render(<PlotCanvasCard {...props} data={lineData()} />);
    const canvas = screen.getByRole("img", { name: "Study plot" });
    fireEvent.keyDown(canvas, { key: "Home" });
    expect(screen.getByRole("status")).toHaveTextContent("Series Speed; source point 1/3");
    fireEvent.keyDown(canvas, { key: "ArrowDown" });
    expect(screen.getByRole("status")).toHaveTextContent("Series Rate; source point 1/3");
    rerender(<PlotCanvasCard {...props} data={{ ...lineData(), x: [0, 2, 4] }} />);
    expect(screen.getByRole("status")).toHaveTextContent("No exact point selected");
  });

  it("navigates deterministic derived histogram bins", () => {
    const data: PlotData = {
      spec: spec("histogram"), x: [0, 0, 1, 2, 2], series: [],
      xLabel: "Speed [m/s]", yLabel: "Count",
    };
    render(<PlotCanvasCard data={data} label="Histogram" selected
      onSelect={vi.fn()} onCanvas={vi.fn()} />);
    const canvas = screen.getByRole("img", { name: "Histogram plot" });
    fireEvent.keyDown(canvas, { key: "Home" });
    expect(screen.getByRole("status")).toHaveTextContent("Histogram bin 1/10");
    fireEvent.keyDown(canvas, { key: "ArrowRight" });
    expect(screen.getByRole("status")).toHaveTextContent("Histogram bin 2/10");
  });

  it("uses rendered CSS pixels on logarithmic axes", () => {
    const data = lineData();
    data.spec = { ...data.spec, x_log: true, y_log: true };
    data.x = [1, 10, 100];
    data.series = [{ label: "Speed", values: [1, 10, 100] }];
    render(<PlotCanvasCard data={data} label="Log" selected
      onSelect={vi.fn()} onCanvas={vi.fn()} />);
    const canvas = screen.getByRole("img", { name: "Log plot" });
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 860, bottom: 420,
      width: 860, height: 420, toJSON: () => ({}),
    });
    fireEvent.click(canvas, { clientX: 92, clientY: 359 });
    expect(screen.getByRole("status")).toHaveTextContent("source point 1/3");
  });
});
