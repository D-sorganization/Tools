import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";

import { PlotCrosshair, PlotTooltip } from "./PlotCrosshair";
import type { CrosshairModel } from "./projection";

const BOUNDS = { x0: 0, y0: 0, x1: 200, y1: 100 };

describe("PlotTooltip", () => {
  it("renders a background rect and one text line per entry", () => {
    const { container } = render(
      <svg width={200} height={100}>
        <PlotTooltip
          lines={["x: 1.00", "a: 2.00"]}
          anchor={{ x: 50, y: 50 }}
          bounds={BOUNDS}
        />
      </svg>,
    );
    const tip = container.querySelector("g.plot-tooltip");
    expect(tip).not.toBeNull();
    expect(tip?.getAttribute("aria-hidden")).toBe("true");
    expect(container.querySelectorAll("g.plot-tooltip rect")).toHaveLength(1);
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    expect(lines).toEqual(["x: 1.00", "a: 2.00"]);
  });

  it("renders nothing for an empty line list", () => {
    const { container } = render(
      <svg width={200} height={100}>
        <PlotTooltip lines={[]} anchor={{ x: 0, y: 0 }} bounds={BOUNDS} />
      </svg>,
    );
    expect(container.querySelector("g.plot-tooltip")).toBeNull();
  });

  it("keeps the box inside the bounds (via placeTooltip)", () => {
    const { container } = render(
      <svg width={200} height={100}>
        <PlotTooltip
          lines={["value"]}
          anchor={{ x: 199, y: 99 }}
          bounds={BOUNDS}
        />
      </svg>,
    );
    const rect = container.querySelector("g.plot-tooltip rect")!;
    const rx = Number(rect.getAttribute("x"));
    const rw = Number(rect.getAttribute("width"));
    expect(rx).toBeGreaterThanOrEqual(BOUNDS.x0);
    expect(rx + rw).toBeLessThanOrEqual(BOUNDS.x1);
  });
});

describe("PlotCrosshair", () => {
  const model: CrosshairModel = {
    lineX: 100,
    dataX: 1.5,
    markers: [
      { label: "a", color: "#ff0000", px: 100, py: 50, value: 12.5 },
      { label: "b", color: "#00ff00", px: 100, py: 20, value: 3 },
    ],
  };

  it("draws a vertical guide line, a marker per series and a tooltip", () => {
    const { container } = render(
      <svg width={200} height={100}>
        <PlotCrosshair model={model} innerWidth={200} innerHeight={100} />
      </svg>,
    );
    const line = container.querySelector("line.plot-crosshair-line");
    expect(line?.getAttribute("x1")).toBe("100");
    expect(line?.getAttribute("y2")).toBe("100"); // spans the inner height
    expect(
      container.querySelectorAll("circle.plot-crosshair-marker"),
    ).toHaveLength(2);
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    expect(lines).toEqual(["x: 1.50", "a: 12.50", "b: 3.00"]);
  });

  it("marks its decorations aria-hidden", () => {
    const { container } = render(
      <svg width={200} height={100}>
        <PlotCrosshair model={model} innerWidth={200} innerHeight={100} />
      </svg>,
    );
    expect(
      container.querySelector("g.plot-crosshair")?.getAttribute("aria-hidden"),
    ).toBe("true");
  });
});
