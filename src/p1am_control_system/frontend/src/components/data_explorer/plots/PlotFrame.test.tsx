import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { PlotFrame, type HoverSeries } from "./PlotFrame";
import { makeProjector } from "./projection";
import {
  mockSvgRect,
  pointerMoveAt,
  pointerLeaveSvg,
} from "./hoverTestUtils";

describe("PlotFrame", () => {
  it("renders an <svg> with the requested size and forwards a ref", () => {
    const ref = createRef<SVGSVGElement>();
    const { container } = render(
      <PlotFrame
        ref={ref}
        width={320}
        height={200}
        xDomain={[0, 10]}
        yDomain={[0, 5]}
        xLabel="t"
        yLabel="v"
      />,
    );
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg).toBe(ref.current);
    expect(svg?.getAttribute("width")).toBe("320");
    expect(svg?.getAttribute("height")).toBe("200");
  });

  it("renders the plot-area border rect and axis labels", () => {
    const { container } = render(
      <PlotFrame
        width={300}
        height={200}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        xLabel="freq"
        yLabel="power"
      />,
    );
    expect(container.querySelectorAll("rect.plot-border")).toHaveLength(1);
    const labels = Array.from(
      container.querySelectorAll("text.plot-axis-label"),
    ).map((n) => n.textContent);
    expect(labels).toContain("freq");
    expect(labels).toContain("power");
  });

  it("omits gridlines when grid is false", () => {
    const { container } = render(
      <PlotFrame
        width={300}
        height={200}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        grid={false}
      />,
    );
    expect(container.querySelectorAll("line.plot-grid")).toHaveLength(0);
  });
});

describe("makeProjector", () => {
  it("maps domain endpoints to the inner pixel area", () => {
    const p = makeProjector({
      width: 200,
      height: 100,
      xDomain: [0, 10],
      yDomain: [0, 4],
      margin: { top: 0, right: 0, bottom: 0, left: 0 },
    });
    expect(p.innerWidth).toBe(200);
    expect(p.innerHeight).toBe(100);
    expect(p.x(0)).toBeCloseTo(0, 6);
    expect(p.x(10)).toBeCloseTo(200, 6);
    // y is inverted: data max at the top (pixel 0).
    expect(p.y(4)).toBeCloseTo(0, 6);
    expect(p.y(0)).toBeCloseTo(100, 6);
  });

  it("applies a log10 transform on a log axis", () => {
    const p = makeProjector({
      width: 200,
      height: 100,
      xDomain: [1, 100],
      yDomain: [0, 1],
      logX: true,
      margin: { top: 0, right: 0, bottom: 0, left: 0 },
    });
    // log10(10) is the midpoint of log10(1)..log10(100) -> pixel 100.
    expect(p.x(10)).toBeCloseTo(100, 4);
    expect(p.x.log).toBe(true);
  });
});

describe("PlotFrame hover crosshair", () => {
  // Default margins: left 52, top 16 (see projection DEFAULT_MARGIN). With a
  // 300x200 frame and xDomain [0,2], innerWidth = 300-52-16 = 232, so data-x 1
  // lands at inner pixel 116, i.e. clientX = 116 + 52 = 168.
  const hover: HoverSeries[] = [
    { label: "a", color: "#ff0000", xs: [0, 1, 2], ys: [0, 10, 20] },
  ];

  function renderHoverFrame() {
    const result = render(
      <PlotFrame
        width={300}
        height={200}
        xDomain={[0, 2]}
        yDomain={[0, 20]}
        hoverSeries={hover}
      />,
    );
    const svg = result.container.querySelector("svg") as SVGSVGElement;
    mockSvgRect(svg, 300, 200);
    return { ...result, svg };
  }

  it("shows no crosshair before the pointer moves", () => {
    const { container } = renderHoverFrame();
    expect(container.querySelector("g.plot-crosshair")).toBeNull();
  });

  it("snaps a crosshair + tooltip to the nearest x-sample on move", () => {
    const { container, svg } = renderHoverFrame();
    pointerMoveAt(svg, 168); // data-x ~= 1 -> nearest sample value 10
    const line = container.querySelector("line.plot-crosshair-line");
    expect(line).not.toBeNull();
    expect(Number(line?.getAttribute("x1"))).toBeCloseTo(116, 0);
    expect(
      container.querySelectorAll("circle.plot-crosshair-marker"),
    ).toHaveLength(1);
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    expect(lines).toContain("a: 10.00");
    expect(lines).toContain("x: 1.00");
  });

  it("clears the crosshair on pointer leave", () => {
    const { container, svg } = renderHoverFrame();
    pointerMoveAt(svg, 168);
    expect(container.querySelector("g.plot-crosshair")).not.toBeNull();
    pointerLeaveSvg(svg);
    expect(container.querySelector("g.plot-crosshair")).toBeNull();
  });

  it("hides the crosshair when the cursor is in the axis margin", () => {
    const { container, svg } = renderHoverFrame();
    pointerMoveAt(svg, 168);
    expect(container.querySelector("g.plot-crosshair")).not.toBeNull();
    pointerMoveAt(svg, 10); // inside the left margin -> inner-x < 0
    expect(container.querySelector("g.plot-crosshair")).toBeNull();
  });

  it("stays inert (no crosshair) when hoverSeries is omitted", () => {
    const { container } = render(
      <PlotFrame width={300} height={200} xDomain={[0, 2]} yDomain={[0, 20]} />,
    );
    const svg = container.querySelector("svg") as SVGSVGElement;
    mockSvgRect(svg, 300, 200);
    pointerMoveAt(svg, 168);
    expect(container.querySelector("g.plot-crosshair")).toBeNull();
  });
});
