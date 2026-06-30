import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { PlotFrame } from "./PlotFrame";
import { makeProjector } from "./projection";

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
