import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { Histogram } from "./Histogram";

describe("Histogram", () => {
  const binEdges = [0, 1, 2, 3, 4];
  const counts = [2, 5, 3, 1];

  it("renders an <svg> and one bar <rect> per bin, forwarding a ref", () => {
    const ref = createRef<SVGSVGElement>();
    const { container } = render(
      <Histogram
        ref={ref}
        width={300}
        height={200}
        binEdges={binEdges}
        counts={counts}
      />,
    );
    expect(container.querySelector("svg")).toBe(ref.current);
    expect(container.querySelectorAll("rect.plot-bar")).toHaveLength(
      counts.length,
    );
  });

  it("scales bar heights so the tallest bin is the largest", () => {
    const { container } = render(
      <Histogram
        width={300}
        height={200}
        binEdges={binEdges}
        counts={counts}
      />,
    );
    const heights = Array.from(
      container.querySelectorAll("rect.plot-bar"),
    ).map((r) => Number(r.getAttribute("height")));
    // counts[1]=5 is the tallest bar.
    const maxIdx = heights.indexOf(Math.max(...heights));
    expect(maxIdx).toBe(1);
    expect(heights[0]).toBeGreaterThan(0);
  });

  it("clamps the bar count to min(edges-1, counts)", () => {
    const { container } = render(
      <Histogram
        width={300}
        height={200}
        binEdges={[0, 1, 2]}
        counts={[3, 4, 5]}
      />,
    );
    expect(container.querySelectorAll("rect.plot-bar")).toHaveLength(2);
  });

  it("applies a custom color", () => {
    const { container } = render(
      <Histogram
        width={300}
        height={200}
        binEdges={binEdges}
        counts={counts}
        color="#123456"
      />,
    );
    const bar = container.querySelector("rect.plot-bar");
    expect(bar?.getAttribute("fill")).toBe("#123456");
  });
});
