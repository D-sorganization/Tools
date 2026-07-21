import { describe, it, expect } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { createRef } from "react";
import { Heatmap } from "./Heatmap";

describe("Heatmap", () => {
  const labels = ["a", "b", "c"];
  const matrix = [
    [1, 0.5, -0.2],
    [0.5, 1, 0.1],
    [-0.2, 0.1, 1],
  ];

  it("renders an <svg> and one cell <rect> per matrix entry, with a ref", () => {
    const ref = createRef<SVGSVGElement>();
    const { container } = render(
      <Heatmap ref={ref} width={300} height={300} labels={labels} matrix={matrix} />,
    );
    expect(container.querySelector("svg")).toBe(ref.current);
    expect(container.querySelectorAll("rect.heatmap-cell")).toHaveLength(9);
  });

  it("renders a label per row and per column", () => {
    const { container } = render(
      <Heatmap width={300} height={300} labels={labels} matrix={matrix} />,
    );
    expect(container.querySelectorAll("text.heatmap-row-label")).toHaveLength(
      labels.length,
    );
    expect(container.querySelectorAll("text.heatmap-col-label")).toHaveLength(
      labels.length,
    );
  });

  it("annotates cells when showValues is enabled", () => {
    const { container } = render(
      <Heatmap
        width={300}
        height={300}
        labels={labels}
        matrix={matrix}
        showValues
      />,
    );
    const texts = Array.from(container.querySelectorAll("g text")).map(
      (n) => n.textContent,
    );
    expect(texts).toContain("1.00");
    expect(texts).toContain("0.50");
    expect(texts).toContain("-0.20");
  });

  it("uses a custom colorFor when supplied", () => {
    const { container } = render(
      <Heatmap
        width={300}
        height={300}
        labels={labels}
        matrix={matrix}
        colorFor={() => "#00ff00"}
      />,
    );
    const cell = container.querySelector("rect.heatmap-cell");
    expect(cell?.getAttribute("fill")).toBe("#00ff00");
  });

  it("shows a cell's row/col/value on hover, and clears on leave", () => {
    const { container } = render(
      <Heatmap width={300} height={300} labels={labels} matrix={matrix} />,
    );
    const cell = container.querySelector(
      'rect.heatmap-cell[data-row="1"][data-col="2"]',
    )!;
    fireEvent.pointerEnter(cell);
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    // matrix[1][2] === 0.1, labels[1] === "b", labels[2] === "c".
    expect(lines).toContain("row: b");
    expect(lines).toContain("col: c");
    expect(lines).toContain("value: 0.10");
    fireEvent.pointerLeave(cell);
    expect(container.querySelector("g.plot-tooltip")).toBeNull();
  });
});
