import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
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
});
