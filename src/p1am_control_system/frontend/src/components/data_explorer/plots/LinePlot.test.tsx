import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { LinePlot } from "./LinePlot";
import { mockSvgRect, pointerMoveAt } from "./hoverTestUtils";

describe("LinePlot", () => {
  const series = [
    {
      name: "a",
      points: [
        [0, 0],
        [1, 1],
        [2, 4],
      ] as [number, number][],
    },
    {
      name: "b",
      color: "#abcdef",
      dash: "4 2",
      points: [
        [0, 2],
        [1, 3],
      ] as [number, number][],
    },
  ];

  it("renders an <svg> and one <path> per series, forwarding a ref", () => {
    const ref = createRef<SVGSVGElement>();
    const { container } = render(
      <LinePlot ref={ref} width={300} height={200} series={series} />,
    );
    const svg = container.querySelector("svg");
    expect(svg).toBe(ref.current);
    expect(container.querySelectorAll("path.plot-line")).toHaveLength(
      series.length,
    );
  });

  it("honours per-series color and dash overrides", () => {
    const { container } = render(
      <LinePlot width={300} height={200} series={series} />,
    );
    const paths = Array.from(
      container.querySelectorAll("path.plot-line"),
    );
    const b = paths.find((p) => p.getAttribute("data-series") === "b");
    expect(b?.getAttribute("stroke")).toBe("#abcdef");
    expect(b?.getAttribute("stroke-dasharray")).toBe("4 2");
  });

  it("renders a legend item per series when legend is enabled", () => {
    const { container } = render(
      <LinePlot width={300} height={200} series={series} legend />,
    );
    expect(container.querySelectorAll("g.plot-legend-item")).toHaveLength(
      series.length,
    );
  });

  it("breaks the path at non-finite points", () => {
    const gapped = [
      {
        name: "g",
        points: [
          [0, 0],
          [1, NaN],
          [2, 2],
        ] as [number, number][],
      },
    ];
    const { container } = render(
      <LinePlot width={300} height={200} series={gapped} />,
    );
    const d = container
      .querySelector("path.plot-line")
      ?.getAttribute("d");
    // Two move commands -> the line is split into two segments.
    expect((d?.match(/M/g) ?? []).length).toBe(2);
    expect(d).toMatch(/^M[^ ]+ M[^ ]+$/);
    expect(d).not.toMatch(/^ | {2}| $/);
  });

  it("emits an empty path when every sample is non-finite", () => {
    const { container } = render(
      <LinePlot
        width={300}
        height={200}
        series={[{ name: "invalid", points: [[NaN, Infinity]] }]}
      />,
    );

    expect(container.querySelector("path.plot-line")?.getAttribute("d")).toBe("");
  });

  it("shows a hover crosshair with each series' value on pointer move", () => {
    const { container } = render(
      <LinePlot width={300} height={200} series={series} />,
    );
    const svg = container.querySelector("svg") as SVGSVGElement;
    mockSvgRect(svg, 300, 200);
    // xDomain [0,2], left margin 52, innerWidth 232 -> data-x 1 at clientX 168.
    pointerMoveAt(svg, 168);
    expect(container.querySelector("line.plot-crosshair-line")).not.toBeNull();
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    expect(lines).toContain("a: 1.00"); // series a sample at x=1
    expect(lines).toContain("b: 3.00"); // series b sample at x=1
  });
});
