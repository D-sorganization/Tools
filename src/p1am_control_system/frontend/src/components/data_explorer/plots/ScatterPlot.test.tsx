import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { mockSvgRect, pointerMoveAt } from "./hoverTestUtils";

describe("ScatterPlot", () => {
  const series = [
    {
      name: "circles",
      points: [
        [0, 0],
        [1, 1],
        [2, 4],
      ] as [number, number][],
    },
  ];

  it("renders an <svg> and one marker per point, forwarding a ref", () => {
    const ref = createRef<SVGSVGElement>();
    const { container } = render(
      <ScatterPlot ref={ref} width={300} height={200} series={series} />,
    );
    expect(container.querySelector("svg")).toBe(ref.current);
    // Default shape is a circle.
    expect(container.querySelectorAll("circle.plot-marker")).toHaveLength(3);
  });

  it("renders square markers as rects when shape=square", () => {
    const squares = [
      {
        name: "sq",
        shape: "square" as const,
        points: [
          [0, 0],
          [1, 1],
        ] as [number, number][],
      },
    ];
    const { container } = render(
      <ScatterPlot width={300} height={200} series={squares} />,
    );
    expect(container.querySelectorAll("rect.plot-marker")).toHaveLength(2);
  });

  it("renders triangle markers as polygons when shape=triangle", () => {
    const tris = [
      {
        name: "tri",
        shape: "triangle" as const,
        points: [[0, 0]] as [number, number][],
      },
    ];
    const { container } = render(
      <ScatterPlot width={300} height={200} series={tris} />,
    );
    expect(container.querySelectorAll("polygon.plot-marker")).toHaveLength(1);
  });

  it("overlays a trendline path when provided", () => {
    const { container } = render(
      <ScatterPlot
        width={300}
        height={200}
        series={series}
        trendline={{
          points: [
            [0, 0],
            [2, 4],
          ],
        }}
      />,
    );
    expect(container.querySelectorAll("path.plot-trendline")).toHaveLength(1);
  });

  it("preserves trendline gaps without extra path separators", () => {
    const { container } = render(
      <ScatterPlot
        width={300}
        height={200}
        series={series}
        trendline={{
          points: [
            [0, 0],
            [1, 1],
            [NaN, 2],
            [2, 4],
          ],
        }}
      />,
    );
    const d = container.querySelector("path.plot-trendline")?.getAttribute("d");

    expect(d?.match(/[ML]/g)).toEqual(["M", "L", "M"]);
    expect(d).not.toMatch(/^ | {2}| $/);
  });

  it("skips non-finite points", () => {
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
      <ScatterPlot width={300} height={200} series={gapped} />,
    );
    expect(container.querySelectorAll("circle.plot-marker")).toHaveLength(2);
  });

  it("shows a hover crosshair with the nearest point's value on move", () => {
    const { container } = render(
      <ScatterPlot width={300} height={200} series={series} />,
    );
    const svg = container.querySelector("svg") as SVGSVGElement;
    mockSvgRect(svg, 300, 200);
    // xDomain [0,2], left margin 52, innerWidth 232 -> data-x 1 at clientX 168.
    pointerMoveAt(svg, 168);
    expect(container.querySelector("line.plot-crosshair-line")).not.toBeNull();
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    expect(lines).toContain("circles: 1.00"); // nearest point (1, 1)
  });
});
