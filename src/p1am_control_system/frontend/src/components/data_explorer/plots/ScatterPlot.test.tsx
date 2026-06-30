import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { ScatterPlot } from "./ScatterPlot";

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
});
