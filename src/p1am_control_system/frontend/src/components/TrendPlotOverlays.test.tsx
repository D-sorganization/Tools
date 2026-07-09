import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";

import { TrendCrosshair, type CrosshairSeries } from "./TrendPlotOverlays";

/** Render a <TrendCrosshair> inside a host <svg> and return the root group. */
function renderCrosshair(series: CrosshairSeries[], xLabel = "12:00:00") {
  return render(
    <svg viewBox="0 0 600 200" width={600} height={200}>
      <TrendCrosshair
        px={300}
        yTop={10}
        yBottom={170}
        plotLeft={40}
        plotRight={590}
        series={series}
        xLabel={xLabel}
      />
    </svg>,
  );
}

const series: CrosshairSeries[] = [
  { label: "K", color: "#f00", text: "512.3 °C", py: 40 },
  { label: "R", color: "#0af", text: "509.8 °C", py: 60 },
];

describe("TrendCrosshair", () => {
  it("renders nothing when there are no series", () => {
    const { container } = renderCrosshair([]);
    expect(container.querySelector('line[stroke-dasharray="3 3"]')).toBeNull();
  });

  it("draws a vertical guide line at the hovered x spanning the plot", () => {
    const { container } = renderCrosshair(series);
    const line = container.querySelector('line[stroke-dasharray="3 3"]');
    expect(line).not.toBeNull();
    expect(line?.getAttribute("x1")).toBe("300");
    expect(line?.getAttribute("x2")).toBe("300");
    expect(line?.getAttribute("y1")).toBe("10");
    expect(line?.getAttribute("y2")).toBe("170");
  });

  it("draws one marker circle per series at its point", () => {
    const { container } = renderCrosshair(series);
    const circles = container.querySelectorAll("circle");
    expect(circles.length).toBe(2);
    expect(circles[0].getAttribute("cx")).toBe("300");
    expect(circles[0].getAttribute("cy")).toBe("40");
    expect(circles[1].getAttribute("cy")).toBe("60");
  });

  it("shows the x label and each series value in the tooltip", () => {
    const { getByText } = renderCrosshair(series, "12:34:56");
    expect(getByText("12:34:56")).toBeInTheDocument();
    // getByText normalizes runs of whitespace, so match with a flexible regex.
    expect(getByText(/K\s+512\.3\s*°C/)).toBeInTheDocument();
    expect(getByText(/R\s+509\.8\s*°C/)).toBeInTheDocument();
  });

  it("keeps the tooltip inside the plot rectangle when hovering near the right edge", () => {
    const { container } = render(
      <svg viewBox="0 0 600 200" width={600} height={200}>
        <TrendCrosshair
          px={588}
          yTop={10}
          yBottom={170}
          plotLeft={40}
          plotRight={590}
          series={series}
          xLabel="now"
        />
      </svg>,
    );
    // The tooltip group is translated; its rect must not overflow plotRight=590.
    const rect = container.querySelector("rect");
    const group = rect?.parentElement as SVGGElement | null;
    const transform = group?.getAttribute("transform") ?? "";
    const m = /translate\(([\d.]+),/.exec(transform);
    const left = m ? Number.parseFloat(m[1]) : NaN;
    const boxW = Number.parseFloat(rect?.getAttribute("width") ?? "0");
    expect(left + boxW).toBeLessThanOrEqual(590 + 0.5);
  });
});
