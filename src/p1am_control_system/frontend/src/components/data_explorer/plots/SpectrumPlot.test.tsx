import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { createRef } from "react";
import { SpectrumPlot } from "./SpectrumPlot";
import { mockSvgRect, pointerMoveAt } from "./hoverTestUtils";

describe("SpectrumPlot", () => {
  const freqs = [0, 1, 2, 3, 4, 5];
  const power = [0.1, 0.2, 1.0, 0.3, 0.2, 0.1];

  it("renders an <svg> with a single spectrum path, forwarding a ref", () => {
    const ref = createRef<SVGSVGElement>();
    const { container } = render(
      <SpectrumPlot ref={ref} width={300} height={200} freqs={freqs} power={power} />,
    );
    expect(container.querySelector("svg")).toBe(ref.current);
    expect(container.querySelectorAll("path.plot-spectrum")).toHaveLength(1);
  });

  it("plots a point per (freq, power) pair as one connected path", () => {
    const { container } = render(
      <SpectrumPlot width={300} height={200} freqs={freqs} power={power} />,
    );
    const d = container
      .querySelector("path.plot-spectrum")
      ?.getAttribute("d");
    // One M (first sample) followed by L commands for the remaining samples.
    expect((d?.match(/M/g) ?? []).length).toBe(1);
    expect((d?.match(/L/g) ?? []).length).toBe(freqs.length - 1);
  });

  it("drops non-positive samples on a log axis", () => {
    const { container } = render(
      <SpectrumPlot
        width={300}
        height={200}
        freqs={freqs}
        power={power}
        logX
        logY
      />,
    );
    const d = container
      .querySelector("path.plot-spectrum")
      ?.getAttribute("d");
    // freq 0 is excluded on a log-x axis, so only 5 of 6 samples remain.
    const points = (d?.match(/[ML]/g) ?? []).length;
    expect(points).toBe(5);
    expect(d).not.toMatch(/^ | {2}| $/);
  });

  it("starts a new segment after each non-finite sample", () => {
    const { container } = render(
      <SpectrumPlot
        width={300}
        height={200}
        freqs={[0, 1, 2, 3]}
        power={[1, 2, NaN, 4]}
      />,
    );
    const d = container.querySelector("path.plot-spectrum")?.getAttribute("d");

    expect(d?.match(/[ML]/g)).toEqual(["M", "L", "M"]);
    expect(d).not.toMatch(/^ | {2}| $/);
  });

  it("shows a hover crosshair with the nearest power sample on move", () => {
    const { container } = render(
      <SpectrumPlot width={300} height={200} freqs={freqs} power={power} />,
    );
    const svg = container.querySelector("svg") as SVGSVGElement;
    mockSvgRect(svg, 300, 200);
    // xDomain [0,5], left margin 52, innerWidth 232 -> freq 2 at clientX ~= 145.
    pointerMoveAt(svg, 145);
    expect(container.querySelector("line.plot-crosshair-line")).not.toBeNull();
    const lines = Array.from(
      container.querySelectorAll("g.plot-tooltip text"),
    ).map((n) => n.textContent);
    expect(lines).toContain("power: 1.00"); // power[2] = 1.0 at freq 2
  });
});
