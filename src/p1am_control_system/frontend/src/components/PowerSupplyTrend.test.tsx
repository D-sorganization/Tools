import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PowerSupplyTrend, type TrendSample } from "./PowerSupplyTrend";

/** Build `n` samples at 10 Hz with smooth, non-flat current/voltage/power. */
function makeSamples(n: number): TrendSample[] {
  const t0 = 1_700_000_000_000; // fixed epoch-ms base so tests are deterministic
  return Array.from({ length: n }, (_, k) => ({
    t: t0 + k * 100,
    i: 5 + Math.sin(k / 7) * 2,
    v: 24 + Math.cos(k / 9) * 3,
    p: 120 + Math.sin(k / 5) * 20,
  }));
}

const fullScales = {
  currentFullScale: 10,
  voltageFullScale: 48,
  powerFullScale: 480,
};

/** Count the trace <path> elements that actually carry geometry. */
function drawnTraces(container: HTMLElement): number {
  return Array.from(container.querySelectorAll("path.ps-trend-line")).filter(
    (p) => (p.getAttribute("d") ?? "").length > 0,
  ).length;
}

describe("PowerSupplyTrend", () => {
  it("shows the placeholder when there is no usable data", () => {
    render(<PowerSupplyTrend samples={[]} {...fullScales} />);
    expect(screen.getByText("waiting for live data…")).toBeInTheDocument();
  });

  it("renders all three traces from live samples", () => {
    const { container } = render(
      <PowerSupplyTrend samples={makeSamples(200)} {...fullScales} />,
    );
    // current, voltage, power — one drawable line path each.
    expect(drawnTraces(container)).toBe(3);
  });

  it("downsamples a long buffer to a bounded point count", () => {
    const { container } = render(
      <PowerSupplyTrend samples={makeSamples(8000)} {...fullScales} />,
    );
    const d = container.querySelector("path.ps-trend-line")?.getAttribute("d") ?? "";
    const points = (d.match(/[ML]/g) ?? []).length;
    expect(points).toBeGreaterThan(1);
    expect(points).toBeLessThanOrEqual(600);
  });

  it("toggles pause/live and keeps rendering traces while frozen", () => {
    const { container, rerender } = render(
      <PowerSupplyTrend samples={makeSamples(100)} {...fullScales} />,
    );
    expect(screen.getByText("Freeze")).toBeInTheDocument();
    expect(drawnTraces(container)).toBe(3);

    // Freeze snapshots the current buffer and flips the button to "Live".
    fireEvent.click(screen.getByRole("button", { name: "Pause plot" }));
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("Frozen")).toBeInTheDocument();

    // New live frames arriving while frozen must not crash or unfreeze.
    rerender(<PowerSupplyTrend samples={makeSamples(260)} {...fullScales} />);
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(drawnTraces(container)).toBe(3);

    // Back to live.
    fireEvent.click(screen.getByRole("button", { name: "Resume live stream" }));
    expect(screen.getByText("Freeze")).toBeInTheDocument();
  });

  it("scrolls back in time and flags the panned state, then RESET returns to live", () => {
    // Buffer longer than the default 3600 s window (45 000 samples @ 10 Hz ≈
    // 4500 s) so there is history to pan back into (offset can exceed 0).
    render(<PowerSupplyTrend samples={makeSamples(45000)} {...fullScales} />);
    // Live edge: no panned/frozen badge.
    expect(screen.queryByText("panned")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Scroll back in time" }));
    expect(screen.getByText("panned")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.queryByText("panned")).not.toBeInTheDocument();
  });
});
