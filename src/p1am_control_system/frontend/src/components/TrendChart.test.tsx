import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TrendChart } from "./TrendChart";
import { TAG_COUNT } from "../lib/tags";

/** Build a history buffer of `rows` frames, each a full tag array. */
function makeHistory(rows: number): number[][] {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: TAG_COUNT }, (_, t) => Math.sin((r + t) / 5) * 50 + 50),
  );
}

const tagValues = Array.from({ length: TAG_COUNT }, () => 0);

describe("TrendChart", () => {
  it("renders from an empty history without crashing", () => {
    render(<TrendChart history={[]} tagValues={tagValues} />);
    expect(screen.getByText("Trends")).toBeInTheDocument();
  });

  it("renders from a short history without crashing", () => {
    render(<TrendChart history={makeHistory(3)} tagValues={tagValues} />);
    expect(screen.getByText("Trends")).toBeInTheDocument();
  });

  it("downsamples a long history to a bounded point count", () => {
    // 4000 rows would produce thousands of SVG points at full resolution; the
    // component strides down to RENDER_MAX_POINTS (600) before path building.
    const { container } = render(
      <TrendChart history={makeHistory(4000)} tagValues={tagValues} />,
    );
    // Default selected tags [0, 1, 10] → one line <path> each (plus area path).
    const paths = container.querySelectorAll("path[stroke]");
    expect(paths.length).toBeGreaterThan(0);

    // A line path with N points has N-1 "L" commands; with downsampling this
    // must be far below the raw 4000-sample count.
    const linePath = paths[0].getAttribute("d") ?? "";
    const lineCommands = (linePath.match(/L/g) ?? []).length;
    expect(lineCommands).toBeGreaterThan(0);
    expect(lineCommands).toBeLessThanOrEqual(600);
  });

  it("freezes the on-screen slice when the Freeze button is clicked", () => {
    const { rerender } = render(
      <TrendChart history={makeHistory(50)} tagValues={tagValues} />,
    );
    // Click Freeze — captures the current slice synchronously in the handler.
    fireEvent.click(screen.getByText("Freeze"));
    expect(screen.getByText("Frozen")).toBeInTheDocument();

    // New live frames arriving while frozen must not crash or unfreeze.
    rerender(<TrendChart history={makeHistory(120)} tagValues={tagValues} />);
    expect(screen.getByText("Frozen")).toBeInTheDocument();
  });
});
