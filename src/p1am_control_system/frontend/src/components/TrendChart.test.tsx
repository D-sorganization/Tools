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

  it("toggles the Pause button to Live and shows a FROZEN indicator", () => {
    const { rerender } = render(
      <TrendChart history={makeHistory(50)} tagValues={tagValues} />,
    );
    // Live to start: the toggle reads "Freeze", no frozen/panned badge.
    expect(screen.getByText("Freeze")).toBeInTheDocument();
    expect(screen.queryByText("FROZEN")).not.toBeInTheDocument();

    // Click Freeze — snapshots the current slice synchronously in the handler.
    fireEvent.click(screen.getByText("Freeze"));
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("FROZEN")).toBeInTheDocument();

    // New live frames arriving while frozen must not crash or unfreeze.
    rerender(<TrendChart history={makeHistory(120)} tagValues={tagValues} />);
    expect(screen.getByText("FROZEN")).toBeInTheDocument();

    // Clicking again resumes the live stream.
    fireEvent.click(screen.getByText("Live"));
    expect(screen.getByText("Freeze")).toBeInTheDocument();
    expect(screen.queryByText("FROZEN")).not.toBeInTheDocument();
  });

  it("changes the visible window when a window button is pressed", () => {
    // 4000 samples @ 10 Hz = 400 s of data — enough to hold any preset window.
    render(<TrendChart history={makeHistory(4000)} tagValues={tagValues} />);

    // Default span is 60 s, so the left (oldest) X-axis edge reads "-60s".
    expect(screen.getByText("-60s")).toBeInTheDocument();

    // Selecting the 30 s window narrows the span; the left edge moves to "-30s".
    fireEvent.click(screen.getByText("30s"));
    expect(screen.getByText("-30s")).toBeInTheDocument();
    expect(screen.queryByText("-60s")).not.toBeInTheDocument();

    // Widening to 300 s pushes the oldest visible edge back to "-300s".
    fireEvent.click(screen.getByText("300s"));
    expect(screen.getByText("-300s")).toBeInTheDocument();
  });

  it("pans back into history and reports the panned offset", () => {
    render(<TrendChart history={makeHistory(4000)} tagValues={tagValues} />);
    // Live to start — no panned/frozen badge.
    expect(screen.queryByText(/^panned/)).not.toBeInTheDocument();

    // The back-in-time (older) button is the one titled accordingly.
    fireEvent.click(screen.getByTitle("Scroll back in time (older)"));
    // Now panned back by ~30% of the 60 s window ≈ 18 s.
    expect(screen.getByText(/^panned -\d+s$/)).toBeInTheDocument();
  });
});
