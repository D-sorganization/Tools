import { describe, it, expect, vi } from "vitest";
import { render, fireEvent, screen } from "@testing-library/react";

import { TrendTimeControls } from "./TrendTimeControls";
import { BUFFER_WINDOW_SECONDS } from "../lib/trendTime";

describe("TrendTimeControls", () => {
  it("caps the committed window at the per-chart maxSeconds", () => {
    const onChange = vi.fn();
    render(
      <TrendTimeControls
        value={600}
        onChange={onChange}
        maxSeconds={BUFFER_WINDOW_SECONDS}
      />,
    );
    // value=600 s → the unit defaults to minutes. 90 min = 5400 s is above the
    // 3600 s (1 h) buffer cap, so it must clamp down to the cap.
    fireEvent.change(screen.getByLabelText("Time window value"), {
      target: { value: "90" },
    });
    expect(onChange).toHaveBeenLastCalledWith(BUFFER_WINDOW_SECONDS);
  });

  it("passes a value under the cap straight through", () => {
    const onChange = vi.fn();
    render(
      <TrendTimeControls
        value={600}
        onChange={onChange}
        maxSeconds={BUFFER_WINDOW_SECONDS}
      />,
    );
    fireEvent.change(screen.getByLabelText("Time window value"), {
      target: { value: "20" }, // 20 min = 1200 s, under the cap
    });
    expect(onChange).toHaveBeenLastCalledWith(1200);
  });

  it("allows windows well over an hour when no per-chart cap is given", () => {
    const onChange = vi.fn();
    render(<TrendTimeControls value={600} onChange={onChange} />);
    // 600 min = 36000 s (10 h) — permitted under the 24 h backfilled default.
    fireEvent.change(screen.getByLabelText("Time window value"), {
      target: { value: "600" },
    });
    expect(onChange).toHaveBeenLastCalledWith(36000);
  });
});
