import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { SimulationRunTs } from "../model/simulation";
import { SwingPlaybackControls } from "./SwingPlaybackControls";

const run = {
  totalDurationS: 0.71,
  impactTimeS: 0.03,
  impactOutcome: { status: "hit", candidateTimeS: 0.03 },
} as unknown as SimulationRunTs;

function renderControls(overrides: Partial<React.ComponentProps<typeof SwingPlaybackControls>> = {}) {
  const props: React.ComponentProps<typeof SwingPlaybackControls> = {
    run,
    playing: false,
    setPlaying: vi.fn(),
    time: 0,
    setTime: vi.fn(),
    loop: false,
    setLoop: vi.fn(),
    rate: 1,
    setRate: vi.fn(),
    toggles: [],
    ...overrides,
  };
  render(<SwingPlaybackControls {...props} />);
  return props;
}

describe("SwingPlaybackControls", () => {
  it("rewinds before replaying a completed animation", () => {
    const props = renderControls({ time: run.totalDurationS });

    fireEvent.click(screen.getByRole("button", { name: "Play" }));

    expect(props.setTime).toHaveBeenCalledWith(0);
    expect(props.setPlaying).toHaveBeenCalledWith(true);
  });

  it("pauses without changing the current timeline position", () => {
    const props = renderControls({ playing: true, time: 0.25 });

    fireEvent.click(screen.getByRole("button", { name: "Pause" }));

    expect(props.setTime).not.toHaveBeenCalled();
    expect(props.setPlaying).toHaveBeenCalledWith(false);
  });

  it("provides granular speed, restart, loop, and path-trail controls", () => {
    const setRate = vi.fn();
    const setTrail = vi.fn();
    const props = renderControls({
      time: 0.25,
      rate: 1,
      setRate,
      toggles: [["Path Trail", false, setTrail, "Show the travelled path", "text-sky-300"]],
    });

    fireEvent.change(screen.getByRole("slider", { name: "Playback speed" }), {
      target: { value: "1.35" },
    });
    expect(setRate).toHaveBeenCalledWith(1.35);

    fireEvent.click(screen.getByRole("button", { name: "Restart" }));
    expect(props.setPlaying).toHaveBeenCalledWith(false);
    expect(props.setTime).toHaveBeenCalledWith(0);

    fireEvent.click(screen.getByRole("checkbox", { name: "Loop" }));
    expect(props.setLoop).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("checkbox", { name: "Path Trail" }));
    expect(setTrail).toHaveBeenCalledWith(true);
  });
});
