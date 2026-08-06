import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FlightPlayback3D } from "./FlightPlayback3D";
import type { FlightPoint } from "../model/flight";

const points: FlightPoint[] = [
  { time: 0, position: [0, 0, 0], velocity: [1, 0, 1] },
  { time: 2, position: [20, 8, 3], velocity: [1, 0, -1] },
];

describe("FlightPlayback3D", () => {
  beforeEach(() => {
    const ctx: unknown = new Proxy(function () {} as object, {
      get: (_target, prop) =>
        prop === "measureText" ? () => ({ width: 0 }) : () => ctx,
      set: () => true,
      apply: () => ctx,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      ctx as CanvasRenderingContext2D,
    );
  });

  afterEach(() => vi.restoreAllMocks());

  it("provides accessible playback, scrubbing, speed, and jump controls", () => {
    render(<FlightPlayback3D points={points} />);

    expect(screen.getByLabelText("Interactive 3D ball-flight playback"))
      .toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("button", { name: "Play Ball Flight" })).toBeEnabled();
    expect(screen.getByRole("slider", { name: "Ball Flight Time" }))
      .toHaveAttribute("max", "2");
    expect(screen.getByLabelText("Playback Speed")).toHaveValue("1");
    fireEvent.click(screen.getByRole("button", { name: "Jump to Impact" }));
    expect(screen.getByText("2.00 / 2.00 s")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Jump to Launch" }));
    expect(screen.getByText("0.00 / 2.00 s")).toBeInTheDocument();
  });

  it("keeps at most one animation frame scheduled and cancels it on unmount", () => {
    let nextId = 0;
    const callbacks = new Map<number, FrameRequestCallback>();
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      nextId += 1;
      callbacks.set(nextId, callback);
      return nextId;
    });
    const cancel = vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      callbacks.delete(id);
    });
    const view = render(<FlightPlayback3D points={points} />);

    fireEvent.click(screen.getByRole("button", { name: "Play Ball Flight" }));
    expect(callbacks.size).toBe(1);
    fireEvent.click(screen.getByRole("button", { name: "Pause Ball Flight" }));
    expect(callbacks.size).toBe(0);
    fireEvent.click(screen.getByRole("button", { name: "Play Ball Flight" }));
    expect(callbacks.size).toBe(1);
    view.unmount();
    expect(callbacks.size).toBe(0);
    expect(cancel).toHaveBeenCalled();
  });
});
