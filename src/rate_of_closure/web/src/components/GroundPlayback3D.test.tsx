import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/flight_to_ground_golden_v1.json";
import { parseFlightToGroundResultRecord } from "../model/flightGroundResultContract";
import { GroundPlaybackTimeline } from "../model/groundPlayback";
import { GroundPlayback3D } from "./GroundPlayback3D";

describe("GroundPlayback3D", () => {
  const timeline = new GroundPlaybackTimeline(
    parseFlightToGroundResultRecord(fixture.result),
  );

  beforeEach(() => {
    const context: unknown = new Proxy(function () {} as object, {
      get: () => () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });

  afterEach(() => vi.restoreAllMocks());

  it("exposes phase-aware controls, exact markers, and locked aspect", () => {
    render(<GroundPlayback3D timeline={timeline} />);
    expect(screen.getByRole("button", { name: "Play ground result" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Jump to Roll" }));
    expect(screen.getByRole("status", { name: "Ground playback position" }))
      .toHaveTextContent("Roll");
    expect(screen.getByText("Carry / first contact")).toBeInTheDocument();
    expect(screen.getByText("Total / rest")).toBeInTheDocument();
    expect(screen.getByLabelText("Interactive 3D ground playback")).toHaveStyle({
      width: "100%", height: "auto", aspectRatio: "860 / 420",
    });
  });

  it("steps exact frames and restarts when playing at the end", () => {
    let nextId = 0;
    const callbacks = new Map<number, FrameRequestCallback>();
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      nextId += 1;
      callbacks.set(nextId, callback);
      return nextId;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      callbacks.delete(id);
    });
    render(<GroundPlayback3D timeline={timeline} />);
    fireEvent.click(screen.getByRole("button", { name: "Jump to Rest" }));
    fireEvent.click(screen.getByRole("button", { name: "Play ground result" }));
    expect(screen.getByRole("status", { name: "Ground playback position" }))
      .toHaveTextContent("Impact");
    expect(callbacks.size).toBe(1);
    fireEvent.click(screen.getByRole("button", { name: "Pause ground result" }));
    expect(callbacks.size).toBe(0);
    fireEvent.click(screen.getByRole("button", { name: "Next exact ground frame" }));
    expect(screen.getByRole("status", { name: "Ground playback position" }))
      .toHaveTextContent("Skid");
  });

  it("loops while retaining one scheduled animation frame", () => {
    let callback: FrameRequestCallback | undefined;
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((value) => {
      callback = value;
      return 1;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined);
    render(<GroundPlayback3D timeline={timeline} />);
    fireEvent.click(screen.getByRole("checkbox", { name: "Loop ground playback" }));
    fireEvent.click(screen.getByRole("button", { name: "Jump to Rest" }));
    fireEvent.click(screen.getByRole("button", { name: "Play ground result" }));
    if (!callback) throw new Error("Expected one animation callback");
    act(() => callback?.(performance.now() + 20));
    expect(screen.getByRole("button", { name: "Pause ground result" })).toBeEnabled();
  });
});
