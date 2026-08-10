import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FlightPlayback3D } from "./FlightPlayback3D";
import type { FlightPoint } from "../model/flight";
import { createSpatialTarget, sphereTolerance, targetPointFromFrame } from "../model/spatialTarget";

const points: FlightPoint[] = [
  { time: 0, position: [0, 0, 0], velocity: [1, 0, 1] },
  { time: 1, position: [10, 8, 2], velocity: [1, 0, 0] },
  { time: 2, position: [20, 0, 3], velocity: [1, 0, -1] },
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
    fireEvent.click(screen.getByRole("button", { name: "Jump to Apex" }));
    expect(screen.getByText("1.00 / 2.00 s")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Jump to Landing" }));
    expect(screen.getByText("2.00 / 2.00 s")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Jump to Launch" }));
    expect(screen.getByText("0.00 / 2.00 s")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Step Forward One Frame" }));
    expect(screen.getByText("1.00 / 2.00 s")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Step Back One Frame" }));
    expect(screen.getByText("0.00 / 2.00 s")).toBeInTheDocument();
  });

  it("preserves its intrinsic aspect ratio at responsive widths", () => {
    render(<FlightPlayback3D points={points} />);

    const canvas = screen.getByLabelText("Interactive 3D ball-flight playback");
    expect(canvas).toHaveAttribute("width", "860");
    expect(canvas).toHaveAttribute("height", "420");
    expect(canvas).toHaveStyle({
      width: "100%",
      height: "auto",
      aspectRatio: "860 / 420",
    });
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

  it("loops at landing and keeps one animation frame scheduled", () => {
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
    render(<FlightPlayback3D points={points} />);

    fireEvent.click(screen.getByRole("checkbox", { name: "Loop Ball Flight Playback" }));
    fireEvent.change(screen.getByRole("slider", { name: "Ball Flight Time" }), {
      target: { value: "1.999" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Play Ball Flight" }));
    const callback = [...callbacks.values()][0];
    if (!callback) throw new Error("Expected one scheduled animation frame");
    callbacks.clear();
    act(() => callback(performance.now() + 10));

    expect(screen.getByRole("button", { name: "Pause Ball Flight" })).toBeEnabled();
    expect(callbacks.size).toBe(1);
    const slider = screen.getByRole("slider", { name: "Ball Flight Time" });
    expect(Number((slider as HTMLInputElement).value)).toBeLessThan(0.1);
  });

  it("keeps the active labeled target rendered while orbiting and zooming", () => {
    const fillText = vi.fn();
    const context: unknown = new Proxy(function () {} as object, {
      get: (_target, prop) => prop === "fillText" ? fillText : () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
    const target = createSpatialTarget({
      label: "Apex Window",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([30, 15, 4], "app"),
      tolerance: sphereTolerance(3),
      elevationSource: "absolute",
    });
    render(<FlightPlayback3D points={points} spatialTarget={target} />);
    const canvas = screen.getByLabelText("Interactive 3D ball-flight playback");

    expect(fillText).toHaveBeenCalledWith(
      expect.stringContaining("ACTIVE TARGET · Apex Window"),
      expect.any(Number), expect.any(Number),
    );
    const initialCalls = fillText.mock.calls.length;
    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 140, clientY: 120 });
    fireEvent.pointerUp(canvas, { pointerId: 1 });
    fireEvent.wheel(canvas, { deltaY: -100 });

    expect(fillText.mock.calls.length).toBeGreaterThan(initialCalls);
    expect(canvas).toHaveAttribute("aria-description", expect.stringContaining("Apex Window"));
  });

  it("snaps exact views and predictably suspends and restores ball tracking", () => {
    render(<FlightPlayback3D points={points} />);
    const canvas = screen.getByLabelText("Interactive 3D ball-flight playback");
    fireEvent.click(screen.getByRole("button", { name: "Overhead" }));
    expect(screen.getByRole("button", { name: "Overhead" }))
      .toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("checkbox", { name: "Track Ball" }));
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking Ball");
    fireEvent.pointerDown(canvas, { pointerId: 4, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(canvas, { pointerId: 4, clientX: 130, clientY: 120 });
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking suspended");
    fireEvent.click(screen.getByRole("button", { name: "Re-center Ball" }));
    expect(screen.getByRole("status", { name: "Camera tracking state" }))
      .toHaveTextContent("Tracking Ball");
  });

  it.each([[[]], [[points[0]]]])(
    "keeps the active target rendered with a %s-point playback",
    (trajectory) => {
      const fillText = vi.fn();
      const context: unknown = new Proxy(function () {} as object, {
        get: (_target, prop) => prop === "fillText" ? fillText : () => context,
        set: () => true,
        apply: () => context,
      });
      vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
        context as CanvasRenderingContext2D,
      );
      const target = createSpatialTarget({
        label: "Always Visible",
        kind: "aerial_waypoint",
        point: targetPointFromFrame([30, 15, 4], "app"),
        tolerance: sphereTolerance(3),
        elevationSource: "absolute",
      });
      render(<FlightPlayback3D points={trajectory} spatialTarget={target} />);
      expect(fillText).toHaveBeenCalledWith(
        expect.stringContaining("ACTIVE TARGET · Always Visible"),
        expect.any(Number), expect.any(Number),
      );
      expect(screen.getByLabelText("Interactive 3D ball-flight playback"))
        .toHaveAttribute("aria-description", expect.stringContaining("Always Visible"));
    },
  );
});
