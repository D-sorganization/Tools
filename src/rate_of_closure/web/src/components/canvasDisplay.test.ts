import { describe, expect, it, vi } from "vitest";

import { canvasContext, observeCanvas } from "./canvasDisplay";

describe("high-DPI canvas display", () => {
  it("sizes the backing store from CSS pixels and DPR while drawing in logical coordinates", () => {
    const canvas = document.createElement("canvas");
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
      width: 430, height: 210, left: 0, top: 0, right: 430, bottom: 210,
      x: 0, y: 0, toJSON: () => ({}),
    });
    vi.stubGlobal("devicePixelRatio", 2);
    const setTransform = vi.fn();
    vi.spyOn(canvas, "getContext").mockReturnValue({ setTransform } as unknown as CanvasRenderingContext2D);

    expect(canvasContext(canvas, { width: 860, height: 420 })).not.toBeNull();
    expect(canvas.width).toBe(860);
    expect(canvas.height).toBe(420);
    expect(setTransform).toHaveBeenCalledWith(1, 0, 0, 1, 0, 0);
    vi.unstubAllGlobals();
  });

  it("redraws through ResizeObserver and disconnects cleanly", () => {
    let notify: () => void = () => undefined;
    const disconnect = vi.fn();
    const observe = vi.fn();
    vi.stubGlobal("ResizeObserver", class {
      constructor(callback: () => void) { notify = callback; }
      observe = observe;
      disconnect = disconnect;
    });
    const canvas = document.createElement("canvas");
    const draw = vi.fn();
    const stop = observeCanvas({ current: canvas }, draw);
    expect(observe).toHaveBeenCalledWith(canvas);
    expect(draw).toHaveBeenCalledTimes(1);
    notify();
    expect(draw).toHaveBeenCalledTimes(2);
    stop();
    expect(disconnect).toHaveBeenCalledOnce();
    vi.unstubAllGlobals();
  });
});
