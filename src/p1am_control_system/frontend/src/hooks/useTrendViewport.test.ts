import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTrendViewport } from "./useTrendViewport";

const bounds = { min: 0, max: 1000 };
const opts = { defaultSpan: 100, minSpan: 10, maxSpan: 5000 };

describe("useTrendViewport", () => {
  it("follows the live edge by default", () => {
    const { result } = renderHook(() => useTrendViewport(opts));
    expect(result.current.live).toBe(true);
    expect(result.current.resolve(bounds)).toEqual({ start: 900, end: 1000 });
  });

  it("pans back into history and stops following", () => {
    const { result } = renderHook(() => useTrendViewport(opts));
    act(() => result.current.panBy(200, bounds));
    expect(result.current.live).toBe(false);
    expect(result.current.resolve(bounds)).toEqual({ start: 700, end: 800 });
  });

  it("toggles pause, and reset returns to live", () => {
    const { result } = renderHook(() => useTrendViewport(opts));
    act(() => result.current.togglePause());
    expect(result.current.paused).toBe(true);
    act(() => result.current.reset());
    expect(result.current.paused).toBe(false);
    expect(result.current.live).toBe(true);
  });

  it("drag-select zooms to the selected pixel range via pxToUnit", () => {
    const { result } = renderHook(() =>
      useTrendViewport({ ...opts, defaultSpan: 1000 }),
    );
    // plot pixels 0..500 map to domain 0..1000
    const pxToUnit = (px: number) => (px / 500) * 1000;
    act(() => result.current.startSelect(100));
    act(() => result.current.moveSelect(250));
    expect(result.current.selectionPx).toEqual({ fromPx: 100, toPx: 250 });
    act(() => result.current.endSelect(bounds, pxToUnit));
    // px 100..250 -> units 200..500
    expect(result.current.resolve(bounds)).toEqual({ start: 200, end: 500 });
    expect(result.current.selectionPx).toBeNull();
  });

  it("ignores a tiny drag below the pixel threshold", () => {
    const { result } = renderHook(() =>
      useTrendViewport({ ...opts, defaultSpan: 1000 }),
    );
    act(() => result.current.startSelect(100));
    act(() => result.current.moveSelect(101));
    act(() => result.current.endSelect(bounds, (px) => px));
    expect(result.current.resolve(bounds)).toEqual({ start: 0, end: 1000 });
  });

  it("cancelSelect drops the in-progress selection without zooming", () => {
    const { result } = renderHook(() => useTrendViewport(opts));
    act(() => result.current.startSelect(50));
    act(() => result.current.cancelSelect());
    expect(result.current.selectionPx).toBeNull();
    expect(result.current.live).toBe(true);
  });
});
