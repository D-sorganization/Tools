import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

import { usePanelLayout } from "./usePanelLayout";
import { loadPanelLayout, MAX_PANEL_HEIGHT } from "../lib/panelLayout";

describe("usePanelLayout", () => {
  const ids = ["a", "b", "c"];

  beforeEach(() => {
    localStorage.clear();
  });

  it("starts in the declared order with no custom heights", () => {
    const { result } = renderHook(() => usePanelLayout("r", ids));
    expect(result.current.order).toEqual(["a", "b", "c"]);
    expect(result.current.heights).toEqual({});
  });

  it("reorders and persists the new order", () => {
    const { result } = renderHook(() => usePanelLayout("r", ids));
    act(() => result.current.move("c", "a"));
    expect(result.current.order).toEqual(["c", "a", "b"]);
    expect(loadPanelLayout("r", ids).order).toEqual(["c", "a", "b"]);
  });

  it("sets, clamps and persists a height", () => {
    const { result } = renderHook(() => usePanelLayout("r", ids));
    act(() => result.current.setHeight("b", 99999));
    expect(result.current.heights.b).toBe(MAX_PANEL_HEIGHT);
    expect(loadPanelLayout("r", ids).heights.b).toBe(MAX_PANEL_HEIGHT);
  });

  it("reset restores the declared order and clears heights", () => {
    const { result } = renderHook(() => usePanelLayout("r", ids));
    act(() => {
      result.current.move("c", "a");
      result.current.setHeight("b", 300);
    });
    act(() => result.current.reset());
    expect(result.current.order).toEqual(ids);
    expect(result.current.heights).toEqual({});
  });

  it("keeps setters referentially stable across renders", () => {
    const { result, rerender } = renderHook(() => usePanelLayout("r", ids));
    const first = result.current.move;
    rerender();
    expect(result.current.move).toBe(first);
  });
});
