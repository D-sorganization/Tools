import { describe, it, expect, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useRef } from "react";
import { useNonPassiveWheel } from "./useNonPassiveWheel";

describe("useNonPassiveWheel", () => {
  it("attaches a non-passive wheel listener and calls the handler", () => {
    const el = document.createElement("div");
    const add = vi.spyOn(el, "addEventListener");
    const handler = vi.fn();
    renderHook(() => {
      const ref = useRef<HTMLDivElement | null>(el);
      useNonPassiveWheel(ref, handler);
    });
    expect(add).toHaveBeenCalledWith("wheel", expect.any(Function), {
      passive: false,
    });
    el.dispatchEvent(new Event("wheel"));
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("removes the listener on unmount", () => {
    const el = document.createElement("div");
    const remove = vi.spyOn(el, "removeEventListener");
    const { unmount } = renderHook(() => {
      const ref = useRef<HTMLDivElement | null>(el);
      useNonPassiveWheel(ref, () => {});
    });
    unmount();
    expect(remove).toHaveBeenCalledWith("wheel", expect.any(Function));
  });

  it("does nothing when the ref is null", () => {
    expect(() =>
      renderHook(() => {
        const ref = useRef<HTMLDivElement | null>(null);
        useNonPassiveWheel(ref, () => {});
      }),
    ).not.toThrow();
  });
});
