import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { useAppWorkspace } from "./useAppWorkspace";
import { VISUAL_LAYOUT_STORAGE_KEY } from "../model/visualLayoutPreferences";

describe("useAppWorkspace visual layout persistence", () => {
  beforeEach(() => {
    const values = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => { values.set(key, value); },
        removeItem: (key: string) => { values.delete(key); },
        clear: () => { values.clear(); },
      },
    });
  });

  it("restores and persists help disclosure plus canonical club camera", async () => {
    window.localStorage.setItem(VISUAL_LAYOUT_STORAGE_KEY, JSON.stringify({
      version: 1,
      clubCamera: { azimuthDeg: -25, elevationDeg: 35, zoom: 1.8 },
      moduleHelpOpen: true,
      shellSidebarFraction: 0.27,
    }));
    const first = renderHook(() => useAppWorkspace());
    expect(first.result.current.moduleHelpOpen).toBe(true);
    expect(first.result.current.clubCamera).toEqual({
      azimuthDeg: -25, elevationDeg: 35, zoom: 1.8,
    });

    act(() => {
      first.result.current.setModuleHelpOpen(false);
      first.result.current.setClubCamera({
        azimuthDeg: 45, elevationDeg: -15, zoom: 2.2,
      });
    });
    await waitFor(() => expect(
      JSON.parse(window.localStorage.getItem(VISUAL_LAYOUT_STORAGE_KEY) ?? ""),
    ).toMatchObject({
      moduleHelpOpen: false,
      clubCamera: { azimuthDeg: 45, elevationDeg: -15, zoom: 2.2 },
    }));
    first.unmount();

    const second = renderHook(() => useAppWorkspace());
    expect(second.result.current.moduleHelpOpen).toBe(false);
    expect(second.result.current.clubCamera.zoom).toBe(2.2);
  });

  it("uses exact defaults for malformed persisted layout", () => {
    window.localStorage.setItem(VISUAL_LAYOUT_STORAGE_KEY, "not-json");
    const { result } = renderHook(() => useAppWorkspace());
    expect(result.current.moduleHelpOpen).toBe(false);
    expect(result.current.clubCamera).toEqual({
      azimuthDeg: 150, elevationDeg: 30, zoom: 1,
    });
  });
});
