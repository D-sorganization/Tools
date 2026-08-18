import { describe, expect, it } from "vitest";

import {
  DEFAULT_VISUAL_LAYOUT,
  loadVisualLayout,
  parseVisualLayout,
  saveVisualLayout,
  VISUAL_LAYOUT_STORAGE_KEY,
  visualLayoutDocument,
} from "./visualLayoutPreferences";

describe("visual layout preferences", () => {
  it("round trips exact presentation-only state", () => {
    const expected = parseVisualLayout({
      version: 1,
      clubCamera: { azimuthDeg: -35, elevationDeg: 42, zoom: 2.5 },
      moduleHelpOpen: true,
      shellSidebarFraction: 0.31,
    });
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => { values.set(key, value); },
    };

    expect(saveVisualLayout(expected, storage)).toBe(true);
    expect(loadVisualLayout(storage)).toEqual(expected);
    expect(JSON.parse(values.get(VISUAL_LAYOUT_STORAGE_KEY) ?? "")).toEqual(
      visualLayoutDocument(expected),
    );
  });

  it.each([
    ["version", { ...visualLayoutDocument(DEFAULT_VISUAL_LAYOUT), version: 2 }],
    ["azimuth", { ...visualLayoutDocument(DEFAULT_VISUAL_LAYOUT), clubCamera: {
      ...DEFAULT_VISUAL_LAYOUT.clubCamera, azimuthDeg: 180,
    } }],
    ["help", { ...visualLayoutDocument(DEFAULT_VISUAL_LAYOUT), moduleHelpOpen: 1 }],
    ["split", { ...visualLayoutDocument(DEFAULT_VISUAL_LAYOUT), shellSidebarFraction: 0.5 }],
  ])("rejects forged %s state", (_name, document) => {
    expect(() => parseVisualLayout(document)).toThrow();
  });

  it("fails closed on corruption and bounds storage failures", () => {
    expect(loadVisualLayout({ getItem: () => "not-json" })).toEqual(DEFAULT_VISUAL_LAYOUT);
    expect(saveVisualLayout(DEFAULT_VISUAL_LAYOUT, {
      setItem: () => { throw new Error("read-only"); },
    })).toBe(false);
  });
});
