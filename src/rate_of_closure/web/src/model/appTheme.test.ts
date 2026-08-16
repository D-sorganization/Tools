import { describe, expect, it } from "vitest";

import {
  APP_THEME_STORAGE_KEY,
  applyAppTheme,
  loadAppTheme,
  saveAppTheme,
} from "./appTheme";

describe("application theme", () => {
  it("loads only supported persisted values and defaults deterministically", () => {
    expect(loadAppTheme({ getItem: () => "light" })).toBe("light");
    expect(loadAppTheme({ getItem: () => "sepia" })).toBe("dark");
    expect(loadAppTheme(null)).toBe("dark");
  });

  it("persists without surfacing unavailable-storage failures", () => {
    const values = new Map<string, string>();
    expect(saveAppTheme("light", { setItem: (key, value) => values.set(key, value) }))
      .toBe(true);
    expect(values.get(APP_THEME_STORAGE_KEY)).toBe("light");
    expect(saveAppTheme("dark", { setItem: () => { throw new Error("blocked"); } }))
      .toBe(false);
  });

  it("applies a semantic root theme without overwriting unrelated classes", () => {
    const root = document.createElement("html");
    root.className = "existing";
    applyAppTheme("light", root);
    expect(root.dataset.theme).toBe("light");
    expect(root.style.colorScheme).toBe("light");
    expect(root).toHaveClass("existing");
  });
});
