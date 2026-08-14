import { describe, expect, it } from "vitest";

import {
  DEFAULT_PRIMARY_VIEW_STATE,
  PRIMARY_VIEW_IDS,
  LEGACY_PRIMARY_VIEW_STORAGE_KEY,
  PRIMARY_VIEW_STORAGE_KEY,
  loadPrimaryViewState,
  movePrimaryView,
  restorePrimaryViewDefaults,
  savePrimaryViewState,
  setPrimaryViewVisibility,
  visiblePrimaryViewIds,
} from "./viewPreferences";

class MemoryStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

describe("primary view preferences", () => {
  it("recovers safely from corrupt or structurally invalid storage", () => {
    const storage = new MemoryStorage();
    storage.setItem(PRIMARY_VIEW_STORAGE_KEY, "{not-json");
    expect(loadPrimaryViewState(storage)).toEqual(DEFAULT_PRIMARY_VIEW_STATE);

    storage.setItem(
      PRIMARY_VIEW_STORAGE_KEY,
      JSON.stringify({ version: 999, order: ["plots"], active: "plots" }),
    );
    expect(loadPrimaryViewState(storage)).toEqual(DEFAULT_PRIMARY_VIEW_STATE);

    storage.setItem(
      PRIMARY_VIEW_STORAGE_KEY,
      JSON.stringify({ version: 2, order: ["plots"], active: "plots" }),
    );
    expect(loadPrimaryViewState(storage).visible).toEqual(PRIMARY_VIEW_IDS);
  });

  it("migrates the legacy v1 order and active view with every module visible", () => {
    const storage = new MemoryStorage();
    storage.setItem(LEGACY_PRIMARY_VIEW_STORAGE_KEY, JSON.stringify({
      version: 1,
      order: ["plots", "explorer"],
      active: "plots",
    }));

    const loaded = loadPrimaryViewState(storage);
    expect(loaded.version).toBe(2);
    expect(loaded.order.slice(0, 2)).toEqual(["plots", "explorer"]);
    expect(loaded.visible).toEqual(PRIMARY_VIEW_IDS);
    expect(loaded.active).toBe("plots");
  });

  it("removes unknowns and duplicates while appending newly introduced views", () => {
    const storage = new MemoryStorage();
    storage.setItem(
      PRIMARY_VIEW_STORAGE_KEY,
      JSON.stringify({
        version: 2,
        order: ["plots", "unknown", "plots", "simulation"],
        visible: ["plots", "unknown", "plots"],
        active: "plots",
      }),
    );

    const loaded = loadPrimaryViewState(storage);
    expect(loaded.order.slice(0, 2)).toEqual(["plots", "simulation"]);
    expect(new Set(loaded.order)).toEqual(new Set(PRIMARY_VIEW_IDS));
    expect(loaded.order).toContain("launch-monitor-analytics");
    expect(loaded.visible).toEqual(["plots", "ground-playback", "explorer"]);
  });

  it("round-trips a valid reordered state", () => {
    const storage = new MemoryStorage();
    const order = movePrimaryView([...PRIMARY_VIEW_IDS], "simulation", "explorer");
    const state = {
      version: 2 as const,
      order,
      visible: [...PRIMARY_VIEW_IDS],
      active: "simulation" as const,
    };

    expect(savePrimaryViewState(state, storage)).toBe(true);
    expect(loadPrimaryViewState(storage)).toEqual(state);
  });

  it("does not mutate order for an invalid move", () => {
    const order = [...PRIMARY_VIEW_IDS];
    expect(movePrimaryView(order, "not-a-view", "plots")).toEqual(order);
  });

  it("can place a dragged tab after the final destination", () => {
    const order = movePrimaryView(
      [...PRIMARY_VIEW_IDS],
      "explorer",
      "glossary",
      true,
    );
    expect(order[order.length - 1]).toBe("explorer");
  });

  it("protects required modules and deterministically falls back when active is hidden", () => {
    const activePlots = { ...DEFAULT_PRIMARY_VIEW_STATE, active: "plots" as const };
    const hidden = setPrimaryViewVisibility(activePlots, "plots", false);
    expect(visiblePrimaryViewIds(hidden)).not.toContain("plots");
    expect(hidden.active).toBe("explorer");

    expect(setPrimaryViewVisibility(hidden, "explorer", false)).toEqual(hidden);
    expect(visiblePrimaryViewIds(hidden)).toContain("explorer");
  });

  it("restores the complete default module order and visibility", () => {
    const customized = setPrimaryViewVisibility(
      { ...DEFAULT_PRIMARY_VIEW_STATE, order: [...PRIMARY_VIEW_IDS].reverse() },
      "plots",
      false,
    );
    expect(customized).not.toEqual(DEFAULT_PRIMARY_VIEW_STATE);
    expect(restorePrimaryViewDefaults()).toEqual(DEFAULT_PRIMARY_VIEW_STATE);
  });
});
