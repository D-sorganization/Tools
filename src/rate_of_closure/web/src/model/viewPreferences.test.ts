import { describe, expect, it } from "vitest";

import {
  DEFAULT_PRIMARY_VIEW_STATE,
  PRIMARY_VIEW_IDS,
  loadPrimaryViewState,
  movePrimaryView,
  savePrimaryViewState,
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
    storage.setItem("rate-of-closure.web.primary-views.v1", "{not-json");
    expect(loadPrimaryViewState(storage)).toEqual(DEFAULT_PRIMARY_VIEW_STATE);

    storage.setItem(
      "rate-of-closure.web.primary-views.v1",
      JSON.stringify({ version: 999, order: ["plots"], active: "plots" }),
    );
    expect(loadPrimaryViewState(storage)).toEqual(DEFAULT_PRIMARY_VIEW_STATE);
  });

  it("removes unknowns and duplicates while appending newly introduced views", () => {
    const storage = new MemoryStorage();
    storage.setItem(
      "rate-of-closure.web.primary-views.v1",
      JSON.stringify({
        version: 1,
        order: ["plots", "unknown", "plots", "simulation"],
        active: "plots",
      }),
    );

    const loaded = loadPrimaryViewState(storage);
    expect(loaded.order.slice(0, 2)).toEqual(["plots", "simulation"]);
    expect(new Set(loaded.order)).toEqual(new Set(PRIMARY_VIEW_IDS));
    expect(loaded.order).toContain("launch-monitor-analytics");
    expect(loaded.order).toContain("neural-model-lab");
  });

  it("round-trips a valid reordered state", () => {
    const storage = new MemoryStorage();
    const order = movePrimaryView([...PRIMARY_VIEW_IDS], "simulation", "explorer");
    const state = { version: 1 as const, order, active: "simulation" as const };

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
});
