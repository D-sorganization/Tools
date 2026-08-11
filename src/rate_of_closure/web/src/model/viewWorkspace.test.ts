import { describe, expect, it } from "vitest";

import {
  VIEW_WORKSPACE_STORAGE_KEY,
  loadViewWorkspace,
  migrateViewWorkspace,
  saveViewWorkspace,
} from "./viewWorkspace";

describe("view workspace persistence", () => {
  it("drops unknown view identities and selects a deterministic active fallback", () => {
    const workspace = migrateViewWorkspace({
      format: "rate_of_closure.view_workspace/1",
      layout: "grid",
      slots: [
        { id: "future", kind: "future", plot_id: null, legend: "outside_right" },
        { id: "swing", kind: "swing", plot_id: null, legend: "hidden" },
        { id: "flight", kind: "flight", plot_id: null, legend: "outside_right" },
      ],
      active_slot_id: "future",
      playback: { time_s: 0.42, playing: false, loop: true, rate: 0.5 },
    });

    expect(workspace.slots.map(({ id }) => id)).toEqual(["swing", "flight"]);
    expect(workspace.activeSlotId).toBe("swing");
    expect(workspace.layout).toBe("grid");
    expect(workspace.playback).toEqual({ timeS: 0.42, playing: false, loop: true, rate: 0.5 });
  });

  it("migrates the legacy visible-view list and persists the canonical document", () => {
    const storage = new Map<string, string>();
    storage.set(VIEW_WORKSPACE_STORAGE_KEY, JSON.stringify({
      version: 1,
      layout: "split_horizontal",
      views: ["impact", "future", "flight"],
      active: "future",
    }));
    const adapter = {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => { storage.set(key, value); },
    };

    const migrated = loadViewWorkspace(adapter);
    expect(migrated.slots.map(({ id }) => id)).toEqual(["impact", "flight"]);
    expect(migrated.activeSlotId).toBe("impact");
    expect(saveViewWorkspace(migrated, adapter)).toBe(true);
    expect(JSON.parse(storage.get(VIEW_WORKSPACE_STORAGE_KEY) ?? "{}")).toMatchObject({
      format: "rate_of_closure.view_workspace/1",
      layout: "split_horizontal",
      active_slot_id: "impact",
    });
  });
});
