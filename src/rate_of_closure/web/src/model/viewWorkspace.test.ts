import { describe, expect, it } from "vitest";

import {
  LEGACY_VIEW_WORKSPACE_STORAGE_KEY,
  VIEW_WORKSPACE_STORAGE_KEY,
  exportViewWorkspace,
  importViewWorkspace,
  loadViewWorkspace,
  migrateViewWorkspace,
  saveViewWorkspace,
  toggleWorkspaceView,
  workspaceForSingleView,
  workspaceWithLayout,
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
    expect(workspace.slots.map(({ legend }) => legend)).toEqual([
      "hidden",
      "outside_right",
    ]);
    expect(workspace.activeSlotId).toBe("swing");
    expect(workspace.layout).toBe("split_horizontal");
    expect(workspace.playback).toEqual({ timeS: 0.42, playing: false, loop: true, rate: 0.5 });
  });

  it("normalizes layout cardinality and preserves per-view legend state", () => {
    const recovered = migrateViewWorkspace({
      layout: "split_vertical",
      slots: [
        { id: "impact", kind: "impact", legend: "hidden" },
        { id: "swing", kind: "swing", legend: "outside_right" },
        { id: "flight", kind: "flight", legend: "hidden" },
      ],
      active_slot_id: "impact",
    });

    expect(recovered.layout).toBe("grid");
    const single = workspaceForSingleView("impact", recovered);
    expect(single.layout).toBe("single");
    expect(single.slots[0].legend).toBe("hidden");

    const split = toggleWorkspaceView(recovered, "flight");
    expect(split.layout).toBe("split_horizontal");
    expect(split.slots.map(({ legend }) => legend)).toEqual([
      "hidden",
      "outside_right",
    ]);

    const grid = workspaceWithLayout("grid", split);
    expect(grid.layout).toBe("grid");
    expect(grid.slots.map(({ legend }) => legend)).toEqual([
      "hidden",
      "outside_right",
      "outside_right",
    ]);
  });

  it("migrates the legacy visible-view list and persists the canonical document", () => {
    const storage = new Map<string, string>();
    storage.set(LEGACY_VIEW_WORKSPACE_STORAGE_KEY, JSON.stringify({
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
      format: "rate_of_closure.view_workspace/2",
      layout: "split_horizontal",
      active_slot_id: "impact",
      camera_preferences: { format: "camera-preferences/v1" },
    });
  });

  it("round-trips a strict versioned export without partial future-version recovery", () => {
    const workspace = migrateViewWorkspace({
      format: "rate_of_closure.view_workspace/1",
      layout: "grid",
      slots: [
        { id: "impact", kind: "impact", legend: "hidden" },
        { id: "swing", kind: "swing", legend: "outside_right" },
        { id: "flight", kind: "flight", legend: "outside_right" },
      ],
      active_slot_id: "flight",
      playback: { time_s: 0.42, playing: false, loop: true, rate: 0.5 },
    });

    const text = exportViewWorkspace(workspace);

    expect(text.endsWith("\n")).toBe(true);
    expect(importViewWorkspace(text)).toEqual(workspace);
    const future = JSON.stringify({ ...JSON.parse(text), format: "future/9" });
    expect(() => importViewWorkspace(future)).toThrow(/unsupported workspace format/);
  });
});
