import { describe, it, expect, beforeEach } from "vitest";

import {
  clampPanelHeight,
  reconcilePanelOrder,
  movePanel,
  loadPanelLayout,
  savePanelLayout,
  MIN_PANEL_HEIGHT,
  MAX_PANEL_HEIGHT,
} from "./panelLayout";

describe("clampPanelHeight", () => {
  it("clamps to the [min, max] range and rounds", () => {
    expect(clampPanelHeight(10)).toBe(MIN_PANEL_HEIGHT);
    expect(clampPanelHeight(99999)).toBe(MAX_PANEL_HEIGHT);
    expect(clampPanelHeight(300.6)).toBe(301);
  });

  it("throws TypeError on a non-finite height (DbC)", () => {
    expect(() => clampPanelHeight(Number.NaN)).toThrow(TypeError);
    // @ts-expect-error deliberate wrong type
    expect(() => clampPanelHeight("300")).toThrow(TypeError);
  });
});

describe("reconcilePanelOrder", () => {
  const known = ["a", "b", "c"];

  it("keeps the saved order for ids that still exist", () => {
    expect(reconcilePanelOrder(["c", "a", "b"], known)).toEqual(["c", "a", "b"]);
  });

  it("drops unknown ids and de-dupes", () => {
    expect(reconcilePanelOrder(["c", "zz", "c", "a"], known)).toEqual(["c", "a", "b"]);
  });

  it("appends panels added since the order was saved", () => {
    expect(reconcilePanelOrder(["b"], known)).toEqual(["b", "a", "c"]);
  });

  it("returns the declared order for an empty saved list", () => {
    expect(reconcilePanelOrder([], known)).toEqual(["a", "b", "c"]);
  });

  it("throws TypeError on non-array input (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => reconcilePanelOrder(null, known)).toThrow(TypeError);
  });
});

describe("movePanel", () => {
  const order = ["a", "b", "c", "d"];

  it("moves a panel to sit immediately before the target", () => {
    expect(movePanel(order, "d", "b")).toEqual(["a", "d", "b", "c"]);
    expect(movePanel(order, "a", "d")).toEqual(["b", "c", "a", "d"]);
  });

  it("is a no-op copy when dragged === target", () => {
    const out = movePanel(order, "b", "b");
    expect(out).toEqual(order);
    expect(out).not.toBe(order);
  });

  it("is a no-op when either id is absent", () => {
    expect(movePanel(order, "z", "b")).toEqual(order);
    expect(movePanel(order, "b", "z")).toEqual(order);
  });

  it("throws TypeError on bad arguments (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => movePanel("nope", "a", "b")).toThrow(TypeError);
    // @ts-expect-error deliberate wrong type
    expect(() => movePanel(order, 1, "b")).toThrow(TypeError);
  });
});

describe("loadPanelLayout / savePanelLayout", () => {
  const known = ["trend", "monitors"];

  beforeEach(() => {
    localStorage.clear();
  });

  it("returns the declared order and no heights when nothing is stored", () => {
    expect(loadPanelLayout("trends", known)).toEqual({
      order: ["trend", "monitors"],
      heights: {},
    });
  });

  it("round-trips a saved layout", () => {
    savePanelLayout("trends", { order: ["monitors", "trend"], heights: { trend: 400 } });
    expect(loadPanelLayout("trends", known)).toEqual({
      order: ["monitors", "trend"],
      heights: { trend: 400 },
    });
  });

  it("reconciles a stored order against the current panels", () => {
    savePanelLayout("trends", { order: ["gone", "monitors"], heights: {} });
    // "gone" dropped, "trend" (new) appended.
    expect(loadPanelLayout("trends", known).order).toEqual(["monitors", "trend"]);
  });

  it("clamps stored heights and ignores non-finite ones", () => {
    savePanelLayout("trends", {
      order: known.slice(),
      heights: { trend: 99999, monitors: Number.POSITIVE_INFINITY },
    });
    const out = loadPanelLayout("trends", known);
    expect(out.heights.trend).toBe(MAX_PANEL_HEIGHT);
    expect(out.heights.monitors).toBeUndefined();
  });

  it("keeps regions independent", () => {
    savePanelLayout("trends", { order: ["monitors", "trend"], heights: {} });
    savePanelLayout("power", { order: ["trend", "monitors"], heights: {} });
    expect(loadPanelLayout("trends", known).order).toEqual(["monitors", "trend"]);
    expect(loadPanelLayout("power", known).order).toEqual(["trend", "monitors"]);
  });

  it("falls back to defaults on corrupt storage", () => {
    localStorage.setItem("p1am.panelLayout.trends.v1", "{not json");
    expect(loadPanelLayout("trends", known).order).toEqual(known);
  });
});
