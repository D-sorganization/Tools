import { describe, expect, it } from "vitest";
import manifestDocument from "../../../visualization_tabs.v1.json" with { type: "json" };

import { PRIMARY_VIEW_IDS } from "./viewPreferences";
import {
  auditRegisteredVisualizationTabs,
  parseVisualizationTabManifest,
  visualizationReferenceEnvironments,
  visualizationTabs,
} from "./visualizationTabManifest";

describe("visualization tab manifest governance", () => {
  it("covers every registered React primary tab with complete state metadata", () => {
    expect(visualizationReferenceEnvironments.react).toEqual({
      viewportPx: [1440, 900],
      additionalViewportsPx: [[1280, 720], [390, 844]],
      responsiveMinimumVisibleHeightPx: 180,
      minimumVisibleWidthPx: 240,
      responsiveMinimumVisibleWidthPx: 120,
      responsiveControlLocators: {
        explorer: "section[aria-label='Scenario inputs']",
        simulation: "section[aria-label='Simulation setup']",
        plots: "section[aria-label='Plot management']",
        flight: "section[aria-label='Flight explorer inputs']",
        "launch-monitor-analytics": "section[aria-label='Analysis contract']",
        "neural-model-lab": "section[aria-label='Neural Model Lab controls']",
        variation: "section[aria-label='Variation setup']",
        putting: "section[aria-label='Putt setup']",
      },
      dpiScales: [],
    });
    const entries = visualizationTabs("react");
    expect(entries.map((entry) => entry.tabId)).toEqual(PRIMARY_VIEW_IDS);
    expect(auditRegisteredVisualizationTabs("react", PRIMARY_VIEW_IDS)).toEqual([]);
    expect(entries.every((entry) => entry.primaryVisualLocator.length > 0)).toBe(true);
    expect(entries.every((entry) =>
      Object.keys(entry.states).sort().join("|") === "empty|error|loading|result",
    )).toBe(true);
  });

  it("reports a newly registered tab that lacks a manifest entry", () => {
    expect(auditRegisteredVisualizationTabs("react", [
      ...PRIMARY_VIEW_IDS, "future-visualization",
    ])).toEqual(["missing manifest entry for react tab future-visualization"]);
  });

  type MutableManifest = {
    tabs: Array<Record<string, unknown>>;
    reference_environments: {
      react: {
        viewport_px: number[];
        responsive_control_locators: Record<string, string>;
      };
      pyqt: Record<string, unknown>;
    };
  };
  it.each([
    ["unknown surface", (value: MutableManifest) => { value.tabs[0].surface = "desktop"; }],
    ["unknown classification", (value: MutableManifest) => {
      value.tabs[0].classification = "container";
    }],
    ["negative height", (value: MutableManifest) => {
      value.tabs[0].minimum_visible_height_px = -1;
    }],
    ["duplicate identity", (value: MutableManifest) => {
      value.tabs.push(structuredClone(value.tabs[0]));
    }],
    ["extra field", (value: MutableManifest) => { value.tabs[0].extra = true; }],
    ["classification-landmark mismatch", (value: MutableManifest) => {
      value.tabs[0].landmark_kind = "semantic-content";
      value.tabs[0].minimum_visible_height_px = 1;
    }],
    ["form-led evidence semantic downgrade", (value: MutableManifest) => {
      value.tabs[0].classification = "form-led-evidence";
      value.tabs[0].landmark_kind = "semantic-content";
      value.tabs[0].minimum_visible_height_px = 1;
    }],
    ["unbounded viewport", (value: MutableManifest) => {
      value.reference_environments.react.viewport_px[0] = Number.MAX_SAFE_INTEGER;
    }],
    ["missing responsive control", (value: MutableManifest) => {
      delete value.reference_environments.react.responsive_control_locators.putting;
    }],
    ["typo responsive control", (value: MutableManifest) => {
      const controls = value.reference_environments.react.responsive_control_locators;
      controls.puting = controls.putting;
      delete controls.putting;
    }],
    ["responsive PyQt field", (value: MutableManifest) => {
      value.reference_environments.pyqt.responsive_control_locators = {};
    }],
    ["one-pixel responsive visual height", (value: MutableManifest) => {
      const react = value.reference_environments.react as Record<string, unknown>;
      react.responsive_minimum_visible_height_px = 1;
    }],
  ])("rejects %s at the runtime boundary", (_label, tamper) => {
    const value: MutableManifest = structuredClone(manifestDocument);
    tamper(value);
    expect(() => parseVisualizationTabManifest(value)).toThrow();
  });

  it.each([2 ** 60, 2 ** 53, true, 1.5, "240", Number.POSITIVE_INFINITY])(
    "rejects non-shared pixel-domain value %s", (minimum) => {
      const value: MutableManifest = structuredClone(manifestDocument);
      value.tabs[0].minimum_visible_height_px = minimum;
      expect(() => parseVisualizationTabManifest(value)).toThrow();
    },
  );

  it("returns deeply immutable entries and reference environments", () => {
    const parsed = parseVisualizationTabManifest(manifestDocument);
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(Object.isFrozen(parsed.entries)).toBe(true);
    expect(Object.isFrozen(parsed.entries[0].states)).toBe(true);
    expect(Object.isFrozen(parsed.environments.react.viewportPx)).toBe(true);
    expect(Object.isFrozen(parsed.environments.react.responsiveControlLocators)).toBe(true);
    expect(() => { parsed.entries[0].states.empty = "tampered"; }).toThrow();
  });
});
