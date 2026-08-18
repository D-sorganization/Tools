import { describe, expect, it } from "vitest";
import accessibilityDocument from "../../../visualization_accessibility.v1.json" with { type: "json" };

import { PRIMARY_VIEW_IDS } from "./viewPreferences";
import {
  parseVisualizationAccessibilityManifest,
  visualizationAccessibilityTabs,
  visualizationManualAt,
} from "./visualizationAccessibilityManifest";

describe("visualization accessibility manifest", () => {
  it("covers every React tab and preserves the human qualification boundary", () => {
    expect(visualizationAccessibilityTabs("react").map((entry) => entry.tabId))
      .toEqual(PRIMARY_VIEW_IDS);
    expect(visualizationAccessibilityTabs("react").every(
      (entry) => entry.evidence === "axe-core-wcag-a-aa-through-2.2",
    )).toBe(true);
    expect(visualizationManualAt).toEqual({
      status: "protocol-ready-human-execution-required",
      protocolPath: "docs/development/rate-visualization-at-protocol.md",
    });
  });

  it("rejects coverage drift, false evidence, and mutable publication", () => {
    const missing = structuredClone(accessibilityDocument);
    missing.tabs.pop();
    expect(() => parseVisualizationAccessibilityManifest(missing)).toThrow(/visibility authority/);

    const falseEvidence = structuredClone(accessibilityDocument);
    falseEvidence.tabs[0].evidence = "manual-screen-reader-approved";
    expect(() => parseVisualizationAccessibilityManifest(falseEvidence)).toThrow(/unsupported/);

    const falseProtocol = structuredClone(accessibilityDocument);
    falseProtocol.manual_at.protocol_path = "docs/approved-at-result.md";
    expect(() => parseVisualizationAccessibilityManifest(falseProtocol)).toThrow(/unsupported/);

    const parsed = parseVisualizationAccessibilityManifest(accessibilityDocument);
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(Object.isFrozen(parsed.tabs)).toBe(true);
    expect(Object.isFrozen(parsed.manualAt)).toBe(true);
  });
});
