/**
 * Help-system contract tests (#4120 V4): every tab has substantive
 * "How to Use This Page" content for a cold user.
 */

import { describe, expect, it } from "vitest";

import { HELP_TEXTS } from "./helptext";
import { FIELD_GUIDANCE } from "./units";

/** The App's TABS labels (pinned; App.tsx renders HELP_TEXTS[tab]). */
const TABS = [
  "Explorer",
  "Calculation Description",
  "Simulation",
  "Plots",
  "Flight Explorer",
  "Ground Surfaces",
  "Ground Playback",
  "Launch Monitor Analytics",
  "Shot Optimizer",
  "Variation",
  "Putting",
  "Glossary",
  "Neural Model Lab",
] as const;

describe("per-tab help", () => {
  it("documents reference frames for directional engineering inputs", () => {
    for (const key of [
      "clubheadSpeedMph",
      "omegaPlaneDps",
      "omegaShaftDps",
      "planeYawDeg",
      "planeSideTiltDeg",
      "planeForwardTiltDeg",
      "fxLaunchDirection",
      "fxSpinAxisTilt",
    ]) {
      expect(FIELD_GUIDANCE[key], key).toContain("Reference frame:");
    }
  });

  it("every tab has an entry", () => {
    for (const tab of TABS) {
      expect(HELP_TEXTS[tab], tab).toBeDefined();
    }
    expect(Object.keys(HELP_TEXTS).sort()).toEqual([...TABS].sort());
  });

  it("every entry is substantive (>300 chars) and titled", () => {
    for (const [key, entry] of Object.entries(HELP_TEXTS)) {
      const total = entry.paragraphs.join(" ");
      expect(total.length, key).toBeGreaterThan(300);
      expect(entry.title, key).toBe("How to Use This Page");
      expect(entry.paragraphs.length, key).toBeGreaterThanOrEqual(2);
    }
  });
});
