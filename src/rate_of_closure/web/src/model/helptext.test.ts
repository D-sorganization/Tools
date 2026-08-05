/**
 * Help-system contract tests (#4120 V4): every tab has substantive
 * "How to Use This Page" content for a cold user.
 */

import { describe, expect, it } from "vitest";

import { HELP_TEXTS } from "./helptext";

/** The App's TABS labels (pinned; App.tsx renders HELP_TEXTS[tab]). */
const TABS = [
  "Explorer",
  "Calculation Description",
  "Simulation",
  "Plots",
  "Flight Explorer",
  "Variation",
  "Putting",
  "Glossary",
] as const;

describe("per-tab help", () => {
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
