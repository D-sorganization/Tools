import { describe, it, expect } from "vitest";
import { HELP, SYSTEM_OVERVIEW } from "./helpContent";
import { TABS } from "../lib/tabs";

describe("help content", () => {
  it("provides a non-empty, titled doc for every HMI tab", () => {
    for (const tab of TABS) {
      const doc = HELP[tab.id];
      expect(doc, `missing help for ${tab.id}`).toBeTruthy();
      expect(doc.title.length, tab.id).toBeGreaterThan(0);
      expect(doc.body.trim().length, tab.id).toBeGreaterThan(40);
    }
  });

  it("covers exactly the defined tab ids (no orphans, no gaps)", () => {
    expect(Object.keys(HELP).sort()).toEqual(TABS.map((t) => t.id).sort());
  });

  it("includes a system overview that names the PLC", () => {
    expect(SYSTEM_OVERVIEW).toContain("P1AM-100");
  });
});
