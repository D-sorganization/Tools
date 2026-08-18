import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { visualizationAccessibilityTabs } from "../src/model/visualizationAccessibilityManifest";
import { visualizationTabs } from "../src/model/visualizationTabManifest";
import { capturePageErrors } from "./variationTestSupport";

test("every React primary tab has no detectable WCAG A or AA violation", async (
  { page }, testInfo,
) => {
  test.skip(testInfo.project.name !== "chromium-desktop", "single Chromium AT authority");
  const errors = capturePageErrors(page);
  await page.goto("/");
  const evidence: Array<{ tabId: string; violations: unknown[] }> = [];

  const visibility = new Map(
    visualizationTabs("react").map((entry) => [entry.tabId, entry]),
  );
  for (const authority of visualizationAccessibilityTabs("react")) {
    const entry = visibility.get(authority.tabId);
    expect(entry, `visibility authority for ${authority.tabId}`).toBeDefined();
    if (entry === undefined) continue;
    await page.locator(`#primary-tab-${entry.tabId}`).click();
    await expect(page.locator(entry.primaryVisualLocator)).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();
    evidence.push({ tabId: entry.tabId, violations: results.violations });
  }

  await testInfo.attach("react-primary-tab-accessibility", {
    body: Buffer.from(JSON.stringify({
      policy: "protected-automated-semantics-not-manual-at-qualification",
      engine: "axe-core-4.13.0-wcag-a-aa-through-2.2",
      evidence,
    }, null, 2)),
    contentType: "application/json",
  });
  expect(errors).toEqual([]);
  expect(
    evidence.filter((entry) => entry.violations.length > 0),
    "primary-tab accessibility violations",
  ).toEqual([]);
});
