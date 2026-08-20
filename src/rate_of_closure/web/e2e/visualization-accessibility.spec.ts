import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { visualizationAccessibilityTabs } from "../src/model/visualizationAccessibilityManifest";
import { visualizationTabs } from "../src/model/visualizationTabManifest";
import { capturePageErrors } from "./variationTestSupport";

const visibility = new Map(
  visualizationTabs("react").map((entry) => [entry.tabId, entry]),
);

for (const authority of visualizationAccessibilityTabs("react")) {
  test(`@trusted-isolated ${authority.tabId} has no detectable WCAG A or AA violation`, async (
    { page }, testInfo,
  ) => {
    test.skip(testInfo.project.name !== "chromium-desktop", "single Chromium AT authority");
    const entry = visibility.get(authority.tabId);
    expect(entry, `visibility authority for ${authority.tabId}`).toBeDefined();
    if (entry === undefined) return;

    const errors = capturePageErrors(page);
    await page.goto("/");
    await page.locator(`#primary-tab-${entry.tabId}`).click();
    await expect(page.locator(entry.primaryVisualLocator)).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();

    await testInfo.attach(`react-primary-tab-accessibility-${entry.tabId}`, {
      body: Buffer.from(JSON.stringify({
        policy: "protected-automated-semantics-not-manual-at-qualification",
        engine: "axe-core-4.13.0-wcag-a-aa-through-2.2",
        tabId: entry.tabId,
        violations: results.violations,
      }, null, 2)),
      contentType: "application/json",
    });
    expect(errors).toEqual([]);
    expect(results.violations, `${entry.tabId} accessibility violations`).toEqual([]);
  });
}
