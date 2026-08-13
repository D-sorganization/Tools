import { expect, test } from "@playwright/test";

import { capturePageErrors } from "./variationTestSupport";

test("linked scatter selects retained rows without recomputing analysis", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium-desktop", "linked-scatter authority");
  const pageErrors = capturePageErrors(page);
  await page.goto("/");
  await page.locator("#primary-tab-launch-monitor-analytics").click();
  const scatter = page.locator("svg[aria-label$=' linked scatter plot']");
  await expect(scatter).toBeVisible();
  await scatter.focus();
  await scatter.press("End");
  await expect(page.locator("#linked-scatter-status"))
    .toContainText("Retained row index 119 (zero-based)");
  await expect(page.locator("circle[aria-label='Selected retained row 119']")).toHaveCount(1);
  await scatter.press("Escape");
  await expect(page.locator("#linked-scatter-status")).toContainText("No retained source row selected");
  await scatter.press("Home");
  await expect(page.locator("#linked-scatter-status"))
    .toContainText("Retained row index 0 (zero-based)");
  await testInfo.attach("linked-scatter-selected-row", {
    body: await page.screenshot(), contentType: "image/png",
  });
  expect(pageErrors).toEqual([]);
});
