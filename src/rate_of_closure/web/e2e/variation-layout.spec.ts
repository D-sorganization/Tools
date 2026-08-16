import { expect, test } from "@playwright/test";

import { capturePageErrors, openVariation } from "./variationTestSupport";

test("variation workspace fits the deterministic desktop and narrow viewports", async (
  { page },
  testInfo,
) => {
  const pageErrors = capturePageErrors(page);
  await openVariation(page);

  await expect(page.getByRole("heading", { name: "Study Setup" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run Variation Study" })).toBeVisible();
  const overflow = await page.evaluate(() => ({
    body: document.body.scrollWidth - document.body.clientWidth,
    root: document.documentElement.scrollWidth - document.documentElement.clientWidth,
  }));
  expect(overflow).toEqual({ body: 0, root: 0 });

  const screenshot = await page.screenshot({
    animations: "disabled",
    caret: "hide",
    fullPage: true,
  });
  expect(screenshot.byteLength).toBeGreaterThan(10_000);
  await testInfo.attach(`variation-${testInfo.project.name}`, {
    body: screenshot,
    contentType: "image/png",
  });
  expect(pageErrors).toEqual([]);
});
