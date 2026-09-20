import { expect, test } from "@playwright/test";

import { capturePageErrors } from "./variationTestSupport";

test.describe("AppToolstrip popover viewport clamping (#4300)", () => {
  test("Tools popover stays within constrained 520 x 900 viewport without horizontal overflow", async (
    { page },
    testInfo,
  ) => {
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize({ width: 520, height: 900 });
    await page.goto("./");

    // Open Tools menu
    const toolsSummary = page.locator("summary").filter({ hasText: "Tools" });
    await expect(toolsSummary).toBeVisible();
    await toolsSummary.click();

    const toolsPopover = page.getByRole("group", { name: "Global tools" });
    await expect(toolsPopover).toBeVisible();

    // Verify all 4 command buttons are visible and readable
    await expect(page.getByRole("button", { name: "Open Glossary" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Toggle Theme/i })).toBeVisible();
    await expect(page.getByRole("button", { name: "Keyboard Shortcuts" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Current Module Help" })).toBeVisible();

    // Verify bounding box stays within the 520 px viewport with gutter
    const box = await toolsPopover.boundingBox();
    expect(box).not.toBeNull();
    if (box) {
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(520);
    }

    // Verify no document horizontal overflow
    const overflow = await page.evaluate(() => ({
      body: document.body.scrollWidth - document.body.clientWidth,
      root: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    }));
    expect(overflow).toEqual({ body: 0, root: 0 });

    const screenshot = await page.screenshot({
      animations: "disabled",
      caret: "hide",
    });
    await testInfo.attach("tools-popover-520x900", {
      body: screenshot,
      contentType: "image/png",
    });

    expect(pageErrors).toEqual([]);
  });

  test("File and View popovers stay within constrained 520 x 900 viewport", async ({ page }) => {
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize({ width: 520, height: 900 });
    await page.goto("./");

    // File menu
    const fileSummary = page.locator("summary").filter({ hasText: "File" });
    await fileSummary.click();
    const filePopover = page.getByRole("group", { name: "File commands" });
    await expect(filePopover).toBeVisible();
    const fileBox = await filePopover.boundingBox();
    expect(fileBox).not.toBeNull();
    if (fileBox) {
      expect(fileBox.x).toBeGreaterThanOrEqual(0);
      expect(fileBox.x + fileBox.width).toBeLessThanOrEqual(520);
    }

    // Close File and open View
    await fileSummary.click();
    const viewSummary = page.locator("summary").filter({ hasText: "View" });
    await viewSummary.click();
    const viewPopover = page.getByRole("group", { name: "Workspace modules" });
    await expect(viewPopover).toBeVisible();
    const viewBox = await viewPopover.boundingBox();
    expect(viewBox).not.toBeNull();
    if (viewBox) {
      expect(viewBox.x).toBeGreaterThanOrEqual(0);
      expect(viewBox.x + viewBox.width).toBeLessThanOrEqual(520);
    }

    expect(pageErrors).toEqual([]);
  });
});
