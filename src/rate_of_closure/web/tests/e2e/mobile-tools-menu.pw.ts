import { expect, test } from "@playwright/test";

const MOBILE_VIEWPORT_WIDTH = 520;
const MOBILE_VIEWPORT_HEIGHT = 900;
const VIEWPORT_GUTTER_PX = 16;

test.use({
  viewport: { width: MOBILE_VIEWPORT_WIDTH, height: MOBILE_VIEWPORT_HEIGHT },
});

test("opened Tools menu remains readable and inside the constrained viewport", async ({ page }) => {
  await page.goto("/");

  const toolsTrigger = page.getByText("Tools", { exact: true });
  await toolsTrigger.focus();
  await page.keyboard.press("Enter");

  const menu = page.getByRole("group", { name: "Global tools" });
  await expect(menu).toBeVisible();
  await expect(menu.getByRole("button", { name: "Open Glossary" })).toBeVisible();
  await expect(menu.getByRole("button", { name: /Toggle Theme/i })).toBeVisible();
  await expect(menu.getByRole("button", { name: "Keyboard Shortcuts" })).toBeVisible();
  await expect(menu.getByRole("button", { name: "Current Module Help" })).toBeVisible();
  await expect(menu.getByText("Alt+G", { exact: true })).toBeVisible();
  await expect(menu.getByText("Alt+T", { exact: true })).toBeVisible();
  await expect(menu.getByText("F1", { exact: true })).toBeVisible();

  const geometry = await menu.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return {
      left: bounds.left,
      right: bounds.right,
      viewportWidth: document.documentElement.clientWidth,
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
    };
  });
  expect(geometry.left).toBeGreaterThanOrEqual(VIEWPORT_GUTTER_PX);
  expect(geometry.right).toBeLessThanOrEqual(
    geometry.viewportWidth - VIEWPORT_GUTTER_PX,
  );
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);
});
