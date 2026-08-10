import { expect, test, type Page } from "@playwright/test";

async function openFlightPlayback(page: Page) {
  await page.goto("/");
  await page.getByRole("tab", { name: "Flight Explorer" }).click();
  await page.getByRole("button", { name: "Run Flight" }).click();
  await expect(page.getByRole("button", { name: "Play Ball Flight" })).toBeEnabled();
  return page.getByRole("region", { name: "3D ball-flight playback" });
}

test("camera commands remain coherent across a bounded playback interaction matrix", async ({ page }) => {
  const playback = await openFlightPlayback(page);
  const timeline = playback.getByRole("slider", { name: "Ball Flight Time" });

  await playback.getByRole("checkbox", { name: "Track Ball" }).check();
  await playback.getByRole("button", { name: "Overhead" }).click();
  await expect(playback.getByRole("button", { name: "Overhead" }))
    .toHaveAttribute("aria-pressed", "true");
  await playback.getByRole("combobox", { name: "Playback Speed" }).selectOption("4");
  await playback.getByRole("checkbox", { name: "Loop Ball Flight Playback" }).check();

  await playback.getByRole("button", { name: "Step Forward One Frame" }).click();
  await expect.poll(async () => Number(await timeline.inputValue())).toBeGreaterThan(0);
  await playback.getByRole("button", { name: "Step Back One Frame" }).click();
  await expect(timeline).toHaveValue("0");

  await playback.getByRole("button", { name: "Play Ball Flight" }).click();
  await expect(playback.getByRole("button", { name: "Pause Ball Flight" })).toBeVisible();
  await playback.getByRole("button", { name: "Pause Ball Flight" }).click();
  await playback.getByRole("button", { name: "Restart Ball Flight" }).click();
  await expect(playback.getByRole("button", { name: "Pause Ball Flight" })).toBeVisible();
  await playback.getByRole("button", { name: "Pause Ball Flight" }).click();

  const canvas = playback.getByLabel("Interactive 3D ball-flight playback");
  await canvas.hover();
  await page.mouse.wheel(0, -120);
  await canvas.dragTo(canvas, {
    sourcePosition: { x: 120, y: 100 },
    targetPosition: { x: 165, y: 125 },
  });
  await expect(playback.getByRole("status", { name: "Camera tracking state" }))
    .toContainText("Tracking suspended");
  await playback.getByRole("button", { name: "Re-center Ball" }).click();
  await expect(playback.getByRole("status", { name: "Camera tracking state" }))
    .toHaveText("Tracking Ball");
  await playback.getByRole("button", { name: "Face On" }).click();
  await expect(playback.getByRole("button", { name: "Face On" }))
    .toHaveAttribute("aria-pressed", "true");
});

test("camera controls and backing canvas remain usable at the project viewport and DPR", async ({ page }, testInfo) => {
  const playback = await openFlightPlayback(page);
  const controls = playback.getByLabel("Ball camera controls");
  const canvas = playback.getByLabel("Interactive 3D ball-flight playback");

  await expect(controls).toBeVisible();
  const geometry = await controls.evaluate((element) => ({
    right: element.getBoundingClientRect().right,
    viewport: document.documentElement.clientWidth,
    scrollWidth: element.scrollWidth,
    clientWidth: element.clientWidth,
  }));
  expect(geometry.right).toBeLessThanOrEqual(geometry.viewport);
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);

  const pixels = await canvas.evaluate((element: HTMLCanvasElement) => ({
    backingWidth: element.width,
    cssWidth: element.getBoundingClientRect().width,
    dpr: window.devicePixelRatio,
  }));
  const expectedDpr = testInfo.project.name === "chromium-constrained-hidpi" ? 2 : 1;
  expect(pixels.dpr).toBe(expectedDpr);
  expect(pixels.backingWidth).toBeGreaterThanOrEqual(
    Math.floor(pixels.cssWidth * pixels.dpr) - 1,
  );
});
