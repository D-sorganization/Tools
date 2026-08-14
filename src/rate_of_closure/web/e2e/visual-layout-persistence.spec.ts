import { expect, test, type Locator } from "@playwright/test";

const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "narrow", width: 390, height: 844 },
] as const;

const meaningfulIntersection = async (locator: Locator) => {
  const box = await locator.boundingBox();
  if (box === null) return null;
  const viewport = locator.page().viewportSize();
  if (viewport === null) return null;
  return {
    width: Math.max(0, Math.min(box.x + box.width, viewport.width) - Math.max(box.x, 0)),
    height: Math.max(0, Math.min(box.y + box.height, viewport.height) - Math.max(box.y, 0)),
  };
};

for (const viewport of viewports) {
  test(`restores visual layout without hiding the primary visual (${viewport.name})`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.evaluate(() => window.localStorage.clear());
    await page.reload();

    const clubCanvas = page.getByRole("img", { name: "Interactive 3D clubhead camera" });
    await clubCanvas.focus();
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("+");
    const cameraStatus = page.getByText(/camera azimuth 145°.*zoom 1\.10×/);
    await expect(cameraStatus).toBeVisible();

    const help = page.locator("details[title='Usage instructions for this page']");
    await help.locator("summary").click();
    await expect(help).toHaveAttribute("open", "");
    await page.getByRole("tab", { name: "Flight Explorer" }).click();
    await expect(page.getByRole("tab", { name: "Flight Explorer" }))
      .toHaveAttribute("aria-selected", "true");

    await page.reload();
    await expect(page.getByRole("tab", { name: "Flight Explorer" }))
      .toHaveAttribute("aria-selected", "true");
    await expect(help).toHaveAttribute("open", "");
    const flightVisual = page.locator(
      "canvas[aria-label='Flight side profile (height vs carry)']",
    );
    const intersection = await meaningfulIntersection(flightVisual);
    expect(intersection).not.toBeNull();
    expect(intersection?.height).toBeGreaterThanOrEqual(viewport.name === "narrow" ? 180 : 240);
    expect(intersection?.width).toBeGreaterThanOrEqual(120);

    await page.getByRole("tab", { name: "Explorer", exact: true }).click();
    await expect(page.getByText(/camera azimuth 145°.*zoom 1\.10×/)).toBeVisible();
    await expect(clubCanvas).toBeInViewport();
  });
}
