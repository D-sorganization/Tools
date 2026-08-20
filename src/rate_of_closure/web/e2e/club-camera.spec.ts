import { expect, test } from "@playwright/test";

import { capturePageErrors } from "./variationTestSupport";

function binaryTetrahedron(): Buffer {
  const faces = [
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
  ];
  const data = Buffer.alloc(84 + faces.length * 50);
  data.writeUInt32LE(faces.length, 80);
  faces.forEach((face, faceIndex) => face.forEach((vertex, vertexIndex) =>
    vertex.forEach((value, axis) => data.writeFloatLE(
      value, 84 + faceIndex * 50 + 12 + vertexIndex * 12 + axis * 4,
    )),
  ));
  return data;
}

for (const viewport of [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 390, height: 844 },
]) {
  test(`club camera/source lifecycle at ${viewport.width}x${viewport.height}`, async (
    { page }, testInfo,
  ) => {
    test.skip(testInfo.project.name !== "chromium-desktop", "canonical club viewports");
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize(viewport);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    const canvas = page.getByRole("img", { name: /Interactive 3D clubhead camera/ });
    const status = page.getByRole("status").filter({ hasText: "camera azimuth" });
    await expect(canvas).toBeVisible();
    await expect(page.getByRole("button", { name: "Play" })).toBeVisible();
    await canvas.focus();
    await canvas.press("ArrowLeft");
    await expect(status).toContainText("azimuth 145°");
    await testInfo.attach(`club-generated-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(), contentType: "image/png",
    });

    const input = page.locator("input[type=file][accept='.stl']");
    await input.setInputFiles({ name: "bounded-head.stl", mimeType: "model/stl",
      buffer: binaryTetrahedron() });
    await expect(page.getByRole("status", { name: "" }).filter({ hasText: "Reading" }))
      .toHaveCount(0);
    await expect(status).toContainText("Imported bounded-head.stl");
    await expect(status).toContainText("azimuth 145°");
    const importedStatus = await status.textContent();
    await testInfo.attach(`club-imported-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(), contentType: "image/png",
    });

    await page.getByRole("tab", { name: "Putting" }).click();
    await page.getByRole("tab", { name: "Explorer", exact: true }).click();
    await expect(status).toHaveText(importedStatus ?? "");
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(
      () => requestAnimationFrame(() => resolve()),
    )));
    await page.waitForTimeout(100);
    const retainedCanvas = await canvas.evaluate((element) => element.toDataURL());
    await input.setInputFiles({ name: "broken.stl", mimeType: "model/stl",
      buffer: Buffer.from("solid broken\nfacet nope\nendsolid broken") });
    await expect(page.getByRole("alert")).toContainText("STL load failed");
    await expect(status).toHaveText(importedStatus ?? "");
    expect(await canvas.evaluate((element) => element.toDataURL())).toBe(retainedCanvas);
    await testInfo.attach(`club-error-prior-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(), contentType: "image/png",
    });
    expect(await page.evaluate(() => document.documentElement.scrollWidth
      - document.documentElement.clientWidth)).toBe(0);
    expect(pageErrors).toEqual([]);
  });
}
