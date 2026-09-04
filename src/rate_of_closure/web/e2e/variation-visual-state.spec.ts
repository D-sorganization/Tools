import {
  expect, test, type Locator, type Page, type TestInfo, type Worker,
} from "@playwright/test";

import { capturePageErrors, openVariation, setNumericField } from "./variationTestSupport";

type StateName = "empty" | "loading-no-prior" | "error-empty" | "result" |
  "loading-prior" | "error-prior" | "individual-result";

const expectedState: Record<StateName, { phase: string; origin: string }> = {
  empty: { phase: "empty", origin: "empty-preview" },
  "loading-no-prior": { phase: "loading", origin: "empty-preview" },
  "error-empty": { phase: "error", origin: "empty-preview" },
  result: { phase: "result", origin: "current-accepted" },
  "loading-prior": { phase: "loading", origin: "prior-accepted" },
  "error-prior": { phase: "error", origin: "prior-accepted" },
  "individual-result": { phase: "result", origin: "current-accepted" },
};

const intersection = (locator: Locator) => locator.evaluate((element) => {
  let rect = element.getBoundingClientRect(); let ancestor = element.parentElement;
  while (ancestor !== null) {
    const style = getComputedStyle(ancestor);
    if ([style.overflow, style.overflowX, style.overflowY]
      .some((value) => ["hidden", "clip", "scroll", "auto"].includes(value))) {
      const clip = ancestor.getBoundingClientRect();
      const left = Math.max(rect.left, clip.left); const top = Math.max(rect.top, clip.top);
      const right = Math.min(rect.right, clip.right); const bottom = Math.min(rect.bottom, clip.bottom);
      rect = new DOMRect(left, top, Math.max(0, right - left), Math.max(0, bottom - top));
    }
    ancestor = ancestor.parentElement;
  }
  const left = Math.max(rect.left, 0); const top = Math.max(rect.top, 0);
  const right = Math.min(rect.right, innerWidth); const bottom = Math.min(rect.bottom, innerHeight);
  return { x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
});

async function capture(page: Page, testInfo: TestInfo, state: StateName) {
  const frame = page.locator("[data-visual-origin]").last();
  await expect(frame).toHaveAttribute("data-phase", expectedState[state].phase);
  await expect(frame).toHaveAttribute("data-visual-origin", expectedState[state].origin);
  const visual = state === "individual-result"
    ? frame.getByRole("group", { name: "One-at-a-time sensitivity matrix" })
    : ["result", "loading-prior", "error-prior"].includes(state)
      ? frame.getByRole("group", { name: "Scatter matrix with marginal histograms" })
      : frame.getByRole("img", { name: "Variation analysis workflow preview" });
  await expect(visual).toBeVisible();
  if (["result", "individual-result"].includes(state)) {
    await visual.scrollIntoViewIfNeeded();
  }
  const narrow = page.viewportSize()?.width === 390;
  const minimum = { width: narrow ? 120 : 240, height: narrow ? 180 : 240 };
  if (["empty", "result", "individual-result"].includes(state)) {
    await expect.poll(async () => {
      const box = await intersection(visual);
      return box.width >= minimum.width && box.height >= minimum.height;
    }).toBe(true);
  }
  const box = await intersection(visual);
  const controls = await page.getByRole("region", { name: "Variation setup" }).boundingBox();
  if (controls === null) throw new Error("Variation geometry unavailable");
  const body = await page.screenshot({ animations: "disabled", caret: "hide" });
  expect(body.byteLength).toBeGreaterThan(10_000);
  await testInfo.attach(`variation-${state}-${testInfo.project.name}`, {
    body, contentType: "image/png",
  });
  await testInfo.attach(`variation-${state}-${testInfo.project.name}.json`, {
    body: JSON.stringify({ state, box, controls, scrollY: await page.evaluate(() => scrollY),
      phase: await frame.getAttribute("data-phase"),
      origin: await frame.getAttribute("data-visual-origin") }),
    contentType: "application/json",
  });
  if (["empty", "result", "individual-result"].includes(state)) {
    expect(box.width).toBeGreaterThanOrEqual(minimum.width);
    expect(box.height).toBeGreaterThanOrEqual(minimum.height);
    expect(box.y + box.height <= controls.y || controls.y + controls.height <= box.y ||
      box.x + box.width <= controls.x || controls.x + controls.width <= box.x).toBe(true);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBe(0);
}

test("Variation keeps one meaningful visual through the production Worker lifecycle", async (
  { page }, testInfo,
) => {
  const pageErrors = capturePageErrors(page);
  const workers: Worker[] = [];
  let blockWorker = true;
  await page.route(/variationExecution\.worker-[\w-]+\.js$/, async (route) => {
    if (blockWorker) {
      await route.fulfill({
        status: 200,
        contentType: "application/javascript",
        body: "self.onmessage = () => {};",
      });
    } else {
      await route.continue();
    }
  });
  page.on("worker", (worker) => workers.push(worker));
  await openVariation(page);
  await capture(page, testInfo, "empty");
  await page.getByRole("combobox", { name: "Pipeline" }).selectOption("delivery");
  await setNumericField(page, "Runs", "4");
  const run = page.getByRole("button", { name: "Run Variation Study" });
  await run.click();
  const scrollAfterActivation = await page.evaluate(() => scrollY);
  await expect(page.locator("[data-phase='loading']")).toBeVisible();
  await capture(page, testInfo, "loading-no-prior");
  expect(await page.evaluate(() => scrollY)).toBe(scrollAfterActivation);
  const cancel = page.getByRole("button", { name: "Cancel Variation Study" });
  await expect(cancel).toBeInViewport();
  await cancel.click();
  const scrollAfterCancel = await page.evaluate(() => scrollY);
  await expect(page.locator("[data-phase='empty'][data-visual-origin='empty-preview']"))
    .toBeVisible();
  await expect(run).toBeEnabled();
  await expect(run).toBeInViewport();
  expect(scrollAfterCancel).toBe(scrollAfterActivation);

  blockWorker = false;
  await run.click();
  await expect(page.getByRole("status", { name: "Variation status" })).toContainText(/Done:/);
  await capture(page, testInfo, "result");
  await expect(page.getByRole("group", { name: "Scatter matrix with marginal histograms" }))
    .not.toBeFocused();
  const returnToControls = page.getByRole("button", { name: "Return to variation controls" });
  await expect(returnToControls).toBeVisible();
  await returnToControls.click();
  await expect(run).toBeInViewport();
  await expect(returnToControls).toBeFocused();

  blockWorker = true;
  await run.click();
  const retainedScroll = await page.evaluate(() => scrollY);
  await expect(page.locator("[data-phase='loading'][data-visual-origin='prior-accepted']"))
    .toBeVisible();
  await capture(page, testInfo, "loading-prior");
  expect(await page.evaluate(() => scrollY)).toBe(retainedScroll);
  await expect(cancel).toBeInViewport();
  await cancel.click();
  const retainedCancelScroll = await page.evaluate(() => scrollY);
  await expect(page.locator("[data-phase='result'][data-visual-origin='prior-accepted']"))
    .toBeVisible();
  await expect(run).toBeEnabled();
  await expect(run).toBeInViewport();
  expect(retainedCancelScroll).toBe(retainedScroll);

  await run.click();
  const retainedErrorScroll = await page.evaluate(() => scrollY);
  await expect.poll(() => workers.length).toBeGreaterThanOrEqual(4);
  const retainedWorker = workers.at(-1);
  if (retainedWorker === undefined) throw new Error("retained Worker was not created");
  await retainedWorker.evaluate(() => setTimeout(() => {
    throw new Error("diagnostic worker failure");
  }, 0));
  await expect(page.getByRole("alert", { name: "Variation status" }))
    .toContainText(/diagnostic worker failure|Variation worker failed/);
  await capture(page, testInfo, "error-prior");
  expect(await page.evaluate(() => scrollY)).toBe(retainedErrorScroll);
  await expect(run).toBeInViewport();

  await page.getByRole("combobox", { name: "Analysis execution" }).selectOption("individual");
  await expect(page.locator("[data-phase='empty'][data-visual-origin='empty-preview']"))
    .toBeVisible();
  await expect(page.getByRole("group", { name: "Scatter matrix with marginal histograms" }))
    .toHaveCount(0);
  await run.click();
  const firstErrorScroll = await page.evaluate(() => scrollY);
  await expect.poll(() => workers.length).toBeGreaterThanOrEqual(5);
  const emptyWorker = workers.at(-1);
  if (emptyWorker === undefined) throw new Error("empty Worker was not created");
  await emptyWorker.evaluate(() => setTimeout(() => {
    throw new Error("diagnostic first failure");
  }, 0));
  await expect(page.getByRole("alert", { name: "Variation status" })).toBeVisible();
  await capture(page, testInfo, "error-empty");
  expect(await page.evaluate(() => scrollY)).toBe(firstErrorScroll);
  await expect(run).toBeInViewport();

  blockWorker = false;
  await page.emulateMedia({ reducedMotion: "reduce" });
  await run.click();
  await expect(page.getByRole("status", { name: "Variation status" })).toContainText(/Done:/);
  await capture(page, testInfo, "individual-result");
  await expect(page.getByRole("button", { name: "Return to variation controls" }))
    .toBeVisible();
  expect(pageErrors).toEqual([]);
});
