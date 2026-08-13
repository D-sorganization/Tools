import { expect, test, type Page, type Worker } from "@playwright/test";

import {
  capturePageErrors,
  openVariation,
  setNumericField,
} from "./variationTestSupport";

const LONG_RUN_COUNT = "500";

async function configureSeededDelivery(page: Page): Promise<void> {
  await setNumericField(page, "Runs", "3");
  await setNumericField(page, "Seed", "20260812");
  await page.getByRole("combobox", { name: "Analysis execution" })
    .selectOption("all_together");
}

async function configureLongSwing(page: Page): Promise<void> {
  await page.getByRole("combobox", { name: "Pipeline" }).selectOption("swing");
  await setNumericField(page, "Runs", LONG_RUN_COUNT);
  await page.getByRole("combobox", { name: "Analysis execution" }).selectOption("both");
}

function captureWorkers(page: Page): Worker[] {
  const workers: Worker[] = [];
  page.on("worker", (worker) => workers.push(worker));
  return workers;
}

test("production Worker completes a seeded study and reruns deterministically", async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  const workers = captureWorkers(page);
  await openVariation(page);
  await configureSeededDelivery(page);

  const runButton = page.getByRole("button", { name: "Run Variation Study" });
  const status = page.getByRole("status", { name: "Variation status" });
  await runButton.click();
  await expect(status).toContainText(/Done: \d+\/3 joint runs/);
  await expect(page.getByRole("progressbar", { name: "Variation execution progress" }))
    .toHaveAttribute("value", "3");
  await expect(page.getByRole("heading", { name: "Summary — Dispersion per Output" }))
    .toBeVisible();
  await expect.poll(() => workers.length).toBe(1);
  expect(workers[0].url()).toMatch(/variationExecution\.worker-[\w-]+\.js$/);
  const firstSummary = await page.getByRole("table").first().innerText();

  await runButton.click();
  await expect(status).toContainText(/Done: \d+\/3 joint runs/);
  await expect.poll(() => workers.length).toBe(2);
  expect(await page.getByRole("table").first().innerText()).toBe(firstSummary);
  expect(pageErrors).toEqual([]);
});

test("cancelling a long Worker run rejects every partial and stale result", async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  const workers = captureWorkers(page);
  await openVariation(page);
  await configureLongSwing(page);

  await page.getByRole("button", { name: "Run Variation Study" }).click();
  await expect.poll(() => workers.length).toBe(1);
  const cancelButton = page.getByRole("button", { name: "Cancel Variation Study" });
  await expect(cancelButton).toBeEnabled();
  await cancelButton.click();

  const status = page.getByRole("status", { name: "Variation status" });
  await expect(status).toHaveText("Cancelled: no partial variation result was accepted.");
  await expect(page.getByRole("heading", { name: "Ready to Analyze Variation" })).toBeVisible();
  await page.waitForTimeout(300);
  await expect(status).toHaveText("Cancelled: no partial variation result was accepted.");
  await expect(page.getByRole("heading", { name: "Summary — Dispersion per Output" }))
    .toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("primary-view navigation terminates an active Worker before unmount", async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  const workers = captureWorkers(page);
  await openVariation(page);
  await configureLongSwing(page);
  await page.getByRole("button", { name: "Run Variation Study" }).click();
  await expect.poll(() => workers.length).toBe(1);
  let workerClosed = false;
  workers[0].on("close", () => { workerClosed = true; });

  await page.getByRole("tab", { name: "Explorer", exact: true }).click();
  await expect(page.getByRole("tabpanel", { name: "Explorer", exact: true })).toBeVisible();
  await expect.poll(() => workerClosed).toBe(true);

  await page.getByRole("tab", { name: "Variation" }).click();
  await expect(page.getByRole("status", { name: "Variation status" })).toHaveText("Ready.");
  await expect(page.getByRole("heading", { name: "Ready to Analyze Variation" })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
