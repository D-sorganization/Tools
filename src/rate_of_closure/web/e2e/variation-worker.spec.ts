import { expect, test, type Page, type Worker } from "@playwright/test";
import { readFile } from "node:fs/promises";

import {
  capturePageErrors,
  openVariation,
  setNumericField,
} from "./variationTestSupport";

const LONG_RUN_COUNT = "500";
const PROGRESS_RUN_COUNT = "24";
const SHOULDER_TORQUE = "swing_sim.swing.shoulder_commanded_torque_offset_nm";
const WRIST_TORQUE = "swing_sim.swing.wrist_commanded_torque_offset_nm";

interface ProgressObservationWindow extends Window {
  variationProgressValues?: number[];
}

async function observeProgressValues(page: Page): Promise<void> {
  await page.evaluate(() => {
    const observed: number[] = [];
    const record = () => {
      const progress = document.querySelector<HTMLProgressElement>(
        'progress[aria-label="Variation execution progress"]',
      );
      if (progress !== null) observed.push(progress.value);
    };
    new MutationObserver(record).observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
    });
    (window as ProgressObservationWindow).variationProgressValues = observed;
  });
}

async function progressValues(page: Page): Promise<number[]> {
  return page.evaluate(
    () => (window as ProgressObservationWindow).variationProgressValues ?? [],
  );
}

async function configureSeededDelivery(
  page: Page,
  runs = "3",
): Promise<void> {
  await page.getByRole("combobox", { name: "Pipeline" }).selectOption("delivery");
  await setNumericField(page, "Runs", runs);
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

async function configureLocalizedSwing(page: Page, runs: string): Promise<void> {
  await page.getByRole("combobox", { name: "Pipeline" }).selectOption("swing");
  await page.getByRole("combobox", { name: "Variable 1" }).selectOption(SHOULDER_TORQUE);
  await page.getByRole("button", { name: "Add Variable" }).click();
  await page.getByRole("combobox", { name: "Variable 2" }).selectOption(WRIST_TORQUE);
  await setNumericField(page, "Shoulder Commanded Torque Offset window start", "0.02");
  await setNumericField(page, "Shoulder Commanded Torque Offset window end", "0.04");
  await setNumericField(page, "Wrist Commanded Torque Offset window start", "0.02");
  await setNumericField(page, "Wrist Commanded Torque Offset window end", "0.04");
  await setNumericField(page, "Runs", runs);
  await setNumericField(page, "Seed", "4142");
  await page.getByRole("combobox", { name: "Analysis execution" }).selectOption("both");
}

test("production Worker completes a seeded study and reruns deterministically", async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  const workers = captureWorkers(page);
  await openVariation(page);
  await configureSeededDelivery(page, PROGRESS_RUN_COUNT);
  await observeProgressValues(page);

  const runButton = page.getByRole("button", { name: "Run Variation Study" });
  const status = page.getByRole("status", { name: "Variation status" });
  await runButton.click();
  await expect(status).toContainText(/Done: \d+\/24 joint runs/);
  await expect(page.getByRole("progressbar", { name: "Variation execution progress" }))
    .toHaveAttribute("value", PROGRESS_RUN_COUNT);
  await expect(page.getByRole("heading", { name: "Summary — Dispersion per Output" }))
    .toBeVisible();
  await expect.poll(() => workers.length).toBe(1);
  expect(workers[0].url()).toMatch(/variationExecution\.worker-[\w-]+\.js$/);
  expect((await progressValues(page)).some(
    (value) => value > 0 && value < Number(PROGRESS_RUN_COUNT),
  )).toBe(true);
  const firstSummary = await page.getByRole("table").first().innerText();

  await runButton.click();
  await expect(status).toContainText(/Done: \d+\/24 joint runs/);
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
  let workerClosed = false;
  workers[0].on("close", () => { workerClosed = true; });
  const cancelButton = page.getByRole("button", { name: "Cancel Variation Study" });
  await expect(cancelButton).toBeEnabled();
  await cancelButton.click();

  const status = page.getByRole("status", { name: "Variation status" });
  await expect(status).toHaveText("Cancelled: no partial variation result was accepted.");
  await expect.poll(() => workerClosed).toBe(true);
  await expect(page.getByRole("heading", { name: "Ready to Analyze Variation" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Summary — Dispersion per Output" }))
    .toHaveCount(0);

  await configureSeededDelivery(page);
  const runButton = page.getByRole("button", { name: "Run Variation Study" });
  await runButton.click();
  await expect(status).toContainText(/Done: \d+\/3 joint runs/);
  await expect.poll(() => workers.length).toBe(2);
  const firstSummary = await page.getByRole("table").first().innerText();
  await runButton.click();
  await expect(status).toContainText(/Done: \d+\/3 joint runs/);
  await expect.poll(() => workers.length).toBe(3);
  expect(await page.getByRole("table").first().innerText()).toBe(firstSummary);
  expect(workers.slice(1).every((worker) => (
    /variationExecution\.worker-[\w-]+\.js$/.test(worker.url())
  ))).toBe(true);
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

test("localized production Worker cancels, reruns, and exports command provenance", async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  const workers = captureWorkers(page);
  await openVariation(page);
  await configureLocalizedSwing(page, LONG_RUN_COUNT);

  const runButton = page.getByRole("button", { name: "Run Variation Study" });
  await runButton.click();
  await expect.poll(() => workers.length).toBe(1);
  let workerClosed = false;
  workers[0].on("close", () => { workerClosed = true; });
  await page.getByRole("button", { name: "Cancel Variation Study" }).click();
  await expect.poll(() => workerClosed).toBe(true);
  await expect(page.getByRole("status", { name: "Variation status" }))
    .toHaveText("Cancelled: no partial variation result was accepted.");

  await setNumericField(page, "Runs", "3");
  await page.getByRole("combobox", { name: "Analysis execution" })
    .selectOption("all_together");
  await runButton.click();
  await expect(page.getByRole("status", { name: "Variation status" }))
    .toContainText(/Done: \d+\/3 joint runs/);
  await expect.poll(() => workers.length).toBe(2);
  await expect(page.getByRole("region", { name: "Localized torque result sources" }))
    .toBeVisible();
  const labels = await page.getByRole("list", { name: "Localized torque source filters" })
    .innerText();
  expect(labels).toContain("joint.shoulder");
  expect(labels).toContain("joint.wrist");
  expect(labels).toContain("[0.02, 0.04) s");
  expect(labels).toContain("N*m");
  expect(labels).toContain("variation_plan.v2:additive_commanded_torque");
  const sourceOptions = await page.getByRole("combobox", {
    name: "Arc perturbation source",
  }).innerText();
  expect(sourceOptions).toContain(SHOULDER_TORQUE);
  expect(sourceOptions).toContain("joint.shoulder");
  expect(sourceOptions).toContain("N*m");
  expect(sourceOptions).toContain("variation_plan.v2:additive_commanded_torque");

  const csvDownloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Localized Torque CSV" }).click();
  const csvPath = await (await csvDownloadPromise).path();
  if (csvPath === null) throw new Error("localized torque CSV download has no path");
  const csv = await readFile(csvPath, "utf8");
  expect(csv).toContain("spec_id,variable_key,joint_id");
  expect(csv).toContain("variation_plan.v2:additive_commanded_torque");

  const jsonDownloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Swing Ensemble JSON" }).click();
  const jsonPath = await (await jsonDownloadPromise).path();
  if (jsonPath === null) throw new Error("swing ensemble JSON download has no path");
  const document = JSON.parse(await readFile(jsonPath, "utf8"));
  expect(document.schemaVersion).toBe(2);
  expect(document.trials[0].localizedTorqueCommands).toHaveLength(2);

  await runButton.click();
  await expect(page.getByRole("status", { name: "Variation status" }))
    .toContainText(/Done: \d+\/3 joint runs/);
  await expect.poll(() => workers.length).toBe(3);
  expect(await page.getByRole("list", { name: "Localized torque source filters" })
    .innerText()).toBe(labels);
  expect(workers.every((worker) => /variationExecution\.worker-[\w-]+\.js$/.test(worker.url())))
    .toBe(true);
  expect(pageErrors).toEqual([]);
});
