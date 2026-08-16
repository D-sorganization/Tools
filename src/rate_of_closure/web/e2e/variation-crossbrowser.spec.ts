import { expect, test, type Locator, type Page, type Worker } from "@playwright/test";

import { capturePageErrors, openVariation, setNumericField } from "./variationTestSupport";

const SHOULDER = "swing_sim.swing.shoulder_commanded_torque_offset_nm";
const WRIST = "swing_sim.swing.wrist_commanded_torque_offset_nm";
const YAW = "swing_sim.swing.yaw_deg";

test.setTimeout(120_000);

async function configureLocalizedStudy(page: Page): Promise<void> {
  await page.getByRole("combobox", { name: "Pipeline" }).selectOption("swing");
  await page.getByRole("combobox", { name: "Variable 1" }).selectOption(SHOULDER);
  await page.getByRole("button", { name: "Add Variable" }).click();
  await page.getByRole("combobox", { name: "Variable 2" }).selectOption(WRIST);
  await page.getByRole("button", { name: "Add Variable" }).click();
  await page.getByRole("combobox", { name: "Variable 3" }).selectOption(YAW);
  for (const joint of ["Shoulder", "Wrist"]) {
    await setNumericField(page, `${joint} Commanded Torque Offset window start`, "0.02");
    await setNumericField(page, `${joint} Commanded Torque Offset window end`, "0.04");
  }
  await setNumericField(page, "Runs", "5");
  await setNumericField(page, "Seed", "4142");
  await page.getByRole("combobox", { name: "Analysis execution" })
    .selectOption("all_together");
}

async function cameraState(panel: Locator): Promise<{ yaw: number; pitch: number; zoom: number }> {
  const output = panel.getByLabel("Arc camera state");
  await expect(output).toBeVisible();
  return {
    yaw: Number(await output.getAttribute("data-yaw-deg")),
    pitch: Number(await output.getAttribute("data-pitch-deg")),
    zoom: Number(await output.getAttribute("data-zoom")),
  };
}

async function expectNoControlOverlap(panel: Locator): Promise<void> {
  const overlaps = await panel.locator("button, input, select").evaluateAll((controls) => {
    const visible = controls.map((control) => ({
      label: control.getAttribute("aria-label") ?? control.textContent?.trim() ?? control.tagName,
      rect: control.getBoundingClientRect(),
    })).filter(({ rect }) => rect.width > 0 && rect.height > 0);
    const conflicts: string[] = [];
    for (let left = 0; left < visible.length; left += 1) {
      for (let right = left + 1; right < visible.length; right += 1) {
        const a = visible[left]; const b = visible[right];
        const width = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left);
        const height = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top);
        if (width > 0.5 && height > 0.5) conflicts.push(`${a.label} <> ${b.label}`);
      }
    }
    return conflicts;
  });
  expect(overlaps).toEqual([]);
}

test("localized variation, confidence mesh, and keyboard camera work across engines", async (
  { page }, testInfo,
) => {
  const pageErrors = capturePageErrors(page);
  const workers: Worker[] = [];
  page.on("worker", (worker) => workers.push(worker));
  await openVariation(page);
  await configureLocalizedStudy(page);

  const run = page.getByRole("button", { name: "Run Variation Study" });
  await run.focus();
  await expect(run).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status", { name: "Variation status" }))
    .toContainText(/Done: \d+\/5 joint runs/);
  await expect.poll(() => workers.length).toBe(1);
  expect(workers[0].url()).toMatch(/variationExecution\.worker-[\w-]+\.js$/);
  await expect(page.getByRole("region", { name: "Localized torque result sources" }))
    .toContainText("joint.shoulder");
  await expect(page.getByRole("region", { name: "Localized torque result sources" }))
    .toContainText("[0.02, 0.04) s");

  const panel = page.getByRole("heading", {
    name: "All Swing Arcs — Interactive 3D Overlay",
  }).locator("..");
  const metric = panel.getByRole("combobox", { name: "Dispersion metric" });
  await metric.selectOption("confidence-ellipsoid-volume");
  const surfaces = panel.getByRole("checkbox", { name: "Show confidence ellipsoid surfaces" });
  await surfaces.focus();
  await page.keyboard.press("Space");
  await expect(surfaces).toBeChecked();
  await expect(panel.getByLabel("Arc visualization legend"))
    .toContainText("Gaussian position-content ellipsoid (not mean CI)");
  await expect(panel).toContainText(/Cyan surfaces show [1-9]\d* estimable/);

  const before = await cameraState(panel);
  const canvas = panel.getByRole("img", { name: /Interactive all-trial swing arcs/ });
  await canvas.focus();
  await expect(canvas).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("+");
  const moved = await cameraState(panel);
  expect(moved.yaw).not.toBe(before.yaw);
  expect(moved.zoom).toBeGreaterThan(before.zoom);
  await panel.getByRole("button", { name: "Reset View" }).click();
  const reset = await cameraState(panel);
  expect(reset).toEqual(before);
  await expect(surfaces).toBeChecked();

  await expectNoControlOverlap(panel);
  const screenshot = await panel.screenshot({ animations: "disabled", caret: "hide" });
  await testInfo.attach(`variation-crossbrowser-${testInfo.project.name}`, {
    body: screenshot, contentType: "image/png",
  });
  expect(screenshot.byteLength).toBeGreaterThan(10_000);
  expect(pageErrors).toEqual([]);
});
