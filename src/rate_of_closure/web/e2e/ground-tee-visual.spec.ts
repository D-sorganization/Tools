import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";

import { expect, test, type Page, type TestInfo } from "@playwright/test";

const MINIMUM_SCREENSHOT_BYTES = 20_000;

interface BrowserProblem {
  readonly kind: "console" | "page";
  readonly message: string;
}

async function capture(
  page: Page,
  testInfo: TestInfo,
  name: string,
): Promise<{ readonly file: string; readonly bytes: number; readonly sha256: string }> {
  const file = `${name}.png`;
  const path = testInfo.outputPath(file);
  await page.evaluate(() => {
    document.documentElement.style.scrollBehavior = "auto";
    document.body.style.scrollBehavior = "auto";
    window.scrollTo(0, 0);
  });
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
  await page.screenshot({
    path,
    animations: "disabled",
  });
  const payload = await readFile(path);
  const metadata = await stat(path);
  expect(metadata.size).toBeGreaterThan(MINIMUM_SCREENSHOT_BYTES);
  await testInfo.attach(name, { path, contentType: "image/png" });
  return {
    file,
    bytes: metadata.size,
    sha256: createHash("sha256").update(payload).digest("hex"),
  };
}

test("records deterministic Driver Ground and Tee visual evidence", async ({ page }, testInfo) => {
  const problems: BrowserProblem[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      problems.push({ kind: "console", message: message.text() });
    }
  });
  page.on("pageerror", (error) => {
    problems.push({ kind: "page", message: error.message });
  });
  await page.addInitScript(() => window.localStorage.clear());

  await page.goto("/");
  expect(page.viewportSize()).toEqual({ width: 1600, height: 1200 });
  await expect(page.getByRole("heading", { name: "Rate of Closure Impact Explorer" }))
    .toBeVisible();
  await page.getByRole("tab", { name: "Simulation", exact: true }).click();
  await page.getByRole("tab", { name: "Strike", exact: true }).click();

  const support = page.getByRole("radiogroup", { name: "Ball Support Mode" });
  const tee = support.getByRole("radio", { name: "Tee" });
  const ground = support.getByRole("radio", { name: "Ground" });
  const height = page.getByRole("textbox", { name: "Tee Height" });
  await expect(tee).toBeChecked();
  await expect(height).toBeEnabled();
  await expect(height).toHaveValue("38.1");
  await expect(page.getByRole("img", { name: "Tee ball support side elevation" }))
    .toBeVisible();
  await expect(page.getByLabel("Representative Tee")).toBeVisible();
  const teeCapture = await capture(page, testInfo, "driver-tee");

  await support.getByText("Ground", { exact: true }).click();
  await expect(ground).toBeChecked();
  await expect(height).toBeDisabled();
  await expect(height).toHaveValue("0");
  await page.getByRole("button", { name: "Run Simulation", exact: true }).click();
  await expect(page.getByRole("status", { name: "Simulation run status" }))
    .toContainText("Completed");
  await expect(page.getByRole("img", { name: "Ground ball support side elevation" }))
    .toBeVisible();
  await expect(page.getByLabel("Representative Tee")).toHaveCount(0);
  const groundCapture = await capture(page, testInfo, "driver-ground");

  expect(groundCapture.sha256).not.toBe(teeCapture.sha256);
  expect(problems).toEqual([]);
  const manifest = {
    schema: "rate-of-closure.ground-tee-visual-evidence/v1",
    browser: "chromium",
    viewport: { width: 1600, height: 1200 },
    states: {
      tee: { supportMode: "tee", teeHeightMm: 38.1, screenshot: teeCapture },
      ground: { supportMode: "ground", teeHeightMm: 0, screenshot: groundCapture },
    },
    browserProblems: problems,
  } as const;
  const manifestPath = testInfo.outputPath("manifest.json");
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await testInfo.attach("manifest", {
    path: manifestPath,
    contentType: "application/json",
  });
});
