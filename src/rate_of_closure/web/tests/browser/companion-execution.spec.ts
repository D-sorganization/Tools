import { expect, test, type Download, type Page, type Request } from "@playwright/test";
import { readFileSync } from "node:fs";

import { startCompanionHarness } from "./support/companionHarness";
import { auditSameOriginNetwork } from "./support/networkAudit";

const CAPABILITY_PATH = "/api/rate-of-closure/v1/capabilities";
const JOB_COLLECTION_PATH = "/api/rate-of-closure/v1/regional-ground/jobs";
const MAX_DOWNLOAD_BYTES = 5_000_000;
const REGIONAL_VARIATION_REQUEST = readFileSync(new URL(
  "../../src/model/__fixtures__/regional_ground_variation_request_golden_v1.json",
  import.meta.url,
));

async function openGroundExecution(page: Page, origin: string): Promise<void> {
  const capability = page.waitForResponse((response) =>
    new URL(response.url()).pathname === CAPABILITY_PATH && response.status() === 200);
  await page.goto(`${origin}/`);
  await capability;
  await page.getByRole("tab", { name: "Variation" }).click();
  await page.getByText("File", { exact: true }).click();
  await page.getByLabel("Open Regional-Ground Variation Request file").setInputFiles({
    name: "regional-ground-variation-request.json",
    mimeType: "application/json",
    buffer: REGIONAL_VARIATION_REQUEST,
  });
  await expect(page.getByText("Imported regional-ground-variation-request.json", {
    exact: false,
  })).toBeVisible();
  await page.getByText("File", { exact: true }).click();
  await page.getByRole("tab", { name: "Ground Playback" }).click();
  await expect(page.getByRole("heading", { name: "Regional-ground study execution" }))
    .toBeVisible();
}

async function prepareCurrentJob(page: Page): Promise<void> {
  const button = page.getByRole("button", { name: "Prepare Current Job" });
  await expect(button).toBeEnabled();
  await button.click();
  await expect(page.getByLabel("Imported execution job summary")).toBeVisible();
}

async function downloadText(download: Download): Promise<string> {
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of stream) {
    const buffer = Buffer.from(chunk);
    bytes += buffer.byteLength;
    if (bytes > MAX_DOWNLOAD_BYTES) throw new Error("browser download exceeded bound");
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString("utf8");
}

const isJobSubmit = (request: Request): boolean => {
  const target = new URL(request.url());
  return request.method() === "POST" && target.pathname === JOB_COLLECTION_PATH;
};

test("prepare-confirm-run-download-reload-import-recover is authority-only", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "Worker", {
      configurable: false,
      value: class ForbiddenBrowserPhysicsWorker {
        constructor() { throw new Error("browser physics worker is forbidden"); }
      },
    });
  });
  const companion = await startCompanionHarness("fast");
  const audit = auditSameOriginNetwork(page, companion.origin);
  const submissions: Request[] = [];
  page.on("request", (request) => { if (isJobSubmit(request)) submissions.push(request); });
  try {
    await openGroundExecution(page, companion.origin);
    await prepareCurrentJob(page);
    expect(submissions).toHaveLength(0);

    const jobDownload = page.waitForEvent("download");
    await page.getByRole("button", {
      name: "Download canonical regional-ground execution job",
    }).click();
    const job = await jobDownload;
    const jobText = await downloadText(job);
    expect(JSON.parse(jobText)).toMatchObject({
      schema_version: "rate-of-closure/regional-ground-execution-job/v1",
    });

    await page.getByRole("checkbox", { name: /I reviewed the accepted job identity/ })
      .check();
    await page.getByRole("button", { name: "Run imported study" }).click();
    await expect(page.getByLabel("Imported study execution status"))
      .toContainText("succeeded", { timeout: 15_000 });
    expect(submissions).toHaveLength(1);

    const resultDownload = page.waitForEvent("download");
    await page.getByRole("button", {
      name: "Download canonical regional-ground study result",
    }).click();
    const resultText = await downloadText(await resultDownload);
    expect(JSON.parse(resultText)).toMatchObject({
      schema_version: "rate-of-closure/regional-ground-execution-result/v1",
    });

    await page.reload();
    await page.getByRole("tab", { name: "Ground Playback" }).click();
    await page.getByTestId("regional-ground-execution-job-file-input")
      .setInputFiles({
        name: "regional-ground-execution-job.json",
        mimeType: "application/json",
        buffer: Buffer.from(jobText),
      });
    await expect(page.getByLabel("Imported execution job summary")).toBeVisible();
    expect(submissions).toHaveLength(1);
    await page.getByRole("button", { name: "Recover retained status" }).click();
    await expect(page.getByLabel("Imported study execution status"))
      .toContainText("succeeded", { timeout: 15_000 });
    expect(submissions).toHaveLength(1);

    const recoveredDownload = page.waitForEvent("download");
    await page.getByRole("button", {
      name: "Download canonical regional-ground study result",
    }).click();
    expect(await downloadText(await recoveredDownload)).toBe(resultText);
    audit.assertClean();
  } finally {
    await companion.close();
  }
});

test("cancellation publishes a terminal status without a partial result", async ({ page }) => {
  const companion = await startCompanionHarness("cancellable");
  const audit = auditSameOriginNetwork(page, companion.origin, {
    maxTransportFailures: 1,
  });
  try {
    await openGroundExecution(page, companion.origin);
    await prepareCurrentJob(page);
    await page.getByRole("checkbox", { name: /I reviewed the accepted job identity/ })
      .check();
    await page.getByRole("button", { name: "Run imported study" }).click();
    const cancel = page.getByRole("button", { name: "Cancel study" });
    await expect(cancel).toBeEnabled();
    await cancel.click();
    await expect(page.getByLabel("Imported study execution status"))
      .toContainText("cancelled", { timeout: 15_000 });
    await expect(page.getByRole("button", {
      name: "Download canonical regional-ground study result",
    })).toBeDisabled();
    audit.assertClean();
  } finally {
    await companion.close();
  }
});

test("changed editor state makes a prepared job stale without submitting", async ({ page }) => {
  const companion = await startCompanionHarness("fast");
  const audit = auditSameOriginNetwork(page, companion.origin);
  const submissions: Request[] = [];
  page.on("request", (request) => { if (isJobSubmit(request)) submissions.push(request); });
  try {
    await openGroundExecution(page, companion.origin);
    await prepareCurrentJob(page);
    await page.getByRole("tab", { name: "Flight Explorer" }).click();
    await page.getByRole("textbox", { name: "Launch Angle" }).fill("14.25");
    await page.getByRole("tab", { name: "Ground Playback" }).click();
    await expect(page.getByRole("alert")).toContainText("prepared job is stale");
    await expect(page.getByRole("checkbox", {
      name: /I reviewed the accepted job identity/,
    })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Run imported study" }))
      .toBeDisabled();
    expect(submissions).toHaveLength(0);
    audit.assertClean();
  } finally {
    await companion.close();
  }
});
