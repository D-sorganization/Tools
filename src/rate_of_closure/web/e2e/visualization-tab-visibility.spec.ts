/**
 * E2E tests verifying visibility and viewport bounds for all registered visualization tabs.
 */
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { expect, test, type Locator, type Page } from "@playwright/test";

import {
  visualizationReferenceEnvironments,
  visualizationTabs,
} from "../src/model/visualizationTabManifest";
import { capturePageErrors } from "./variationTestSupport";

interface VisualEvidence {
  tabId: string;
  classification: string;
  locator: string;
  minimumVisibleHeightPx: number;
  rect: { x: number; y: number; width: number; height: number };
  visibleIntersection: { width: number; height: number };
  horizontalOverflowPx: number;
}

const PAINT_SAMPLE_INTERVAL_MS = 100;
const REQUIRED_STABLE_PAINT_SAMPLES = 3;
const MAX_PAINT_SAMPLES = 20;
// This registered evidence pass visits every tab at three reference viewports
// and captures stable images for the 1440x900 authority.  The trusted runner
// needs more than the suite's 45-second interactive-test default even when all
// rendering contracts pass, so give this publication gate its own bounded
// budget without relaxing any visual assertion.
const VISUAL_EVIDENCE_TIMEOUT_MS = 180_000;

// #5454: the tab pills use `transition-all`, and a transition that is still
// interpolating a border or glow when the screenshot is taken leaves
// run-to-run anti-aliasing noise in the tab strip, so candidate hashes differed
// between runs on the same commit.  Snapping transitions and animations to
// their end state changes only timing, never the settled pixels, and no
// assertion or tolerance is involved.  Injected once per page.
const SETTLE_STYLE_ID = "visual-baseline-settle-style";

const captureStablePage = async (page: Page): Promise<Buffer> => {
  await page.evaluate(async (styleId) => {
    if (document.getElementById(styleId) === null) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent =
        "*, *::before, *::after { transition: none !important; animation: none !important; }";
      document.head.appendChild(style);
    }
    await document.fonts.ready;
    await new Promise<void>((resolvePaint) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolvePaint()));
    });
  }, SETTLE_STYLE_ID);
  let previous: Buffer | null = null;
  let stableSamples = 0;
  for (let sample = 0; sample < MAX_PAINT_SAMPLES; sample += 1) {
    await page.waitForTimeout(PAINT_SAMPLE_INTERVAL_MS);
    const image = await page.screenshot({ animations: "disabled", caret: "hide" });
    stableSamples = previous?.equals(image) ? stableSamples + 1 : 1;
    if (stableSamples >= REQUIRED_STABLE_PAINT_SAMPLES) return image;
    previous = image;
  }
  throw new Error(`page paint did not stabilize within ${MAX_PAINT_SAMPLES} samples`);
};

const intersection = async (locator: Locator): Promise<VisualEvidence["visibleIntersection"]> =>
  locator.evaluate((element) => {
    let rect = element.getBoundingClientRect();
    let ancestor = element.parentElement;
    while (ancestor !== null) {
      const style = getComputedStyle(ancestor);
      if ([style.overflow, style.overflowX, style.overflowY]
        .some((value) => value === "hidden" || value === "clip" || value === "scroll" || value === "auto")) {
        const clip = ancestor.getBoundingClientRect();
        const left = Math.max(rect.left, clip.left); const top = Math.max(rect.top, clip.top);
        const right = Math.min(rect.right, clip.right); const bottom = Math.min(rect.bottom, clip.bottom);
        rect = new DOMRect(left, top, Math.max(0, right - left), Math.max(0, bottom - top));
      }
      ancestor = ancestor.parentElement;
    }
    const left = Math.max(rect.left, 0); const top = Math.max(rect.top, 0);
    const right = Math.min(rect.right, innerWidth); const bottom = Math.min(rect.bottom, innerHeight);
    return { width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
  });

type ManifestTab = ReturnType<typeof visualizationTabs>[number];

/** Manifest-owned minimum visible landmark size for one reference viewport. */
const requiredVisibleSize = (
  entry: ManifestTab, viewport: { width: number; height: number },
): { width: number; height: number } => {
  const reference = visualizationReferenceEnvironments.react;
  const desktop = viewport.width >= 1280;
  return {
    width: entry.landmarkKind === "semantic-content" ? 1
      : desktop ? reference.minimumVisibleWidthPx : reference.responsiveMinimumVisibleWidthPx,
    height: desktop ? entry.minimumVisibleHeightPx : reference.responsiveMinimumVisibleHeightPx,
  };
};

const auditTab = async (page: Page, tabId: string, locatorText: string,
  classification: string, minimumVisibleHeightPx: number): Promise<VisualEvidence> => {
  const tab = page.locator(`#primary-tab-${tabId}`);
  await tab.scrollIntoViewIfNeeded();
  await tab.click();
  // Tab activation can preserve an incidental scroll anchor from the prior
  // landmark.  Normalize before auditing so both the geometry evidence and any
  // later initial-page capture describe the canonical top-of-page viewport.
  await page.evaluate(() => window.scrollTo(0, 0));
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
  const locator = page.locator(locatorText);
  await expect(locator).toHaveCount(1);
  await expect(locator).toBeVisible();
  const rect = await locator.boundingBox();
  if (rect === null) throw new Error(`${tabId} primary visual has no rectangle`);
  const visibleIntersection = await intersection(locator);
  const horizontalOverflowPx = await page.evaluate(() => Math.max(
    document.body.scrollWidth - document.body.clientWidth,
    document.documentElement.scrollWidth - document.documentElement.clientWidth,
  ));
  return {
    tabId, classification, locator: locatorText, minimumVisibleHeightPx, rect,
    visibleIntersection, horizontalOverflowPx,
  };
};

test("every registered React tab exposes its primary visual in the initial viewport", async (
  { page }, testInfo,
) => {
  test.setTimeout(VISUAL_EVIDENCE_TIMEOUT_MS);
  test.skip(testInfo.project.name !== "chromium-desktop", "manifest viewport authority");
  const pageErrors = capturePageErrors(page);
  const evidence: Array<{ viewport: { width: number; height: number }; tabs: VisualEvidence[] }> = [];
  const candidates: Array<{ tabId: string; file: string; sha256: string }> = [];
  const candidateRoot = resolve(
    process.env.RATE_VISUAL_BASELINE_CANDIDATE_DIR ??
      testInfo.outputPath("visual-baseline-candidates"),
  );
  const reactCandidateRoot = resolve(candidateRoot, "react");
  await mkdir(reactCandidateRoot, { recursive: true });
  const reference = visualizationReferenceEnvironments.react;
  const viewports = [reference.viewportPx, ...reference.additionalViewportsPx]
    .map(([width, height]) => ({ width, height }));
  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
    await page.goto("/");
    const tabs: VisualEvidence[] = [];
    for (const entry of visualizationTabs("react")) {
      const audited = await auditTab(
        page, entry.tabId, entry.primaryVisualLocator, entry.classification,
        entry.minimumVisibleHeightPx,
      );
      const label = `${entry.tabId} at ${viewport.width}x${viewport.height}`;
      expect.soft(audited.rect.width, `${label} width`).toBeGreaterThan(0);
      expect.soft(audited.rect.height, `${label} height`).toBeGreaterThan(0);
      const required = requiredVisibleSize(entry, viewport);
      expect.soft(audited.visibleIntersection.width, `${label} visible width`)
        .toBeGreaterThanOrEqual(required.width);
      expect.soft(audited.visibleIntersection.height, `${label} visible height`)
        .toBeGreaterThanOrEqual(required.height);
      expect.soft(audited.horizontalOverflowPx, `${label} document overflow`).toBe(0);
      if (entry.tabId === "variation") {
        // Initial-state evidence must not imply computed Morris results.  The
        // qualified target/source controls are introduced only with a parsed,
        // completed report and are exercised in MorrisResults.test.tsx.
        await expect(page.getByRole("region", { name: "Morris screening results" }))
          .toHaveCount(0);
      }
      if (viewport.width < 1280 && entry.landmarkKind === "visual") {
        const controlSelector = reference.responsiveControlLocators[entry.tabId];
        if (controlSelector !== undefined) {
          const longControls = page.locator(controlSelector);
          await expect(longControls).toHaveCount(1);
          await expect(longControls).toBeVisible();
          const controlsRect = await longControls.boundingBox();
          if (controlsRect === null) throw new Error(`${label} controls have no rectangle`);
          expect.soft(audited.rect.y, `${label} visual-first order`)
            .toBeLessThanOrEqual(controlsRect.y);
        }
      }
      if (viewport.width < 1280 && entry.tabId === "putting") {
        // RM #1507 (2026-09-02): the shared playback transport overflowed the
        // 390x844 document by 6 px on Linux Chromium because the speed
        // <select> laid out wider than its flex hypothetical size and pushed
        // the position readout past the viewport.  Pin the readout's right
        // edge inside the narrow viewport so the regression names its element.
        const readout = page.locator("output[aria-label='Putt playback position']");
        await expect(readout).toBeVisible();
        const readoutRect = await readout.boundingBox();
        if (readoutRect === null) throw new Error(`${label} playback readout has no rectangle`);
        expect.soft(readoutRect.x + readoutRect.width, `${label} playback readout right edge`)
          .toBeLessThanOrEqual(viewport.width);
        // RM #1507 (2026-09-03): ADR-0045 F2's green-import row was a second
        // 6 px overflow source at 390x844 — the file input's font-stack
        // dependent intrinsic width propagated through the flex default
        // min-width:auto, so the row could not shrink on Linux.  Pin the
        // row's rightmost control inside the narrow viewport so a recurrence
        // names its element.
        const planarReset = page.getByRole("button", { name: "Use Planar Green" });
        await expect(planarReset).toBeVisible();
        const planarRect = await planarReset.boundingBox();
        if (planarRect === null) throw new Error(`${label} planar-green reset has no rectangle`);
        expect.soft(planarRect.x + planarRect.width, `${label} green-import row right edge`)
          .toBeLessThanOrEqual(viewport.width);
      }
      if (viewport.width === 1440 && viewport.height === 900) {
        if (entry.tabId === "explorer") {
          await expect(page.getByRole("button", { name: "Play" })).toBeVisible();
        }
        if (entry.tabId === "putting") {
          // #4800 P7: the delivered-stroke parameters are part of this
          // tab's registered surface, so the authority viewport must show
          // them beside the primary visual rather than behind disclosure.
          for (const control of [
            "Aim °", "Face angle °", "Putter path °", "Strike toward toe mm",
          ]) {
            await expect(page.getByRole("textbox", { name: control })).toBeVisible();
          }
          // #4800 P8: playback rides the shared transport, so the same
          // viewport must reach it — the Putt wording and Strike/Finish
          // jumps that bind the subject-neutral bar.
          for (const control of ["Play Putt", "Jump to Strike", "Jump to Finish"]) {
            await expect(page.getByRole("button", { name: control })).toBeVisible();
          }
          await expect(page.getByRole("slider", { name: "Putt Time" })).toBeVisible();
        }
        if (entry.tabId === "launch-monitor-analytics") {
          // ADR-0048 G1-D3: source-backed strokes gained reports its excluded
          // rows (status plus per-reason counts) beside the result rather than
          // dropping them in silence. That line is deliberately NOT asserted
          // here: it only renders once a *licensed* expected-strokes baseline
          // artifact has been loaded and every course-state column mapped, and
          // this repository bundles no baseline table by design (see
          // docs/rate_of_closure/SOURCE_BACKED_STROKES_GAINED.md, "Availability
          // Boundary"). Its coverage lives in the runtime-parity suites —
          // launchMonitorSourceBackedStrokesGained.test.ts and
          // tests/rate_of_closure/test_launch_monitor_strokes_gained.py — which
          // assert the same nine malformed-row cases in both runtimes. What the
          // authority viewport does own is the panel itself staying visible and
          // labelled as the local compatibility path.
          await expect(
            page.getByRole("heading", { name: "Source-Backed Strokes Gained" }),
          ).toBeVisible();
        }
        const file = `initial-${entry.tabId}-1440x900.png`;
        const image = await captureStablePage(page);
        await writeFile(resolve(reactCandidateRoot, file), image);
        candidates.push({
          tabId: entry.tabId,
          file,
          sha256: createHash("sha256").update(image).digest("hex"),
        });
      }
      tabs.push(audited);
    }
    evidence.push({ viewport, tabs });
  }
  await testInfo.attach("visualization-tab-visibility-react-v1", {
    body: Buffer.from(JSON.stringify({
      artifactPolicy: "diagnostic-only-not-approved-golden", evidence,
    }, null, 2)),
    contentType: "application/json",
  });
  await writeFile(resolve(reactCandidateRoot, "manifest.json"), `${JSON.stringify({
    schemaId: "rate-of-closure/visual-baseline-candidates",
    schemaVersion: 1,
    artifactPolicy: "candidate-diagnostic-not-approved-until-protected-merge",
    sourceCommit: process.env.RATE_VISUAL_BASELINE_SOURCE_COMMIT ??
      process.env.GITHUB_SHA ?? "local-diagnostic",
    surface: "react",
    environment:
      `${process.platform}-chromium-desktop-1440x900-dark-reduced-motion-inter-5.3.0`,
    captures: candidates,
  }, null, 2)}\n`);
  expect(candidates).toHaveLength(visualizationTabs("react").length);
  expect(pageErrors).toEqual([]);
});

test("1440x900 visual-baseline candidates are deterministic across fresh sessions", async (
  { browser }, testInfo,
) => {
  // #5454: capture every registered React tab in two independent browser
  // contexts on the same build and require byte-identical images, so a real
  // visual change is distinguishable from run-to-run noise.
  test.setTimeout(VISUAL_EVIDENCE_TIMEOUT_MS);
  test.skip(testInfo.project.name !== "chromium-desktop", "manifest viewport authority");
  const captureAll = async (): Promise<Record<string, string>> => {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      colorScheme: "dark",
      reducedMotion: "reduce",
    });
    try {
      const page = await context.newPage();
      await page.goto("/");
      const hashes: Record<string, string> = {};
      for (const entry of visualizationTabs("react")) {
        const tab = page.locator(`#primary-tab-${entry.tabId}`);
        await tab.scrollIntoViewIfNeeded();
        await tab.click();
        await page.evaluate(() => window.scrollTo(0, 0));
        hashes[entry.tabId] = createHash("sha256")
          .update(await captureStablePage(page)).digest("hex");
      }
      return hashes;
    } finally {
      await context.close();
    }
  };
  const first = await captureAll();
  const second = await captureAll();
  expect(second).toEqual(first);
});

const DEMO_SOURCE = "Source: Built-In Demonstration Data";
const LAUNCH_MONITOR_IMPORT = {
  name: "imported-launch-monitor.csv",
  mimeType: "text/csv",
  buffer: Buffer.from([
    "club_speed,attack_angle,ball_speed,monitor_vendor",
    ...Array.from({ length: 40 }, (_, index) =>
      `${40 + 0.2 * index},${-3 + 0.1 * (index % 11)},${58 + 0.3 * index},` +
      `${index % 2 ? "TrackMan" : "Foresight"}`),
  ].join("\n")),
};
// One numeric column cannot form a relationship, so the import fails closed.
const MALFORMED_LAUNCH_MONITOR_IMPORT = {
  name: "malformed-launch-monitor.csv",
  mimeType: "text/csv",
  buffer: Buffer.from("ball_speed\n60\n61\n62\n"),
};

test("launch monitor scatter stays in the first viewport through every registered state", async (
  { page }, testInfo,
) => {
  // #4433: the initial-state audit above cannot see a visual that a later
  // state pushes below the fold.  Drive result, error, loading, and the
  // demonstration (empty) preview through the production handlers and
  // re-measure the registered landmark from the top of the page each time.
  test.setTimeout(VISUAL_EVIDENCE_TIMEOUT_MS);
  test.skip(testInfo.project.name !== "chromium-desktop", "manifest viewport authority");
  const entry = visualizationTabs("react")
    .find((candidate) => candidate.tabId === "launch-monitor-analytics");
  if (entry === undefined) throw new Error("launch-monitor-analytics is not registered");
  expect(Object.keys(entry.states).sort()).toEqual(["empty", "error", "loading", "result"]);
  const pageErrors = capturePageErrors(page);
  // Hold browser file reads open on request so the pending import is observable.
  await page.addInitScript(() => {
    const gate = window as unknown as { holdReads?: boolean; releaseRead?: () => void };
    const read = Blob.prototype.arrayBuffer;
    Blob.prototype.arrayBuffer = function heldArrayBuffer(this: Blob) {
      if (gate.holdReads !== true) return read.call(this);
      return new Promise<ArrayBuffer>((resolveRead) => {
        gate.releaseRead = () => resolveRead(read.call(this));
      });
    };
  });
  const reference = visualizationReferenceEnvironments.react;
  const viewports = [reference.viewportPx, ...reference.additionalViewportsPx]
    .map(([width, height]) => ({ width, height }));
  const visual = page.locator(entry.primaryVisualLocator);
  const fileInput = page.getByLabel("Launch monitor CSV or JSON file");
  const source = page.getByText(/^Source: /);
  const correlations = page.getByRole("heading", { name: "Correlations and Multiplicity Control" });
  for (const viewport of viewports) {
    const assertFirstViewport = async (state: string): Promise<void> => {
      const label = `launch-monitor-analytics ${state} at ${viewport.width}x${viewport.height}`;
      await page.evaluate(() => window.scrollTo(0, 0));
      expect(await page.evaluate(() => window.scrollY), `${label} scroll`).toBe(0);
      await expect(visual, label).toHaveCount(1);
      await expect(visual, label).toBeVisible();
      const visible = await intersection(visual);
      const required = requiredVisibleSize(entry, viewport);
      expect.soft(visible.width, `${label} visible width`)
        .toBeGreaterThanOrEqual(required.width);
      expect.soft(visible.height, `${label} visible height`)
        .toBeGreaterThanOrEqual(required.height);
      const overflow = await page.evaluate(() => {
        const scroller = document.scrollingElement ?? document.documentElement;
        return scroller.scrollWidth - scroller.clientWidth;
      });
      expect.soft(overflow, `${label} document overflow`).toBeLessThanOrEqual(0);
    };
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.locator(`#primary-tab-${entry.tabId}`).click();

    await page.getByRole("button", { name: "Run Analysis" }).click();
    await expect(correlations).toBeVisible();
    await assertFirstViewport("result");

    await fileInput.setInputFiles(MALFORMED_LAUNCH_MONITOR_IMPORT);
    await expect(page.getByRole("alert")).toBeVisible();
    await expect(source).toContainText(DEMO_SOURCE);
    await expect(correlations, "prior result retained beside the error").toBeVisible();
    await assertFirstViewport("error");

    await page.evaluate(() => { (window as unknown as { holdReads: boolean }).holdReads = true; });
    await fileInput.setInputFiles(LAUNCH_MONITOR_IMPORT);
    await page.waitForFunction(() => typeof (window as unknown as {
      releaseRead?: () => void;
    }).releaseRead === "function");
    await expect(source, "import is still pending").toContainText(DEMO_SOURCE);
    await assertFirstViewport("loading");
    await page.evaluate(() => {
      const gate = window as unknown as { holdReads: boolean; releaseRead?: () => void };
      gate.holdReads = false;
      gate.releaseRead?.();
    });
    await expect(source).toContainText(LAUNCH_MONITOR_IMPORT.name);

    await page.getByRole("button", { name: "Load Demo" }).click();
    await expect(source).toContainText(DEMO_SOURCE);
    await expect(correlations).toHaveCount(0);
    await assertFirstViewport("empty");
  }
  expect(pageErrors).toEqual([]);
});

test("visible intersection clips a landmark through an overflow ancestor", async ({ page }) => {
  await page.setContent(`<div style="height:100px;overflow:hidden">
    <div style="height:180px"></div><div data-landmark style="height:240px"></div></div>`);
  expect((await intersection(page.locator("[data-landmark]"))).height).toBe(0);
  await page.setContent(`<div style="width:1px;overflow:hidden">
    <div data-landmark style="width:240px;height:240px"></div></div>`);
  expect((await intersection(page.locator("[data-landmark]"))).width).toBe(1);
  await page.setContent(`<div style="height:1px;overflow:hidden">
    <div data-landmark style="width:240px;height:240px"></div></div>`);
  const height = (await intersection(page.locator("[data-landmark]"))).height;
  expect(height).toBe(1);
  expect(height).toBeLessThan(
    visualizationReferenceEnvironments.react.responsiveMinimumVisibleHeightPx,
  );
});

test("candidate capture waits through a scheduled browser paint", async ({ page }) => {
  await page.setContent('<main data-paint-state="pending">Pending paint</main>');
  await page.evaluate(() => {
    window.setTimeout(() => {
      const landmark = document.querySelector("[data-paint-state]");
      if (!(landmark instanceof HTMLElement)) return;
      landmark.dataset.paintState = "complete";
      landmark.style.backgroundColor = "rgb(0, 128, 0)";
    }, 150);
  });

  const image = await captureStablePage(page);

  expect(image.byteLength).toBeGreaterThan(0);
  await expect(page.locator('[data-paint-state="complete"]')).toHaveCSS(
    "background-color",
    "rgb(0, 128, 0)",
  );
});
