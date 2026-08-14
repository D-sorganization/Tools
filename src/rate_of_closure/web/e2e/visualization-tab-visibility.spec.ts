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

const auditTab = async (page: Page, tabId: string, locatorText: string,
  classification: string, minimumVisibleHeightPx: number): Promise<VisualEvidence> => {
  const tab = page.locator(`#primary-tab-${tabId}`);
  await tab.scrollIntoViewIfNeeded();
  await tab.click();
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
      const requiredWidth = entry.landmarkKind === "semantic-content" ? 1
        : viewport.width >= 1280 ? reference.minimumVisibleWidthPx
          : reference.responsiveMinimumVisibleWidthPx;
      expect.soft(audited.visibleIntersection.width, `${label} visible width`)
        .toBeGreaterThanOrEqual(requiredWidth);
      const requiredHeight = viewport.width >= 1280
        ? entry.minimumVisibleHeightPx
        : reference.responsiveMinimumVisibleHeightPx;
      expect.soft(audited.visibleIntersection.height, `${label} visible height`)
        .toBeGreaterThanOrEqual(requiredHeight);
      expect.soft(audited.horizontalOverflowPx, `${label} document overflow`).toBe(0);
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
      if (viewport.width === 1440 && viewport.height === 900) {
        const file = `initial-${entry.tabId}-1440x900.png`;
        const image = await page.screenshot({ animations: "disabled", caret: "hide" });
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
    environment: `${process.platform}-chromium-desktop-1440x900-dark-reduced-motion`,
    captures: candidates,
  }, null, 2)}\n`);
  expect(candidates).toHaveLength(visualizationTabs("react").length);
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
