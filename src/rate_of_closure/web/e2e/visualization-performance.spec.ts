import { expect, test, type Locator, type Page } from "@playwright/test";

import {
  visualizationPerformanceBudgets,
  visualizationPerformanceTabs,
} from "../src/model/visualizationPerformanceManifest";
import {
  visualizationReferenceEnvironments,
  visualizationTabs,
} from "../src/model/visualizationTabManifest";
import { capturePageErrors } from "./variationTestSupport";

type Rect = { x: number; y: number; width: number; height: number };
interface StableRectEvidence {
  settleMs: number;
  first: Rect;
  final: Rect;
  maxStepPx: number;
}

interface TabPerformanceEvidence {
  tabId: string;
  workload: string;
  openMs: number;
  resizeSettleMs: number;
  maxOpenStepPx: number;
  maxResizeStepPx: number;
  postSettleShiftPx: number;
  layoutShiftScoreMicrounits: number;
}

const rectShift = (left: Rect, right: Rect): number => Math.max(
  Math.abs(left.x - right.x), Math.abs(left.y - right.y),
  Math.abs(left.width - right.width), Math.abs(left.height - right.height),
);

const stableRect = async (
  locator: Locator, timeoutMs: number, stableFrames: number, tolerancePx: number,
): Promise<StableRectEvidence> => locator.evaluate(async (element, options) => {
  const read = (): Rect => {
    const rect = element.getBoundingClientRect();
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  };
  const shift = (left: Rect, right: Rect): number => Math.max(
    Math.abs(left.x - right.x), Math.abs(left.y - right.y),
    Math.abs(left.width - right.width), Math.abs(left.height - right.height),
  );
  const started = performance.now();
  const first = read();
  let previous = first;
  let stable = 0;
  let maxStepPx = 0;
  while (performance.now() - started <= options.timeoutMs) {
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    const current = read();
    const step = shift(previous, current);
    maxStepPx = Math.max(maxStepPx, step);
    stable = step <= options.tolerancePx ? stable + 1 : 0;
    previous = current;
    if (stable >= options.stableFrames) {
      return { settleMs: performance.now() - started, first, final: current, maxStepPx };
    }
  }
  throw new Error(`visual rectangle did not settle within ${options.timeoutMs} ms`);
}, { timeoutMs, stableFrames, tolerancePx });

const resetLayoutShift = async (page: Page): Promise<void> => {
  await page.evaluate(() => {
    (window as Window & { __rateLayoutShiftValues: number[] }).__rateLayoutShiftValues = [];
  });
};

const layoutShiftMicrounits = async (page: Page): Promise<number> => page.evaluate(() =>
  Math.round((window as Window & { __rateLayoutShiftValues: number[] })
    .__rateLayoutShiftValues.reduce((total, value) => total + value, 0) * 1_000_000));

test("all React tabs satisfy protected open, resize, and layout-stability budgets", async (
  { page }, testInfo,
) => {
  test.skip(testInfo.project.name !== "chromium-desktop", "single Chromium budget authority");
  await page.addInitScript(() => {
    const target = window as Window & { __rateLayoutShiftValues: number[] };
    target.__rateLayoutShiftValues = [];
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const shift = entry as PerformanceEntry & {
          hadRecentInput?: boolean; value?: number;
        };
        const value = shift.value;
        if (shift.hadRecentInput !== true && typeof value === "number" && Number.isFinite(value)) {
          target.__rateLayoutShiftValues.push(value);
        }
      }
    }).observe({ type: "layout-shift", buffered: true });
  });
  const errors = capturePageErrors(page);
  const budget = visualizationPerformanceBudgets.react;
  const expectedCls = budget.maxLayoutShiftScoreMicrounits;
  if (expectedCls === null) throw new Error("React CLS budget is required");
  const performanceTabs = visualizationPerformanceTabs("react");
  const visibility = new Map(visualizationTabs("react").map((entry) => [entry.tabId, entry]));
  const references = visualizationReferenceEnvironments.react;
  const viewports = [references.viewportPx, ...references.additionalViewportsPx]
    .map(([width, height]) => ({ width, height }));
  const evidence: Array<{ viewport: { width: number; height: number };
    tabs: TabPerformanceEvidence[] }> = [];

  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await page.goto("/");
    const tabs: TabPerformanceEvidence[] = [];
    for (const entry of performanceTabs) {
      const visualEntry = visibility.get(entry.tabId);
      if (visualEntry === undefined) throw new Error(`missing visual entry ${entry.tabId}`);
      await resetLayoutShift(page);
      const started = Date.now();
      await page.locator(`#primary-tab-${entry.tabId}`).click();
      const visual = page.locator(visualEntry.primaryVisualLocator);
      await expect(visual).toHaveCount(1);
      await expect(visual).toBeVisible();
      const opened = await stableRect(
        visual, budget.tabOpenBudgetMs, budget.stableFrameCount,
        budget.stabilityTolerancePx,
      );
      const openMs = Date.now() - started;
      await page.waitForTimeout(100);
      const cls = await layoutShiftMicrounits(page);
      const beforeQuiet = await visual.boundingBox();
      if (beforeQuiet === null) throw new Error(`${entry.tabId} has no settled rectangle`);
      await page.waitForTimeout(100);
      const afterQuiet = await visual.boundingBox();
      if (afterQuiet === null) throw new Error(`${entry.tabId} lost its rectangle`);
      const narrow = { width: viewport.width - 8, height: viewport.height };
      const resizeStarted = Date.now();
      await page.setViewportSize(narrow);
      const shrunk = await stableRect(
        visual, budget.resizeSettleBudgetMs, budget.stableFrameCount,
        budget.stabilityTolerancePx,
      );
      await page.setViewportSize(viewport);
      const restored = await stableRect(
        visual, budget.resizeSettleBudgetMs, budget.stableFrameCount,
        budget.stabilityTolerancePx,
      );
      const resizeSettleMs = Date.now() - resizeStarted;
      const label = `${entry.tabId} at ${viewport.width}x${viewport.height}`;
      expect.soft(openMs, `${label} open latency`).toBeLessThanOrEqual(budget.tabOpenBudgetMs);
      expect.soft(resizeSettleMs, `${label} resize latency`)
        .toBeLessThanOrEqual(budget.resizeSettleBudgetMs);
      expect.soft(cls, `${label} cumulative layout shift`).toBeLessThanOrEqual(expectedCls);
      const postSettleShiftPx = rectShift(beforeQuiet, afterQuiet);
      expect.soft(postSettleShiftPx, `${label} post-settle movement`)
        .toBeLessThanOrEqual(budget.maxPostSettleShiftPx);
      tabs.push({
        tabId: entry.tabId, workload: entry.workload, openMs, resizeSettleMs,
        maxOpenStepPx: opened.maxStepPx,
        maxResizeStepPx: Math.max(shrunk.maxStepPx, restored.maxStepPx),
        postSettleShiftPx, layoutShiftScoreMicrounits: cls,
      });
    }
    evidence.push({ viewport, tabs });
  }
  await testInfo.attach("visualization-performance-react-v1", {
    body: Buffer.from(JSON.stringify({
      measurementPolicy: "protected-diagnostic-not-user-hardware-qualification",
      budget, evidence,
    }, null, 2)),
    contentType: "application/json",
  });
  expect(errors).toEqual([]);
});
