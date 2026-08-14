/**
 * Hover-hint completeness sweep (#4120 V4): every interactive element
 * (buttons, selects, inputs, checkboxes, canvases) of every panel must
 * carry a hover hint — a `title` — or an `aria-label` (canvases may
 * use their aria-label as the accessible hint).
 */

import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import { DEFAULT_SCENARIO } from "../model/impact";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";
import { Derivation } from "./Derivation";
import { FlightExplorerPanel } from "./FlightExplorerPanel";
import { GlossaryPanel } from "./GlossaryPanel";
import { LaunchMonitorAnalyticsPanel } from "./LaunchMonitorAnalyticsPanel";
import { PlotsPanel } from "./PlotsPanel";
import { PuttingPanel } from "./PuttingPanel";
import { SimulationPanel } from "./SimulationPanel";
import { VariationPanel } from "./VariationPanel";

function TestVariationPanel() {
  // VariationPanel owns its workspace when no controlled snapshot is supplied.
  return <VariationPanel />;
}

beforeAll(() => {
  // jsdom has no canvas implementation — stub the 2D context so the
  // drawing components mount without crashing.
  // Every method returns the proxy itself so chained objects (e.g.
  // gradients with addColorStop) keep working.
  const ctx: unknown = new Proxy(function () {} as object, {
    get: (_target, prop) =>
      prop === "measureText" ? () => ({ width: 0 }) : () => ctx,
    set: () => true,
    apply: () => ctx,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    ctx as unknown as CanvasRenderingContext2D,
  );
});

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function assertHints(container: HTMLElement, panel: string) {
  const interactive = container.querySelectorAll(
    "button, select, input, textarea, canvas, [role='button'], [role='slider']",
  );
  expect(interactive.length, `${panel} should render controls`).toBeGreaterThan(
    0,
  );
  const missing: string[] = [];
  for (const el of interactive) {
    if (el.getAttribute("aria-hidden") === "true") {
      continue; // hidden helpers (e.g. the STL file input proxy)
    }
    const hint =
      el.getAttribute("title") ??
      el.getAttribute("aria-label") ??
      (el.closest("label")?.getAttribute("title") || null) ??
      (el.closest("[title]")?.getAttribute("title") || null);
    if (!hint || !hint.trim()) {
      missing.push(
        `${el.tagName.toLowerCase()}:${(el.textContent ?? "").slice(0, 40)}`,
      );
    }
  }
  expect(missing, `${panel} elements without hover hints`).toEqual([]);
}

describe("hover-hint completeness", () => {
  it("App shell (Explorer tab, nav, units, scenario inputs)", () => {
    const { container } = render(<App />);
    assertHints(container, "App/Explorer");
  });

  it("GlossaryPanel", () => {
    const { container } = render(<GlossaryPanel />);
    assertHints(container, "GlossaryPanel");
  });

  it("FlightExplorerPanel", () => {
    const { container } = render(<FlightExplorerPanel
      spatialTarget={spatialTargetFromRegion(DEFAULT_TARGET)}
      onSpatialTargetChange={() => undefined} />);
    assertHints(container, "FlightExplorerPanel");
  });

  it("VariationPanel", () => {
    const { container } = render(<TestVariationPanel />);
    assertHints(container, "VariationPanel");
  });

  it("LaunchMonitorAnalyticsPanel", () => {
    const { container } = render(<LaunchMonitorAnalyticsPanel />);
    assertHints(container, "LaunchMonitorAnalyticsPanel");
  });

  it("PlotsPanel", () => {
    const { container } = render(
      <PlotsPanel scenario={DEFAULT_SCENARIO} loftDeg={10.5} />,
    );
    assertHints(container, "PlotsPanel");
  });

  it("SimulationPanel", () => {
    const { container } = render(
      <SimulationPanel
        scenario={DEFAULT_SCENARIO}
        loftDeg={10.5}
        onScenarioChange={() => undefined}
        spatialTarget={spatialTargetFromRegion(DEFAULT_TARGET)}
        onSpatialTargetChange={() => undefined}
      />,
    );
    assertHints(container, "SimulationPanel");
  });

  it("PuttingPanel", () => {
    const { container } = render(<PuttingPanel />);
    assertHints(container, "PuttingPanel");
  });

  it("Derivation renders (read-only page, no unlabeled controls)", () => {
    const { container } = render(<Derivation scenario={DEFAULT_SCENARIO} />);
    const interactive = container.querySelectorAll("button, select, input");
    for (const el of interactive) {
      expect(
        el.getAttribute("title") ?? el.getAttribute("aria-label"),
      ).toBeTruthy();
    }
  });
});
