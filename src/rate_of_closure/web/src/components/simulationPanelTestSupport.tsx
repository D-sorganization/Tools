import { render } from "@testing-library/react";
import { vi } from "vitest";

import type { ClubSpec } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import type { SpatialTargetTs } from "../model/spatialTarget";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";
import { SimulationPanel } from "./SimulationPanel";

export const defaultSpatialTarget = spatialTargetFromRegion(DEFAULT_TARGET);

export function installSimulationPanelTestEnvironment(): void {
  const stored = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      clear: () => stored.clear(),
      getItem: (key: string) => stored.get(key) ?? null,
      removeItem: (key: string) => stored.delete(key),
      setItem: (key: string, value: string) => stored.set(key, value),
    },
  });
  const context: unknown = new Proxy(function () {} as object, {
    get: (_target, property) =>
      property === "measureText" ? () => ({ width: 0 }) : () => context,
    set: () => true,
    apply: () => context,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    context as CanvasRenderingContext2D,
  );
}

export function renderSimulationPanel(
  clubSpec?: ClubSpec | null,
  onSpatialTargetChange: (target: SpatialTargetTs) => void = () => undefined,
) {
  return render(
    <SimulationPanel
      scenario={{ ...DEFAULT_SCENARIO, impactOffsetToeMm: 20 }}
      loftDeg={10.5}
      clubSpec={clubSpec}
      onScenarioChange={() => undefined}
      spatialTarget={defaultSpatialTarget}
      onSpatialTargetChange={onSpatialTargetChange}
    />,
  );
}
