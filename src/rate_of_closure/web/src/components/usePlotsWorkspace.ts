/** Commit-phase, bounded computation cache for the managed Plots workspace. */
import { useCallback, useEffect, useRef, useState } from "react";

import { type ImpactScenario } from "../model/impact";
import { type PlotContext } from "../model/plotcatalog";
import {
  computePlotData,
  specToJson,
  type PlotData,
  type PlotSimulationExecutor,
  type PlotSpec,
} from "../model/plotspec";
import { validatePlotWorkspace } from "../model/plotWorkspaceLimits";
import { runSimulation, type SimulationInput } from "../model/simulation";

export interface ManagedPlot {
  id: number;
  label: string;
  spec: PlotSpec;
}

export interface ComputedPlot {
  plot: ManagedPlot;
  data: PlotData | null;
  error: string | null;
  pending: boolean;
  retained: boolean;
}

interface Params {
  initialPlot: ManagedPlot;
  scenario: ImpactScenario;
  loftDeg: number;
  executeSimulation?: PlotSimulationExecutor;
}

interface CacheEntry {
  specKey: string;
  data: PlotData | null;
  error: string | null;
  signature: string;
  executor: PlotSimulationExecutor;
}

interface CacheSnapshot {
  signature: string;
  executor: PlotSimulationExecutor;
  context: PlotContext | null;
  entries: Map<number, CacheEntry>;
}

function plotContext(
  scenario: ImpactScenario,
  loftDeg: number,
  executeSimulation: PlotSimulationExecutor,
): PlotContext {
  const input: SimulationInput = {
    sourceKind: "manual",
    clubheadSpeedMph: scenario.clubheadSpeedMph,
    omegaDps: [0, 0, 0],
    loftDeg,
    impactOffsetToeMm: scenario.impactOffsetToeMm,
    impactOffsetHighMm: scenario.impactOffsetHighMm,
    planeYawDeg: 0,
    planeSideTiltDeg: -45,
    planeForwardTiltDeg: 0,
    impactTimeS: null,
    swingDurationS: 1.5,
  };
  return { scenario, input, run: executeSimulation(input) };
}

function specKey(spec: PlotSpec): string {
  return JSON.stringify(specToJson(spec));
}

export function usePlotsWorkspace({
  initialPlot,
  scenario,
  loftDeg,
  executeSimulation = runSimulation,
}: Params) {
  const [plots, setPlots] = useState<ManagedPlot[]>([initialPlot]);
  const [selectedId, setSelectedId] = useState(initialPlot.id);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const nextId = useRef(initialPlot.id + 1);
  const signature = JSON.stringify({ scenario, loftDeg });
  const initialCache = useRef<CacheSnapshot>({
    signature: "",
    executor: executeSimulation,
    context: null,
    entries: new Map(),
  });
  const cacheRef = initialCache;
  const [snapshot, setSnapshot] = useState(initialCache.current);

  useEffect(() => {
    const prior = cacheRef.current;
    const authorityChanged =
      prior.signature !== signature || prior.executor !== executeSimulation;
    let context = authorityChanged ? null : prior.context;
    const fallback = prior.entries;
    const entries = authorityChanged
      ? new Map<number, CacheEntry>()
      : new Map(prior.entries);
    const failedEntry = (plot: ManagedPlot, key: string, error: unknown): CacheEntry => {
      const retained = fallback.get(plot.id);
      if (retained?.specKey === key && retained.data !== null) {
        return { ...retained, error: String(error).slice(0, 512) };
      }
      return {
        specKey: key,
        data: null,
        error: String(error).slice(0, 512),
        signature,
        executor: executeSimulation,
      };
    };
    try {
      context ??= plotContext(scenario, loftDeg, executeSimulation);
      for (const plot of plots) {
        const key = specKey(plot.spec);
        if (entries.get(plot.id)?.specKey === key) continue;
        try {
          entries.set(plot.id, {
            specKey: key,
            data: computePlotData(plot.spec, context, executeSimulation),
            error: null,
            signature,
            executor: executeSimulation,
          });
        } catch (error) {
          entries.set(plot.id, failedEntry(plot, key, error));
        }
      }
    } catch (error) {
      context = null;
      for (const plot of plots) {
        const key = specKey(plot.spec);
        entries.set(plot.id, failedEntry(plot, key, error));
      }
    }
    const next = { signature, executor: executeSimulation, context, entries };
    cacheRef.current = next;
    setSnapshot(next);
  }, [cacheRef, executeSimulation, loftDeg, plots, scenario, signature]);

  const authorityCurrent =
    snapshot.signature === signature && snapshot.executor === executeSimulation;
  const computed: ComputedPlot[] = plots.map((plot) => {
    const entry = snapshot.entries.get(plot.id);
    if (!entry || entry.specKey !== specKey(plot.spec)) {
      return { plot, data: null, error: null, pending: true, retained: false };
    }
    if (!authorityCurrent) {
      return {
        plot,
        data: entry.data,
        error: entry.data
          ? "Computing replacement; prior accepted plot retained"
          : entry.error,
        pending: entry.data === null,
        retained: entry.data !== null,
      };
    }
    return {
      plot,
      data: entry.data,
      error: entry.error,
      pending: false,
      retained: entry.signature !== signature || entry.executor !== executeSimulation,
    };
  });

  const addPlot = useCallback(
    (label: string, spec: PlotSpec): void => {
      try {
        validatePlotWorkspace([...plots.map((plot) => plot.spec), spec]);
      } catch (error) {
        setWorkspaceError(String(error));
        return;
      }
      const id = nextId.current;
      nextId.current += 1;
      setPlots([...plots, { id, label, spec }]);
      setSelectedId(id);
      setWorkspaceError(null);
    },
    [plots],
  );

  const removeSelected = useCallback((): void => {
    const next = plots.filter((plot) => plot.id !== selectedId);
    setPlots(next);
    if (next.length) setSelectedId(next[next.length - 1].id);
  }, [plots, selectedId]);

  return {
    addPlot,
    computed,
    context: authorityCurrent ? snapshot.context : null,
    plots,
    removeSelected,
    selectedId,
    setSelectedId,
    setWorkspaceError,
    workspaceError,
  };
}
