import { useCallback, useMemo, useRef, useState } from "react";
import {
  createViewport,
  isLive,
  panBy as vpPanBy,
  reset as vpReset,
  resolveVisible,
  setPaused as vpSetPaused,
  zoomBy as vpZoomBy,
  zoomToRange as vpZoomToRange,
  type DomainBounds,
  type SpanLimits,
  type TrendViewport,
  type VisibleRange,
} from "../lib/trendViewport";

/**
 * React binding for the {@link TrendViewport} model — the ONE place trend charts
 * get pan / zoom / pause / drag-to-zoom behavior (DRY). It owns the viewport
 * state and a pixel drag-selection; the chart supplies its data bounds and a
 * pixel→domain mapping (LOD: the hook never needs the chart's geometry, only a
 * converter), so this works for index-based and time-based charts alike.
 */
export interface UseTrendViewportOptions {
  /** Initial (and reset) visible span, in the chart's domain units. */
  defaultSpan: number;
  minSpan: number;
  maxSpan: number;
}

/** A drag selection in plot pixels, for drawing the zoom-region overlay. */
export interface PixelSelection {
  fromPx: number;
  toPx: number;
}

export interface TrendViewportApi {
  viewport: TrendViewport;
  paused: boolean;
  /** True when following the live edge (not paused, not panned back). */
  live: boolean;
  limits: SpanLimits;
  /** Resolve the on-screen window for the current data bounds. */
  resolve: (bounds: DomainBounds) => VisibleRange;
  togglePause: () => void;
  setPaused: (paused: boolean) => void;
  reset: () => void;
  panBy: (deltaUnits: number, bounds: DomainBounds) => void;
  zoomBy: (factor: number, focus: number, bounds: DomainBounds) => void;
  /** In-progress drag selection (plot pixels), or null. */
  selectionPx: PixelSelection | null;
  startSelect: (px: number) => void;
  moveSelect: (px: number) => void;
  /**
   * Finish a drag: if it spans at least `minPx`, zoom to it. `pxToUnit` maps a
   * plot pixel to a domain unit (the chart owns that scale).
   */
  endSelect: (
    bounds: DomainBounds,
    pxToUnit: (px: number) => number,
    minPx?: number,
  ) => void;
  cancelSelect: () => void;
}

export function useTrendViewport(
  opts: UseTrendViewportOptions,
): TrendViewportApi {
  const limits = useMemo<SpanLimits>(
    () => ({ minSpan: opts.minSpan, maxSpan: opts.maxSpan }),
    [opts.minSpan, opts.maxSpan],
  );
  const [viewport, setViewport] = useState<TrendViewport>(() =>
    createViewport(opts.defaultSpan),
  );
  const [selectionPx, setSelectionPx] = useState<PixelSelection | null>(null);
  // Mirror the selection in a ref so endSelect reads the latest without nesting
  // setState calls inside an updater (which StrictMode can double-invoke).
  const selRef = useRef<PixelSelection | null>(null);

  const resolve = useCallback(
    (bounds: DomainBounds) => resolveVisible(viewport, bounds, limits),
    [viewport, limits],
  );

  const togglePause = useCallback(
    () => setViewport((v) => vpSetPaused(v, !v.paused)),
    [],
  );
  const setPaused = useCallback(
    (paused: boolean) => setViewport((v) => vpSetPaused(v, paused)),
    [],
  );
  const reset = useCallback(() => {
    selRef.current = null;
    setSelectionPx(null);
    setViewport(vpReset(opts.defaultSpan));
  }, [opts.defaultSpan]);
  const panBy = useCallback(
    (deltaUnits: number, bounds: DomainBounds) =>
      setViewport((v) => vpPanBy(v, deltaUnits, bounds, limits)),
    [limits],
  );
  const zoomBy = useCallback(
    (factor: number, focus: number, bounds: DomainBounds) =>
      setViewport((v) => vpZoomBy(v, factor, focus, bounds, limits)),
    [limits],
  );

  const startSelect = useCallback((px: number) => {
    selRef.current = { fromPx: px, toPx: px };
    setSelectionPx(selRef.current);
  }, []);
  const moveSelect = useCallback((px: number) => {
    if (!selRef.current) return;
    selRef.current = { fromPx: selRef.current.fromPx, toPx: px };
    setSelectionPx(selRef.current);
  }, []);
  const cancelSelect = useCallback(() => {
    selRef.current = null;
    setSelectionPx(null);
  }, []);
  const endSelect = useCallback(
    (bounds: DomainBounds, pxToUnit: (px: number) => number, minPx = 4) => {
      const sel = selRef.current;
      selRef.current = null;
      setSelectionPx(null);
      if (sel && Math.abs(sel.toPx - sel.fromPx) >= minPx) {
        setViewport((v) =>
          vpZoomToRange(v, pxToUnit(sel.fromPx), pxToUnit(sel.toPx), bounds, limits),
        );
      }
    },
    [limits],
  );

  return {
    viewport,
    paused: viewport.paused,
    live: isLive(viewport),
    limits,
    resolve,
    togglePause,
    setPaused,
    reset,
    panBy,
    zoomBy,
    selectionPx,
    startSelect,
    moveSelect,
    endSelect,
    cancelSelect,
  };
}
