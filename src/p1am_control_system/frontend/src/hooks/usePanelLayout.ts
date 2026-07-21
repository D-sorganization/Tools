import { useCallback, useEffect, useState } from "react";

import {
  type PanelLayout,
  loadPanelLayout,
  savePanelLayout,
  movePanel,
  clampPanelHeight,
} from "../lib/panelLayout";

export interface UsePanelLayout {
  /** Current panel id order. */
  order: string[];
  /** Per-panel body height (px); absent means "natural height". */
  heights: Record<string, number>;
  /** Move `dragged` to sit before `target` (persisted). */
  move: (draggedId: string, targetId: string) => void;
  /** Set a panel's body height in px (clamped + persisted). */
  setHeight: (id: string, px: number) => void;
  /** Restore the declared order and clear custom heights. */
  reset: () => void;
}

/**
 * React binding over the pure {@link file://../lib/panelLayout.ts} model: loads a
 * region's persisted arrangement, exposes reorder/resize/reset, and writes every
 * change back to localStorage. All setters are stable (empty-dep useCallback) so
 * they don't retrigger the memoized, 10 Hz-rerendering panels.
 *
 * @param regionId Stable id for this panel group (usually a tab id).
 * @param knownIds The panel ids that currently exist, in their declared order.
 *   Pass a STABLE reference (module constant / memoized) — it seeds the initial
 *   load and the reset order.
 */
export function usePanelLayout(
  regionId: string,
  knownIds: string[],
): UsePanelLayout {
  const [layout, setLayout] = useState<PanelLayout>(() =>
    loadPanelLayout(regionId, knownIds),
  );

  useEffect(() => {
    savePanelLayout(regionId, layout);
  }, [regionId, layout]);

  const move = useCallback((draggedId: string, targetId: string) => {
    setLayout((l) => ({ ...l, order: movePanel(l.order, draggedId, targetId) }));
  }, []);

  const setHeight = useCallback((id: string, px: number) => {
    setLayout((l) => ({
      ...l,
      heights: { ...l.heights, [id]: clampPanelHeight(px) },
    }));
  }, []);

  const reset = useCallback(() => {
    setLayout({ order: [...knownIds], heights: {} });
  }, [knownIds]);

  return { order: layout.order, heights: layout.heights, move, setHeight, reset };
}
