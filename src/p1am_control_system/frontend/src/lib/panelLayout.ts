/**
 * Persisted, per-region panel arrangement: order + per-panel body height.
 *
 * Lets the operator drag dashboard panels into the order they want and resize
 * them vertically, with the arrangement remembered across reloads (like the tab
 * bar's order in {@link file://./tabs.ts}). Kept as pure functions with explicit
 * precondition checks (DbC) and no React/DOM here, so the reorder/clamp/reconcile
 * logic is unit-testable and reused by the {@link usePanelLayout} hook and the
 * PanelStack component (DRY). A "region" is a stable id for one panel group
 * (typically a tab id), so each group keeps its own independent arrangement.
 */

/** A region's arrangement: panel id order + optional per-panel body height (px). */
export interface PanelLayout {
  order: string[];
  heights: Record<string, number>;
}

/** Vertical resize bounds (px) — keeps a panel usable and on-screen. */
export const MIN_PANEL_HEIGHT = 120;
export const MAX_PANEL_HEIGHT = 1600;

/**
 * Clamp a requested body height to the supported range.
 * @throws TypeError if `px` is not a finite number.
 */
export function clampPanelHeight(px: number): number {
  if (typeof px !== "number" || !Number.isFinite(px)) {
    throw new TypeError("px must be a finite number");
  }
  return Math.max(MIN_PANEL_HEIGHT, Math.min(MAX_PANEL_HEIGHT, Math.round(px)));
}

/**
 * Reconcile a saved id order with the panels that currently exist: keep the
 * saved order for ids that still exist (de-duped), drop unknown ids, and append
 * any panels added since the order was saved — so a new panel is never silently
 * hidden and a removed one never lingers. Mirrors the tab-bar reconcile.
 *
 * @throws TypeError if `saved` or `knownIds` is not an array.
 */
export function reconcilePanelOrder(
  saved: readonly unknown[],
  knownIds: readonly string[],
): string[] {
  if (!Array.isArray(saved)) throw new TypeError("saved must be an array");
  if (!Array.isArray(knownIds)) throw new TypeError("knownIds must be an array");
  const known = new Set(knownIds);
  const seen = new Set<string>();
  const kept: string[] = [];
  for (const id of saved) {
    if (typeof id === "string" && known.has(id) && !seen.has(id)) {
      kept.push(id);
      seen.add(id);
    }
  }
  for (const id of knownIds) {
    if (!seen.has(id)) kept.push(id);
  }
  return kept;
}

/**
 * Move `draggedId` to sit immediately before `targetId` in `order`. A no-op
 * (returns a copy) when the two ids are equal or either is absent, so a stray
 * drop can never drop or duplicate a panel.
 *
 * @throws TypeError if `order` is not an array or the ids are not strings.
 */
export function movePanel(
  order: readonly string[],
  draggedId: string,
  targetId: string,
): string[] {
  if (!Array.isArray(order)) throw new TypeError("order must be an array");
  if (typeof draggedId !== "string" || typeof targetId !== "string") {
    throw new TypeError("draggedId and targetId must be strings");
  }
  if (draggedId === targetId) return [...order];
  if (!order.includes(draggedId) || !order.includes(targetId)) return [...order];
  const without = order.filter((id) => id !== draggedId);
  const at = without.indexOf(targetId);
  return [...without.slice(0, at), draggedId, ...without.slice(at)];
}

function keyFor(regionId: string): string {
  return `p1am.panelLayout.${regionId}.v1`;
}

/**
 * Load a region's persisted layout, reconciled against the panels that exist
 * now. Falls back to the declared `knownIds` order (no custom heights) when
 * nothing is stored or storage is corrupt/unavailable.
 *
 * @throws TypeError if `regionId` is not a string or `knownIds` is not an array.
 */
export function loadPanelLayout(
  regionId: string,
  knownIds: readonly string[],
): PanelLayout {
  if (typeof regionId !== "string") throw new TypeError("regionId must be a string");
  if (!Array.isArray(knownIds)) throw new TypeError("knownIds must be an array");
  try {
    const raw = localStorage.getItem(keyFor(regionId));
    if (raw) {
      const saved = JSON.parse(raw) as Partial<PanelLayout>;
      const order = reconcilePanelOrder(
        Array.isArray(saved.order) ? saved.order : [],
        knownIds,
      );
      const heights: Record<string, number> = {};
      const savedHeights = saved.heights;
      if (savedHeights && typeof savedHeights === "object") {
        for (const id of knownIds) {
          const h = (savedHeights as Record<string, unknown>)[id];
          if (typeof h === "number" && Number.isFinite(h)) {
            heights[id] = clampPanelHeight(h);
          }
        }
      }
      return { order, heights };
    }
  } catch {
    /* corrupt/unavailable storage — fall through to defaults */
  }
  return { order: [...knownIds], heights: {} };
}

/** Persist a region's layout (best-effort; a full/blocked store is non-fatal). */
export function savePanelLayout(regionId: string, layout: PanelLayout): void {
  if (typeof regionId !== "string") throw new TypeError("regionId must be a string");
  try {
    localStorage.setItem(keyFor(regionId), JSON.stringify(layout));
  } catch {
    /* storage unavailable — non-fatal */
  }
}
