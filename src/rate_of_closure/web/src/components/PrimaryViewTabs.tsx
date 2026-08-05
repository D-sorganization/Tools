import { useRef, type DragEvent, type KeyboardEvent } from "react";

import {
  movePrimaryView,
  primaryViewLabel,
  shiftPrimaryView,
  type PrimaryViewId,
  type PrimaryViewState,
} from "../model/viewPreferences";

const DRAG_DATA_TYPE = "application/x-rate-of-closure-primary-view";

export function PrimaryViewTabs({
  state,
  onActiveChange,
  onOrderChange,
}: {
  state: PrimaryViewState;
  onActiveChange: (view: PrimaryViewId) => void;
  onOrderChange: (order: PrimaryViewId[]) => void;
}) {
  const dragSource = useRef<PrimaryViewId | null>(null);

  const reorderFromDrag = (event: DragEvent, destination: PrimaryViewId) => {
    event.preventDefault();
    const moving =
      (event.dataTransfer.getData(DRAG_DATA_TYPE) as PrimaryViewId) ||
      dragSource.current;
    const bounds = event.currentTarget.getBoundingClientRect();
    const placeAfter = event.clientX > bounds.left + bounds.width / 2;
    if (moving !== null) {
      onOrderChange(movePrimaryView(state.order, moving, destination, placeAfter));
    }
    dragSource.current = null;
  };

  const handleKey = (event: KeyboardEvent, view: PrimaryViewId) => {
    const horizontal = event.key === "ArrowLeft" || event.key === "ArrowRight";
    if (event.altKey && horizontal) {
      event.preventDefault();
      const offset = event.key === "ArrowLeft" ? -1 : 1;
      onOrderChange(shiftPrimaryView(state.order, view, offset));
      return;
    }
    if (!horizontal && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    const current = state.order.indexOf(view);
    const destination = event.key === "Home"
      ? 0
      : event.key === "End"
        ? state.order.length - 1
        : (current + (event.key === "ArrowLeft" ? -1 : 1) + state.order.length)
          % state.order.length;
    const next = state.order[destination];
    onActiveChange(next);
    document.getElementById(`primary-tab-${next}`)?.focus();
  };

  return (
    <nav aria-label="Primary views" className="mb-5">
      <p className="mb-2 text-xs text-slate-500">
        Drag a tab to personalize this workspace. The order and active view are saved on this device.
      </p>
      <div
        role="tablist"
        aria-label="Workbench views"
        className="flex gap-2 overflow-x-auto pb-2"
      >
        {state.order.map((view) => {
          const label = primaryViewLabel(view);
          const selected = state.active === view;
          return (
            <button
              id={`primary-tab-${view}`}
              key={view}
              type="button"
              role="tab"
              draggable
              aria-selected={selected}
              aria-controls={`primary-panel-${view}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => onActiveChange(view)}
              onKeyDown={(event) => handleKey(event, view)}
              onDragStart={(event) => {
                dragSource.current = view;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData(DRAG_DATA_TYPE, view);
              }}
              onDragEnd={() => {
                dragSource.current = null;
              }}
              onDragOver={(event) => {
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
              }}
              onDrop={(event) => reorderFromDrag(event, view)}
              title={`Open ${label}. Drag to reorder; Alt+Left/Right also moves this tab.`}
              className={
                "group flex shrink-0 cursor-grab items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-medium transition-all active:cursor-grabbing " +
                (selected
                  ? "border-sky-400/60 bg-sky-500/10 text-sky-300 shadow-[0_0_18px_rgba(56,189,248,0.25)]"
                  : "border-slate-700/80 bg-slate-900/60 text-slate-300 hover:border-slate-500 hover:text-slate-100")
              }
            >
              <span
                aria-hidden="true"
                className="select-none text-[10px] tracking-[-0.15em] text-slate-500 group-hover:text-slate-300"
              >
                ⋮⋮
              </span>
              {label}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
