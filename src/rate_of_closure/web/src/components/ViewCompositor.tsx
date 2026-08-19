import { useRef, type KeyboardEvent, type ReactNode } from "react";

import {
  VIEW_KINDS,
  VIEW_LAYOUTS,
  toggleWorkspaceView,
  workspaceForSingleView,
  workspaceWithLayout,
  type ViewKind,
  type ViewLayout,
  type ViewWorkspace,
} from "../model/viewWorkspace";

const LABELS: Record<ViewKind, string> = {
  impact: "Impact",
  swing: "Swing",
  flight: "Flight",
};
const LAYOUT_LABELS: Record<ViewLayout, string> = {
  single: "Single",
  split_horizontal: "Split Horizontal",
  split_vertical: "Split Vertical",
  grid: "Grid",
};

interface Props {
  readonly workspace: ViewWorkspace;
  readonly onWorkspaceChange: (workspace: ViewWorkspace) => void;
  readonly renderViewport: (kind: ViewKind) => ReactNode;
  readonly runIdentity: string;
  readonly timeS: number;
}

function gridClass(layout: ViewLayout): string {
  if (layout === "split_vertical") return "grid grid-cols-1 gap-4";
  if (layout === "split_horizontal") return "grid gap-4 lg:grid-cols-2";
  if (layout === "grid") return "grid gap-4 lg:grid-cols-2";
  return "grid grid-cols-1 gap-4";
}

function hostClass(kind: ViewKind, layout: ViewLayout): string {
  const flightSpan = layout === "grid" && kind === "flight" ? " lg:col-span-2" : "";
  return "min-h-72 min-w-0 overflow-hidden rounded-xl border " +
    `border-slate-700 bg-slate-950/40 p-3${flightSpan}`;
}

/** Arrange independent viewport hosts around one synchronized run timeline. */
export function ViewCompositor(props: Props) {
  const visible = props.workspace.slots.map(({ kind }) => kind);
  const tabRefs = useRef<Partial<Record<ViewKind, HTMLButtonElement>>>({});
  const navigateTabs = (event: KeyboardEvent<HTMLButtonElement>, kind: ViewKind) => {
    const current = VIEW_KINDS.indexOf(kind);
    const destination = event.key === "Home"
      ? 0
      : event.key === "End"
        ? VIEW_KINDS.length - 1
        : event.key === "ArrowRight"
          ? (current + 1) % VIEW_KINDS.length
          : event.key === "ArrowLeft"
            ? (current - 1 + VIEW_KINDS.length) % VIEW_KINDS.length
            : null;
    if (destination === null) return;
    event.preventDefault();
    const next = VIEW_KINDS[destination];
    props.onWorkspaceChange(workspaceForSingleView(next, props.workspace));
    tabRefs.current[next]?.focus();
  };
  return (
    <section aria-label="Synchronized simulation view compositor" className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-700 p-3">
        <div role="tablist" aria-label="Synchronized viewport quick views"
          className="flex flex-wrap gap-2">
          {VIEW_KINDS.map((kind) => {
            const selected = props.workspace.layout === "single" && visible[0] === kind;
            return (
              <button key={kind} type="button" role="tab" aria-selected={selected}
                ref={(element) => { tabRefs.current[kind] = element ?? undefined; }}
                tabIndex={selected ? 0 : -1}
                onKeyDown={(event) => navigateTabs(event, kind)}
                onClick={() => props.onWorkspaceChange(
                  workspaceForSingleView(kind, props.workspace),
                )}
                className={`rounded-full border px-3 py-1 text-sm ${selected
                  ? "border-sky-400 bg-sky-500/10 text-sky-200"
                  : "border-slate-600 text-slate-300"}`}>
                {LABELS[kind]}
              </button>
            );
          })}
        </div>
        <label className="text-sm text-slate-300">
          Viewport Layout
          <select
            aria-label="Viewport layout"
            value={props.workspace.layout}
            onChange={(event) => props.onWorkspaceChange(workspaceWithLayout(
              event.target.value as ViewLayout,
              props.workspace,
            ))}
            className="ml-2 rounded border border-slate-600 bg-slate-900 px-2 py-1"
          >
            {VIEW_LAYOUTS.map((layout) => (
              <option key={layout} value={layout}>{LAYOUT_LABELS[layout]}</option>
            ))}
          </select>
        </label>
        {VIEW_KINDS.map((kind) => (
          <label key={kind} className="flex items-center gap-1 text-sm text-slate-300">
            <input
              type="checkbox"
              aria-label={`Show ${LABELS[kind]} viewport`}
              checked={visible.includes(kind)}
              onChange={() => props.onWorkspaceChange(toggleWorkspaceView(props.workspace, kind))}
            />
            {LABELS[kind]}
          </label>
        ))}
      </div>
      <div className={gridClass(props.workspace.layout)}>
        {props.workspace.slots.map(({ id, kind }) => (
          <article
            key={id}
            role="region"
            aria-label={`${LABELS[kind]} synchronized viewport`}
            data-viewport-id={id}
            data-run-identity={props.runIdentity}
            data-playback-time={props.timeS.toFixed(3)}
            className={hostClass(kind, props.workspace.layout)}
          >
            <h3 className="mb-2 text-sm font-semibold text-sky-200">{LABELS[kind]} View</h3>
            {props.renderViewport(kind)}
          </article>
        ))}
      </div>
    </section>
  );
}
