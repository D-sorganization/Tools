import { FIELD_GUIDANCE } from "../model/units";
import { type WebSourceKind } from "../model/simulation";

const MODEL_DETAILS: Record<
  WebSourceKind,
  { label: string; description: string }
> = {
  manual: {
    label: "Manual Constant-Twist Delivery",
    description:
      "Inspect a specified constant-twist delivery without solving linked-body dynamics.",
  },
  double_pendulum: {
    label: "Double Pendulum",
    description:
      "Two linked segments with passive or prescribed torques and optional fixed-joint constraints.",
  },
  triple_pendulum: {
    label: "Triple Pendulum",
    description:
      "Three linked segments. Prescribed-torque and fixed-joint controls are unavailable for this model.",
  },
};

function statusClasses(status: string, warning: boolean): string {
  if (status.startsWith("Run failed")) {
    return "border-red-500/60 bg-red-950/40 text-red-200";
  }
  if (warning) return "border-amber-500/50 bg-amber-950/30 text-amber-200";
  if (status === "Not run") {
    return "border-slate-600/70 bg-slate-950/50 text-slate-300";
  }
  return "border-emerald-500/50 bg-emerald-950/30 text-emerald-200";
}

export function SimulationStatusHeader({
  sourceKind,
  onSourceKindChange,
  status,
  warning,
}: {
  sourceKind: WebSourceKind;
  onSourceKindChange: (source: WebSourceKind) => void;
  status: string;
  warning: boolean;
}) {
  const activeModel = MODEL_DETAILS[sourceKind];
  return (
    <header className="grid gap-4 rounded-xl border border-slate-700/80 bg-slate-900/80 px-5 py-4 shadow-lg shadow-black/20 sm:grid-cols-[minmax(220px,1fr)_minmax(260px,1.35fr)] lg:col-span-2 lg:grid-cols-[minmax(220px,0.8fr)_minmax(280px,1fr)_auto] lg:items-center">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-sky-400">
          Simulation Workspace
        </p>
        <h2 className="mt-0.5 text-lg font-semibold text-slate-100">
          Swing-to-Impact Simulation
        </h2>
        <p className="mt-1 text-xs text-slate-500">
          Select the governing swing model, configure applicable inputs, then run.
        </p>
      </div>

      <label className="block text-sm" title={FIELD_GUIDANCE.swingSource}>
        <span className="mb-1 flex items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
          <span>Swing Model</span>
          <span className="normal-case tracking-normal text-slate-500">
            Model-specific controls follow
          </span>
        </span>
        <select
          aria-label="Swing Source"
          value={sourceKind}
          title={FIELD_GUIDANCE.swingSource}
          onChange={(event) =>
            onSourceKindChange(event.target.value as WebSourceKind)
          }
          className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 font-medium text-slate-100 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400/30"
        >
          {(Object.keys(MODEL_DETAILS) as WebSourceKind[]).map((source) => (
            <option key={source} value={source}>
              {MODEL_DETAILS[source].label}
            </option>
          ))}
        </select>
        <span className="mt-1.5 block text-xs leading-relaxed text-slate-400">
          {activeModel.description}
        </span>
      </label>

      <div className="sm:col-span-2 lg:col-span-1 lg:text-right">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Run Status
        </p>
        <p
          role="status"
          aria-label="Simulation run status"
          aria-live="polite"
          className={`inline-flex max-w-full rounded-full border px-3 py-1.5 text-left text-sm font-medium ${statusClasses(status, warning)}`}
        >
          {status}
        </p>
      </div>
    </header>
  );
}
