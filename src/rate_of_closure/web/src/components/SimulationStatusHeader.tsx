import { type WebSourceKind } from "../model/simulation";

const MODEL_LABELS: Record<WebSourceKind, string> = {
  manual: "Manual Constant-Twist Delivery",
  double_pendulum: "Double Pendulum",
  triple_pendulum: "Triple Pendulum",
};

export function SimulationStatusHeader({
  sourceKind,
  status,
  warning,
}: {
  sourceKind: WebSourceKind;
  status: string;
  warning: boolean;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-700/80 bg-slate-900/80 px-5 py-3 shadow-lg shadow-black/20 lg:col-span-2">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-sky-400">
          Active Swing Model
        </p>
        <p className="text-base font-semibold text-slate-100">
          {MODEL_LABELS[sourceKind]}
        </p>
      </div>
      <p
        role="status"
        aria-live="polite"
        className={
          "rounded-full border px-3 py-1.5 text-sm font-medium " +
          (warning
            ? "border-amber-500/50 bg-amber-950/30 text-amber-200"
            : "border-emerald-500/50 bg-emerald-950/30 text-emerald-200")
        }
      >
        {status}
      </p>
    </header>
  );
}
