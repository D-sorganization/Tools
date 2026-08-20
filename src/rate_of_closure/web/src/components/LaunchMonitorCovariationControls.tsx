import type { CovariationUiSettings } from "../model/launchMonitorCovariation";
import type { CovariationController } from "./useCovariationController";

const field = "w-full rounded border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-slate-100 focus:border-sky-400 focus:outline-none";

function IdentityControls({ controller, identityLocked = false }: {
  controller: CovariationController; identityLocked?: boolean;
}) {
  const { settings, groupingColumns, players, update } = controller;
  return <>
    <label className="text-sm text-slate-300">Player identity/grouping column
      <select className={`${field} mt-1`} value={settings.playerColumn}
        disabled={identityLocked}
        title="Explicitly choose the column containing stable player identifiers; blank identities are excluded"
        aria-label="Covariation player column"
        onChange={(event) => update({ playerColumn: event.target.value, selectedPlayer: "" })}>
        <option value="">Select an identity column</option>
        {groupingColumns.map((column) => <option key={column}>{column}</option>)}
      </select>
    </label>
    <label className="text-sm text-slate-300">Player focus
      <select className={`${field} mt-1`} value={settings.selectedPlayer}
        title="Show all players or focus the plot and player table on one player"
        aria-label="Covariation player focus" onChange={(event) => update({ selectedPlayer: event.target.value })}>
        <option value="">All eligible players</option>
        {players.map((player) => <option key={player}>{player}</option>)}
      </select>
    </label>
  </>;
}

function VariableControls({ controller }: { controller: CovariationController }) {
  const { settings, numeric, update } = controller;
  const selector = (axis: "X" | "Y", value: string) => <label className="text-sm text-slate-300">
    {axis} variable
    <select className={`${field} mt-1`} value={value}
      title={`Choose any populated numeric metric for the ${axis === "X" ? "horizontal" : "vertical"} axis`}
      aria-label={`Covariation ${axis} variable`}
      onChange={(event) => update(axis === "X" ? { xColumn: event.target.value } : { yColumn: event.target.value })}>
      {numeric.map((column) => <option key={column}>{column}</option>)}
    </select>
  </label>;
  return <>{selector("X", settings.xColumn)}{selector("Y", settings.yColumn)}</>;
}

function InferenceControls({ controller }: { controller: CovariationController }) {
  const { settings, update } = controller;
  return <>
    <label className="text-sm text-slate-300">Coefficient
      <select className={`${field} mt-1`} value={settings.method}
        title="Pearson measures linear association; Spearman measures monotonic rank association"
        aria-label="Covariation coefficient method"
        onChange={(event) => update({ method: event.target.value as CovariationUiSettings["method"] })}>
        <option value="pearson">Pearson r</option><option value="spearman">Spearman ρ</option>
      </select>
    </label>
    <label className="text-sm text-slate-300">Minimum N/player
      <input className={`${field} mt-1`} type="number" min="4" step="1" value={settings.minSamples}
        title="Players below this pairwise-complete shot count are shown but excluded from meta-analysis"
        aria-label="Covariation minimum player sample count"
        onChange={(event) => update({ minSamples: Number(event.target.value) })} />
    </label>
    <label className="text-sm text-slate-300">Confidence
      <input className={`${field} mt-1`} type="number" min="0.51" max="0.999" step="0.01"
        value={settings.confidenceLevel} title="Confidence level for per-player Fisher-z Pearson intervals"
        aria-label="Covariation confidence level"
        onChange={(event) => update({ confidenceLevel: Number(event.target.value) })} />
    </label>
  </>;
}

export function LaunchMonitorCovariationControls({ controller, identityLocked = false }: {
  controller: CovariationController;
  identityLocked?: boolean;
}) {
  return <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
    <IdentityControls controller={controller} identityLocked={identityLocked} />
    <VariableControls controller={controller} />
    <InferenceControls controller={controller} />
    <div className="flex items-end gap-2">
      <button type="button" onClick={controller.analyze}
        title="Calculate player, centered, pooled, between, and meta associations"
        className="rounded bg-emerald-600 px-3 py-2 text-sm font-semibold hover:bg-emerald-500">Analyze Player Covariation</button>
      <button type="button" onClick={controller.rankPairs}
        title="Scan all numeric column pairs using the current player and sample rules"
        className="rounded border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800">Rank Variable Pairs</button>
    </div>
  </div>;
}

export function CovariationGuide() {
  return <details className="rounded border border-slate-700/70 bg-slate-950/40 p-3">
    <summary className="cursor-pointer text-sm font-semibold text-sky-200"
      title="Show formulas, assumptions, and interpretation limits">Covariation calculation guide</summary>
    <div className="mt-3 space-y-2 text-xs text-slate-300">
      <p><strong>Within-player centered:</strong> correlate xᵢⱼ−x̄ⱼ with yᵢⱼ−ȳⱼ across pairwise-complete shots. <strong>Between-player:</strong> unweighted correlation of player means.</p>
      <p><strong>Meta-analysis:</strong> transform each eligible Pearson r with Fisher z. Fixed weights are n−3. Random effects use DerSimonian–Laird τ² and weights 1/(1/(n−3)+τ²). I² reports estimated heterogeneity.</p>
      <p><strong>Inference:</strong> association does not establish causality. Measurement error, range restriction, omitted variables, session conditions, and repeated-shot selection can change estimates. Pair ranking is exploratory and creates a multiple-comparisons burden.</p>
    </div>
  </details>;
}
