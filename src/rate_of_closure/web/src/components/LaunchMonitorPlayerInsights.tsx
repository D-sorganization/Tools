import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import {
  dispersionSummary,
  metricLabel,
  sessionTrend,
  sessionTrendExportRows,
  STROKES_GAINED_REFERENCE,
  strokesGainedProxy,
} from "../model/launchMonitorPlayerAnalytics";
import { downloadCsv, downloadSvg } from "../model/launchMonitorDownloads";
import type { CovariationUiSettings } from "../model/launchMonitorCovariation";
import { DispersionPlot, SessionTrendPlot } from "./LaunchMonitorCharts";
import { LaunchMonitorCovariation } from "./LaunchMonitorCovariation";

const card = "rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20";

const finite = (value: number | null, digits = 3) => value === null ? "—" : value.toFixed(digits);

export function LaunchMonitorPlayerInsights({
  rows, outcome, targetDistanceYards, setTargetDistanceYards,
  covariationSettings, setCovariationSettings,
}: {
  rows: LaunchMonitorRow[];
  outcome: string;
  targetDistanceYards: number;
  setTargetDistanceYards: (value: number) => void;
  covariationSettings: CovariationUiSettings;
  setCovariationSettings: (value: CovariationUiSettings) => void;
}) {
  const dispersion = dispersionSummary(rows);
  const gained = strokesGainedProxy(rows, targetDistanceYards);
  const trend = sessionTrend(rows, outcome);
  const meanGained = gained.length
    ? gained.reduce((sum, shot) => sum + shot.strokesGainedProxy, 0) / gained.length : null;

  return <div className="space-y-5">
    <div className={card}>
      <LaunchMonitorCovariation rows={rows} savedSettings={covariationSettings}
        onSettingsChange={setCovariationSettings} />
    </div>
    <div className={card}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold text-slate-200">Directional Dispersion</h3>
          <p className="mt-1 text-xs text-slate-400" title="Signed lateral deviation uses the app-native convention: negative is left of target and positive is right.">
            Signed yards from target: left is negative, right is positive. Hover a shot for backing values.
          </p>
        </div>
        {dispersion && <div className="flex gap-2">
          <button type="button" title="Download the plotted shot-level carry and signed lateral values as CSV"
            onClick={() => downloadCsv("launch-monitor-dispersion.csv", dispersion.points)}
            className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Export Data</button>
          <button type="button" title="Save the dispersion plot as a scalable SVG image"
            onClick={() => downloadSvg("launch-monitor-dispersion.svg", "launch-monitor-dispersion-plot")}
            className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Save Plot</button>
        </div>}
      </div>
      {dispersion ? <>
        <div className="my-3 grid gap-2 sm:grid-cols-4 text-sm">
          <p title="Arithmetic mean of signed lateral yards; negative indicates a left bias.">Mean: {finite(dispersion.meanLateralYards)} yd</p>
          <p title="Sample standard deviation around the mean lateral result.">SD: {finite(dispersion.standardDeviationYards)} yd</p>
          <p title="Root mean square lateral error from the target line.">RMS: {finite(dispersion.rmsYards)} yd</p>
          <p title="Counts are based on the sign of each retained lateral result.">L / C / R: {dispersion.leftCount} / {dispersion.centerCount} / {dispersion.rightCount}</p>
        </div>
        <DispersionPlot summary={dispersion} />
      </> : <p className="mt-3 text-sm text-amber-200">No recognized lateral column. Import a field such as lateral, offline, carry_side, or observed_lateral_m.</p>}
    </div>

    <div className={card}>
      <h3 className="font-semibold text-slate-200">Strokes Gained Ball-Striking Proxy</h3>
      <p className="mt-1 text-xs text-slate-400">
        Broadie-style formula: SG = E(strokes before) − 1 − E(strokes after). Remaining distance is
        √((target − carry)² + lateral²). The bundled fairway reference is an explicit internal
        interpolation table, not an official PGA TOUR benchmark. It excludes lie, recovery, hazards,
        wind, elevation, and putting context. <a className="underline" target="_blank" rel="noreferrer"
          href="https://doi.org/10.1287/inte.1120.0626">Method source</a>
      </p>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="text-sm text-slate-300">Target Distance (yd)
          <input type="number" min="1" step="1" value={targetDistanceYards}
            title="Set target distance in yards for the transparent strokes-gained proxy"
            aria-label="Strokes gained target distance in yards"
            onChange={(event) => setTargetDistanceYards(Number(event.target.value))}
            className="ml-2 rounded border border-slate-700 bg-slate-950 px-2 py-2" />
        </label>
        <p title="Mean of the shot-level proxy values shown in the exported backing data."
          className="text-lg font-semibold text-emerald-300">Mean proxy: {finite(meanGained)}</p>
        <button type="button" title="Download shot-level inputs, remaining distances, benchmark values, and proxy results"
          disabled={!gained.length} onClick={() => downloadCsv("launch-monitor-strokes-gained-proxy.csv", gained)}
          className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800 disabled:opacity-40">Export Backing Data</button>
        <button type="button" title="Download the exact interpolation reference points used by this calculation"
          onClick={() => downloadCsv("strokes-gained-reference.csv", [...STROKES_GAINED_REFERENCE])}
          className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Export Reference</button>
      </div>
      {!gained.length && <p className="mt-3 text-sm text-amber-200">A recognized carry and lateral column are required.</p>}
    </div>

    <div className={card}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold text-slate-200">Session Trend</h3>
          <p className="mt-1 text-xs text-slate-400">
            Each point is the arithmetic mean of {metricLabel(outcome)} by session_id. Session IDs are
            sorted lexicographically within each player; every player receives an independent OLS slope.
            Players are never pooled, and trends describe association, not causation.
          </p>
        </div>
        {trend && <div className="flex gap-2">
          <button type="button" title="Download session identifiers, counts, ordering, and means as CSV"
            onClick={() => downloadCsv("launch-monitor-session-trend.csv", sessionTrendExportRows(trend))}
            className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Export Data</button>
          <button type="button" title="Save the session trend plot as a scalable SVG image"
            onClick={() => downloadSvg("launch-monitor-session-trend.svg", "launch-monitor-session-plot")}
            className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Save Plot</button>
        </div>}
      </div>
      {trend ? <>
        <div className="my-3 grid gap-2 sm:grid-cols-2">
          {trend.players.map((player) => <p key={player.playerId} className="text-sm"
            title="Player-specific OLS slope of session means against this player's zero-based sorted session order.">
            {player.playerId}: slope {finite(player.slopePerSession)} {trend.unit}/session · change {finite(player.changeFirstToLast)} {trend.unit}
          </p>)}
        </div>
        <SessionTrendPlot trend={trend} />
      </> : <p className="mt-3 text-sm text-amber-200">No populated outcome values are available.</p>}
    </div>

    <details className={card}>
      <summary className="cursor-pointer font-semibold text-sky-200" title="Show formulas, assumptions, limitations, and export lineage">Calculation Guide</summary>
      <div className="mt-3 space-y-3 text-sm text-slate-300">
        <p><strong>Dispersion mean:</strong> Σ lateralᵢ / n. <strong>Sample SD:</strong> √(Σ(lateralᵢ − mean)²/(n−1)). <strong>RMS:</strong> √(Σ lateralᵢ²/n).</p>
        <p><strong>Session slope:</strong> Σ(xᵢ−x̄)(ȳᵢ−ȳ)/(Σ(xᵢ−x̄)²), where x is sorted session order and ȳ is the session mean.</p>
        <p><strong>Units:</strong> column names are mapped conservatively; meters are converted using 1 m = 1.0936133 yd. Unknown metrics remain unitless. Exported files retain the values used by each view.</p>
        <p><strong>Interpretation:</strong> results are descriptive. Measurement conventions, sample selection, omitted variables, and repeated shots can explain apparent changes.</p>
      </div>
    </details>
  </div>;
}
