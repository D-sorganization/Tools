import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import {
  type CovariationResult,
  type CovariationUiSettings,
  type PairRanking,
} from "../model/launchMonitorCovariation";
import { downloadCsv, downloadJson, downloadSvg } from "../model/launchMonitorDownloads";
import { metricLabel } from "../model/launchMonitorMetricUnits";
import { LaunchMonitorCovariationChart } from "./LaunchMonitorCovariationChart";
import {
  CovariationGuide,
  LaunchMonitorCovariationControls,
} from "./LaunchMonitorCovariationControls";
import { useCovariationController } from "./useCovariationController";

const finite = (value: number | null, digits = 3) => value === null ? "—" : value.toFixed(digits);

const summaryRows = (result: CovariationResult) => [
  { label: "Pooled raw", ...result.pooledRaw },
  { label: "Within-player centered", ...result.withinPlayerCentered },
  { label: "Between-player means", ...result.betweenPlayer },
];

const exportRows = (result: CovariationResult) => result.perPlayer.map((player) => ({
  xColumn: result.request.xColumn, yColumn: result.request.yColumn,
  confidenceLevel: result.request.confidenceLevel, minSamples: result.request.minSamples,
  ...player,
}));

export function LaunchMonitorCovariation(props: {
  rows: LaunchMonitorRow[];
  savedSettings?: CovariationUiSettings;
  onSettingsChange?: (settings: CovariationUiSettings) => void;
  lockedPlayerColumn?: string;
}) {
  return <CovariationWorkspace {...props} />;
}

function CovariationWorkspace({ rows, savedSettings, onSettingsChange, lockedPlayerColumn }: {
  rows: LaunchMonitorRow[];
  savedSettings?: CovariationUiSettings;
  onSettingsChange?: (settings: CovariationUiSettings) => void;
  lockedPlayerColumn?: string;
}) {
  const controller = useCovariationController(rows, {
    savedSettings, onSettingsChange, lockedPlayerColumn,
  });
  const { settings, result, ranking, error } = controller;
  const coefficient = (summary: { pearsonR: number | null; spearmanR: number | null }) =>
    settings.method === "pearson" ? summary.pearsonR : summary.spearmanR;

  return <section aria-label="Within-player covariation" className="space-y-4">
    <div>
      <h3 className="font-semibold text-slate-200">Within-Player and Population Covariation</h3>
      <p className="mt-1 text-xs text-slate-400">
        Compare arbitrary metrics within each player, after player-mean centering, in raw pooled shots,
        and between player means. This separates repeatable player-level tendencies from population composition.
      </p>
    </div>
    <LaunchMonitorCovariationControls controller={controller} identityLocked={Boolean(lockedPlayerColumn)} />
    {error && <p role="alert" className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200">{error}</p>}
    {result && <CovariationResults rows={rows} settings={settings} result={result}
      coefficient={coefficient} />}
    {ranking && <RankingTable rows={ranking} />}
    <CovariationGuide />
  </section>;
}

function CovariationResults({ rows, settings, result, coefficient }: {
  rows: LaunchMonitorRow[]; settings: CovariationUiSettings; result: CovariationResult;
  coefficient: (summary: { pearsonR: number | null; spearmanR: number | null }) => number | null;
}) {
  return <div className="space-y-4">
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {summaryRows(result).map((summary) => <div key={summary.label} className="rounded border border-slate-700 p-3">
        <p className="text-xs text-slate-400">{summary.label}</p>
        <p className="text-xl font-semibold">{finite(coefficient(summary))}</p>
        <p className="text-xs text-slate-500">N={summary.sampleCount} · players={summary.groupCount}</p>
      </div>)}
      <div className="rounded border border-slate-700 p-3">
        <p className="text-xs text-slate-400">Random-effects meta-analysis</p>
        <p className="text-xl font-semibold">r={finite(result.meta.randomEffectR)}</p>
        <p className="text-xs text-slate-500">I²={finite(result.meta.iSquaredPct)}% · players={result.meta.contributorCount}</p>
      </div>
    </div>
    <LaunchMonitorCovariationChart rows={rows} xColumn={settings.xColumn} yColumn={settings.yColumn}
      playerColumn={settings.playerColumn} selectedPlayer={settings.selectedPlayer} />
    <p className="text-xs text-slate-500">Axes retain source units: {metricLabel(settings.xColumn)} and {metricLabel(settings.yColumn)}. Values are centered within player, so zero is that player’s own mean.</p>
    <CovariationExportButtons result={result} />
    <PlayerAssociationTable result={result} settings={settings} />
    {result.warnings.map((warning) => <p key={warning} className="text-sm text-amber-200">{warning}</p>)}
  </div>;
}

function CovariationExportButtons({ result }: { result: CovariationResult }) {
  return <div className="flex flex-wrap gap-2">
      <button type="button" title="Download per-player coefficients, intervals, eligibility, and meta weights as CSV"
        onClick={() => downloadCsv("launch-monitor-player-covariation.csv", exportRows(result))}
        className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Export Covariation CSV</button>
      <button type="button" title="Download source row identity, raw values, and player-centered values for every complete shot"
        onClick={() => downloadCsv("launch-monitor-player-covariation-backing.csv", result.backingData)}
        className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Export Centered Shots CSV</button>
      <button type="button" title="Download the complete request, formulas, summaries, player estimates, meta-analysis, and warnings"
        onClick={() => downloadJson("launch-monitor-player-covariation.json", result)}
        className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Export Covariation JSON</button>
      <button type="button" title="Save the player-centered scatter plot as a scalable SVG image"
        onClick={() => downloadSvg("launch-monitor-player-covariation.svg", "launch-monitor-covariation-plot")}
        className="rounded border border-slate-700 px-3 py-2 text-xs hover:bg-slate-800">Save Covariation Plot</button>
    </div>;
}

function PlayerAssociationTable({ result, settings }: {
  result: CovariationResult; settings: CovariationUiSettings;
}) {
  const players = result.perPlayer.filter(
    (player) => !settings.selectedPlayer || player.playerId === settings.selectedPlayer,
  );
  return <div className="overflow-x-auto"><table className="w-full text-left text-sm">
      <thead className="text-slate-400"><tr><th>Player</th><th>N</th><th>Pearson r</th><th>Spearman ρ</th><th>95% CI</th><th>Slope</th><th>Status</th></tr></thead>
      <tbody>{players.map((player) => <tr key={player.playerId} className="border-t border-slate-800">
        <td className="py-2">{player.playerId}</td><td>{player.sampleCount}</td>
        <td>{finite(player.pearsonR)}</td><td>{finite(player.spearmanR)}</td>
        <td>[{finite(player.ciLower)}, {finite(player.ciUpper)}]</td>
        <td>{finite(player.slope)} {metricLabel(settings.yColumn)}/{metricLabel(settings.xColumn)}</td>
        <td>{player.status}</td>
      </tr>)}</tbody>
    </table></div>
}

function RankingTable({ rows }: { rows: PairRanking[] }) {
  return <div className="overflow-x-auto rounded border border-slate-700 p-3">
    <h4 className="font-semibold text-slate-200">Ranked Population Pairs</h4>
    <p className="my-2 text-xs text-amber-200">This exploratory scan makes multiple comparisons; ranks and unadjusted effects require confirmation on held-out data.</p>
    <table className="w-full text-left text-sm"><thead className="text-slate-400"><tr>
      <th>X</th><th>Y</th><th>Random-effects r</th><th>Players</th><th>Direction consistency</th>
    </tr></thead><tbody>{rows.slice(0, 50).map((row) => <tr key={`${row.xColumn}-${row.yColumn}`} className="border-t border-slate-800">
      <td className="py-2">{row.xColumn}</td><td>{row.yColumn}</td><td>{finite(row.randomEffectR)}</td>
      <td>{row.contributorCount}</td><td>{row.directionConsistency === null ? "—" : `${(row.directionConsistency * 100).toFixed(1)}%`}</td>
    </tr>)}</tbody></table>
  </div>;
}
