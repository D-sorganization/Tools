import { useMemo, useState } from "react";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import {
  analyzePlayerCovariation,
  defaultCovariationSettings,
  rankCovariationPairs,
  type CovariationResult,
  type CovariationUiSettings,
  type PairRanking,
} from "../model/launchMonitorCovariation";

export interface CovariationController {
  settings: CovariationUiSettings;
  numeric: string[];
  groupingColumns: string[];
  players: string[];
  result: CovariationResult | null;
  ranking: PairRanking[] | null;
  error: string | null;
  update: (change: Partial<CovariationUiSettings>) => void;
  analyze: () => void;
  rankPairs: () => void;
}

const numericColumns = (rows: LaunchMonitorRow[]) => Object.keys(rows[0] ?? {}).filter(
  (column) => rows.filter((row) => Number.isFinite(Number(row[column]))).length >= 3,
).sort();

const playerValues = (rows: LaunchMonitorRow[], column: string) => [...new Set(rows.map(
  (row) => String(row[column] ?? "").trim(),
))].filter(Boolean).sort();

const errorText = (caught: unknown) => caught instanceof Error ? caught.message : String(caught);

export function useCovariationController(rows: LaunchMonitorRow[], options: {
  savedSettings?: CovariationUiSettings;
  onSettingsChange?: (settings: CovariationUiSettings) => void;
  lockedPlayerColumn?: string;
}): CovariationController {
  const [local, setLocal] = useState(() => ({
    ...(options.savedSettings ?? defaultCovariationSettings(rows)),
    ...(options.lockedPlayerColumn ? { playerColumn: options.lockedPlayerColumn } : {}),
  }));
  const [result, setResult] = useState<CovariationResult | null>(null);
  const [ranking, setRanking] = useState<PairRanking[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const settings = {
    ...local,
    ...(options.lockedPlayerColumn ? { playerColumn: options.lockedPlayerColumn } : {}),
  };
  const numeric = useMemo(() => numericColumns(rows), [rows]);
  const groupingColumns = useMemo(() => Object.keys(rows[0] ?? {}).sort(), [rows]);
  const players = useMemo(() => playerValues(rows, settings.playerColumn), [rows, settings.playerColumn]);
  const update = (change: Partial<CovariationUiSettings>) => {
    const next = { ...settings, ...change };
    setLocal(next); options.onSettingsChange?.(next);
    setResult(null); setRanking(null); setError(null);
  };
  const analyze = () => {
    try {
      setResult(analyzePlayerCovariation(rows, {
        xColumn: settings.xColumn, yColumn: settings.yColumn,
        playerColumn: settings.playerColumn, minSamples: settings.minSamples,
        confidenceLevel: settings.confidenceLevel,
      })); setError(null);
    } catch (caught) { setError(errorText(caught)); }
  };
  const rankPairs = () => {
    try {
      setRanking(rankCovariationPairs(rows, {
        columns: numeric, playerColumn: settings.playerColumn,
        minSamples: settings.minSamples, confidenceLevel: settings.confidenceLevel,
      })); setError(null);
    } catch (caught) { setError(errorText(caught)); }
  };
  return { settings, numeric, groupingColumns, players, result, ranking, error, update, analyze, rankPairs };
}
