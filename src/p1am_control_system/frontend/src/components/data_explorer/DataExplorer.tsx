import React, { useCallback, useEffect, useMemo, useState } from "react";

import { buildDataset, getSignals } from "../../api/explorer";
import type { DatasetRequest, DatasetResponse, SignalInfo } from "../../api/explorerSchemas";
import { ApiError } from "../../api/client";
import { CollapsibleSection } from "../CollapsibleSection";
import {
  CorrelationPanel,
  PcaPanel,
  SpectralPanel,
  StatisticsPanel,
} from "./AnalysisPanels";
import { ExportSessionPanel } from "./ExportSessionPanel";
import { PipelinePanel } from "./PipelinePanel";
import { PlotPanel } from "./PlotPanel";
import { SourcePanel } from "./SourcePanel";
import {
  columnNames,
  defaultPipeline,
  defaultPlotConfig,
  reconcilePlotColumns,
  type CsvSource,
  type ExplorerSession,
  type HistorianForm,
  type NotifyFn,
  type Pipeline,
  type PlotConfig,
  type SourceMode,
} from "./explorerState";

/**
 * The Data Explorer tab: a flexible, web-native reimplementation of the desktop
 * Data Processor. Pull tag history (or a browser-parsed CSV) into a dataset,
 * run a filter/transform/derived-column pipeline, then visualize, correlate,
 * spectrally analyze, run PCA, and export plots + datasets.
 *
 * This container owns all state and the historian/dataset API calls; the panels
 * are controlled views. Read-only analysis panels fetch their own (idempotent)
 * results on demand.
 */

export interface DataExplorerProps {
  triggerNotification: NotifyFn;
}

const cardWrap: React.CSSProperties = {
  background: "var(--panel-bg)",
  border: "1px solid var(--panel-border)",
  borderRadius: "8px",
  padding: "0.85rem 1rem",
  boxShadow: "var(--card-shadow)",
};

/** Local datetime-input string -> UTC ISO so it matches UTC-stored history. */
function toIso(local: string): string {
  const ms = Date.parse(local);
  if (!Number.isFinite(ms)) throw new Error(`invalid date/time: "${local}"`);
  return new Date(ms).toISOString();
}

function initialHistorian(): HistorianForm {
  const now = Date.now();
  const pad = (n: number) => String(n).padStart(2, "0");
  const fmt = (ms: number) => {
    const d = new Date(ms);
    return (
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
      `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
    );
  };
  return { tags: [], start: fmt(now - 3600_000), end: fmt(now), maxPoints: 5000 };
}

export const DataExplorer: React.FC<DataExplorerProps> = ({ triggerNotification }) => {
  const [signals, setSignals] = useState<SignalInfo[]>([]);
  const [mode, setMode] = useState<SourceMode>("historian");
  const [historian, setHistorian] = useState<HistorianForm>(initialHistorian);
  const [csv, setCsv] = useState<CsvSource | null>(null);
  const [pipeline, setPipeline] = useState<Pipeline>(defaultPipeline);
  const [plot, setPlot] = useState<PlotConfig>(defaultPlotConfig);
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSignals()
      .then((r) => setSignals(r.signals))
      .catch(() => setSignals([]));
  }, []);

  // Column names available for pipeline targets / plotting — from the built
  // dataset if present, else from the configured source.
  const availableColumns = useMemo(() => {
    if (dataset) return columnNames(dataset);
    if (mode === "historian") return historian.tags;
    return csv ? csv.columns.map((c) => c.name) : [];
  }, [dataset, mode, historian.tags, csv]);

  const handleCsvLoaded = (loaded: CsvSource | null) => {
    setCsv(loaded);
    setError(null);
  };

  const onBuild = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let req: DatasetRequest;
      if (mode === "historian") {
        req = {
          historian: {
            tags: historian.tags,
            start_time: toIso(historian.start),
            end_time: toIso(historian.end),
            max_points: historian.maxPoints,
          },
          resample: pipeline.resample ?? undefined,
          filters: pipeline.filters,
          derived: pipeline.derived,
          trim: pipeline.trim ?? undefined,
          max_points: historian.maxPoints,
        };
      } else {
        if (!csv) throw new Error("No CSV loaded");
        const n = csv.columns[0]?.values.length ?? 0;
        let index = csv.index;
        if (!index) {
          // ⚡ Bolt Optimization: Pre-allocate array and populate with a standard for loop to avoid Array.from overhead
          index = new Array(n);
          for (let i = 0; i < n; i++) {
            index[i] = i;
          }
        }
        req = {
          inline: { index, columns: csv.columns },
          resample: pipeline.resample ?? undefined,
          filters: pipeline.filters,
          derived: pipeline.derived,
          trim: pipeline.trim ?? undefined,
          max_points: historian.maxPoints,
        };
      }
      const result = await buildDataset(req);
      setDataset(result);
      setPlot((p) => {
        const reconciled = reconcilePlotColumns(p, columnNames(result));
        return reconciled.columns.length
          ? reconciled
          : { ...reconciled, columns: columnNames(result).slice(0, 1) };
      });
      if (result.row_count === 0) {
        triggerNotification("Dataset built but contains no rows for that range", "info");
      } else {
        triggerNotification(
          `Dataset built: ${result.columns.length} columns × ${result.row_count} rows`,
          "success",
        );
      }
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? String(err.message)
          : err instanceof Error
            ? err.message
            : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [mode, historian, csv, pipeline, triggerNotification]);

  const getSession = useCallback(
    (): ExplorerSession => ({ sourceMode: mode, historian, pipeline, plot }),
    [mode, historian, pipeline, plot],
  );
  const applySession = (s: ExplorerSession) => {
    setMode(s.sourceMode);
    setHistorian(s.historian);
    setPipeline(s.pipeline);
    setPlot(s.plot);
    triggerNotification("Session loaded — rebuild to apply", "info");
  };

  const analysisProps = dataset
    ? { dataset, config: plot, onConfigChange: setPlot, triggerNotification }
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={cardWrap}>
        <CollapsibleSection title="1 · Data Source" defaultOpen>
          <SourcePanel
            signals={signals}
            mode={mode}
            onModeChange={setMode}
            historian={historian}
            onHistorianChange={setHistorian}
            csv={csv}
            onCsvLoaded={handleCsvLoaded}
            onError={setError}
            onBuild={() => void onBuild()}
            loading={loading}
            error={error}
          />
        </CollapsibleSection>
      </div>

      <div style={cardWrap}>
        <CollapsibleSection title="2 · Pipeline (resample · filters · derived)" defaultOpen={false}>
          <PipelinePanel
            columns={availableColumns}
            pipeline={pipeline}
            onChange={setPipeline}
          />
        </CollapsibleSection>
      </div>

      {dataset ? (
        <>
          <div style={cardWrap}>
            <CollapsibleSection title="3 · Visualize" defaultOpen>
              <PlotPanel
                dataset={dataset}
                config={plot}
                onConfigChange={setPlot}
                triggerNotification={triggerNotification}
              />
            </CollapsibleSection>
          </div>

          <div style={cardWrap}>
            <CollapsibleSection title="Statistics" defaultOpen={false}>
              <StatisticsPanel {...analysisProps!} />
            </CollapsibleSection>
          </div>
          <div style={cardWrap}>
            <CollapsibleSection title="Correlation matrix" defaultOpen={false}>
              <CorrelationPanel {...analysisProps!} />
            </CollapsibleSection>
          </div>
          <div style={cardWrap}>
            <CollapsibleSection title="Spectral analysis" defaultOpen={false}>
              <SpectralPanel {...analysisProps!} />
            </CollapsibleSection>
          </div>
          <div style={cardWrap}>
            <CollapsibleSection title="PCA" defaultOpen={false}>
              <PcaPanel {...analysisProps!} />
            </CollapsibleSection>
          </div>
        </>
      ) : (
        <div style={{ ...cardWrap, color: "var(--text-muted)", fontSize: "0.82rem" }}>
          Build a dataset to visualize, correlate, run a spectrum/PCA, and export.
        </div>
      )}

      <div style={cardWrap}>
        <CollapsibleSection title="Export &amp; sessions" defaultOpen={false}>
          <ExportSessionPanel
            dataset={dataset}
            getSession={getSession}
            applySession={applySession}
            triggerNotification={triggerNotification}
          />
        </CollapsibleSection>
      </div>
    </div>
  );
};
