import React, { useState } from "react";

import { getCorrelation, getPca, getSpectrum, getStatistics } from "../../api/explorer";
import {
  CORRELATION_METHODS,
  SPECTRUM_METHODS,
  WINDOW_KINDS,
  type ColumnStatistics,
  type CorrelationMethod,
  type CorrelationResponse,
  type DatasetResponse,
  type PcaResponse,
  type SpectrumMethod,
  type SpectrumResponse,
  type WindowKind,
} from "../../api/explorerSchemas";
import { colorForIndex } from "../../lib/explorer/palette";
import { Heatmap } from "./plots/Heatmap";
import { ScatterPlot } from "./plots/ScatterPlot";
import { SpectrumPlot } from "./plots/SpectrumPlot";
import { columnValues, type NotifyFn, type PlotConfig } from "./explorerState";
import { Btn, Check, Field, Row, Select } from "./ui";

/**
 * Read-only analysis panels over the built dataset: descriptive statistics, a
 * correlation heatmap, a power spectrum, and PCA. Each fetches its own result
 * on demand (the endpoints are idempotent and read-only) so the container stays
 * thin.
 */

interface BaseProps {
  dataset: DatasetResponse;
  config: PlotConfig;
  onConfigChange: (c: PlotConfig) => void;
  triggerNotification: NotifyFn;
}

const fmt = (v: number): string =>
  Number.isFinite(v) ? (Math.abs(v) >= 1e6 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : v.toFixed(4)) : "—";

const W = 700;
const H = 320;

// --- Statistics --------------------------------------------------------------

export const StatisticsPanel: React.FC<BaseProps> = ({ dataset, triggerNotification }) => {
  const [stats, setStats] = useState<ColumnStatistics[] | null>(null);
  const [busy, setBusy] = useState(false);
  const compute = () => {
    setBusy(true);
    getStatistics(dataset.columns)
      .then((r) => setStats(r.stats))
      .catch((err) =>
        triggerNotification(`Statistics failed: ${err instanceof Error ? err.message : err}`, "error"),
      )
      .finally(() => setBusy(false));
  };
  const cols: Exclude<keyof ColumnStatistics, "name">[] = [
    "count", "mean", "std", "min", "p25", "median", "p75", "max", "rms",
  ];
  return (
    <div>
      <Btn variant="primary" onClick={compute} disabled={busy}>
        {busy ? "Computing…" : "Compute statistics"}
      </Btn>
      {stats && (
        <div style={{ overflowX: "auto", marginTop: "0.6rem" }}>
          <table style={{ borderCollapse: "collapse", fontSize: "0.74rem", width: "100%" }}>
            <thead>
              <tr>
                <th style={thStyle}>signal</th>
                {cols.map((c) => (
                  <th key={c} style={thStyle}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stats.map((s) => (
                <tr key={s.name}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{s.name}</td>
                  {cols.map((c) => (
                    <td key={c} style={tdStyle}>{c === "count" ? s[c] : fmt(s[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const thStyle: React.CSSProperties = {
  textAlign: "right",
  padding: "0.25rem 0.5rem",
  borderBottom: "1px solid var(--panel-border)",
  color: "var(--text-secondary)",
  whiteSpace: "nowrap",
};
const tdStyle: React.CSSProperties = {
  textAlign: "right",
  padding: "0.2rem 0.5rem",
  borderBottom: "1px solid var(--cell-border)",
  fontFamily: "var(--font-mono)",
};

// --- Correlation -------------------------------------------------------------

export const CorrelationPanel: React.FC<BaseProps> = ({
  dataset, config, onConfigChange, triggerNotification,
}) => {
  const [result, setResult] = useState<CorrelationResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const compute = () => {
    setBusy(true);
    getCorrelation(dataset.columns, config.correlationMethod)
      .then(setResult)
      .catch((err) =>
        triggerNotification(`Correlation failed: ${err instanceof Error ? err.message : err}`, "error"),
      )
      .finally(() => setBusy(false));
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <Row>
        <Field label="method">
          <Select
            value={config.correlationMethod}
            onChange={(e) =>
              onConfigChange({ ...config, correlationMethod: e.target.value as CorrelationMethod })
            }
          >
            {CORRELATION_METHODS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </Select>
        </Field>
        <Btn variant="primary" onClick={compute} disabled={busy}>
          {busy ? "Computing…" : "Compute correlation"}
        </Btn>
      </Row>
      {result && (
        <Heatmap
          width={Math.min(W, 90 + result.labels.length * 64)}
          height={Math.min(W, 90 + result.labels.length * 64)}
          labels={result.labels}
          matrix={result.matrix}
          showValues={result.labels.length <= 12}
        />
      )}
    </div>
  );
};

// --- Spectral ----------------------------------------------------------------

export const SpectralPanel: React.FC<BaseProps> = ({
  dataset, config, onConfigChange, triggerNotification,
}) => {
  const [result, setResult] = useState<SpectrumResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [logY, setLogY] = useState(true);
  const names = dataset.columns.map((c) => c.name);
  const compute = () => {
    if (!config.spectrumColumn) return;
    setBusy(true);
    const rate = dataset.sample_rate_hz && dataset.sample_rate_hz > 0 ? dataset.sample_rate_hz : 1;
    getSpectrum({
      values: columnValues(dataset, config.spectrumColumn),
      sample_rate_hz: rate,
      method: config.spectrumMethod,
      window: config.window,
      detrend: true,
    })
      .then(setResult)
      .catch((err) =>
        triggerNotification(`Spectrum failed: ${err instanceof Error ? err.message : err}`, "error"),
      )
      .finally(() => setBusy(false));
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <Row>
        <Field label="column">
          <Select
            value={config.spectrumColumn}
            onChange={(e) => onConfigChange({ ...config, spectrumColumn: e.target.value })}
          >
            {names.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </Select>
        </Field>
        <Field label="method">
          <Select
            value={config.spectrumMethod}
            onChange={(e) => onConfigChange({ ...config, spectrumMethod: e.target.value as SpectrumMethod })}
          >
            {SPECTRUM_METHODS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </Select>
        </Field>
        <Field label="window">
          <Select
            value={config.window}
            onChange={(e) => onConfigChange({ ...config, window: e.target.value as WindowKind })}
          >
            {WINDOW_KINDS.map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </Select>
        </Field>
        <Check label="log power" checked={logY} onChange={setLogY} />
        <Btn variant="primary" onClick={compute} disabled={busy || !config.spectrumColumn}>
          {busy ? "Computing…" : "Compute spectrum"}
        </Btn>
      </Row>
      <span style={{ fontSize: "0.66rem", color: "var(--text-muted)" }}>
        Sample rate inferred from the dataset index: {dataset.sample_rate_hz?.toFixed(3) ?? "—"} Hz
      </span>
      {result && (
        <SpectrumPlot
          width={W}
          height={H}
          freqs={result.freqs}
          power={result.power}
          logY={logY}
          xLabel="frequency (Hz)"
          yLabel="power"
        />
      )}
    </div>
  );
};

// --- PCA ---------------------------------------------------------------------

export const PcaPanel: React.FC<BaseProps> = ({
  dataset, config, onConfigChange, triggerNotification,
}) => {
  const [result, setResult] = useState<PcaResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const compute = () => {
    setBusy(true);
    getPca({ columns: dataset.columns, standardize: config.standardize })
      .then(setResult)
      .catch((err) =>
        triggerNotification(`PCA failed: ${err instanceof Error ? err.message : err}`, "error"),
      )
      .finally(() => setBusy(false));
  };
  const scores: [number, number][] =
    result && result.scores_pc1.length
      ? result.scores_pc1.map((x, i) => [x, result.scores_pc2[i] ?? 0])
      : [];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <Row>
        <Check
          label="standardize"
          checked={config.standardize}
          onChange={(v) => onConfigChange({ ...config, standardize: v })}
        />
        <Btn variant="primary" onClick={compute} disabled={busy}>
          {busy ? "Computing…" : "Compute PCA"}
        </Btn>
      </Row>
      {result && (
        <>
          <div style={{ fontSize: "0.74rem" }}>
            <span style={{ color: "var(--text-secondary)" }}>Explained variance:</span>{" "}
            {result.explained_variance_ratio.map((v, i) => (
              <span key={i} style={{ fontFamily: "var(--font-mono)", marginRight: "0.6rem" }}>
                PC{i + 1}={(v * 100).toFixed(1)}%
              </span>
            ))}
          </div>
          {/* Variance bars */}
          <div style={{ display: "flex", alignItems: "flex-end", gap: "0.3rem", height: "60px" }}>
            {result.explained_variance_ratio.slice(0, 12).map((v, i) => (
              <div
                key={i}
                title={`PC${i + 1}: ${(v * 100).toFixed(1)}%`}
                style={{
                  width: "1.5rem",
                  height: `${Math.max(2, v * 60)}px`,
                  background: colorForIndex(i),
                  borderRadius: "2px 2px 0 0",
                }}
              />
            ))}
          </div>
          {scores.length > 0 && (
            <ScatterPlot
              width={W}
              height={H}
              series={[{ name: "scores", color: colorForIndex(0), points: scores, size: 2.5 }]}
              xLabel="PC1"
              yLabel="PC2"
              grid
            />
          )}
        </>
      )}
    </div>
  );
};
