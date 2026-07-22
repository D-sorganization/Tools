import React, { useEffect, useRef, useState } from "react";

import { getHistogram, getTrendline } from "../../api/explorer";
import {
  TRENDLINE_KINDS,
  type DatasetResponse,
  type TrendlineKind,
} from "../../api/explorerSchemas";
import { colorForIndex } from "../../lib/explorer/palette";
import {
  downloadBlob,
  serializeSvg,
  svgToPngBlob,
} from "../../lib/explorer/download";
import { Histogram } from "./plots/Histogram";
import { LinePlot, type LineSeries } from "./plots/LinePlot";
import { ScatterPlot } from "./plots/ScatterPlot";
import {
  columnNames,
  columnValues,
  linePoints,
  scatterPoints,
  type NotifyFn,
  type PlotConfig,
} from "./explorerState";
import { Btn, Check, Field, NumInput, Row, Select } from "./ui";

/**
 * Visualize the built dataset as a line / scatter / histogram chart with axis
 * styling, an optional fitted trendline overlay (scatter), and PNG/SVG export.
 * Correlation, spectral and PCA each have their own dedicated panels.
 */

export interface PlotPanelProps {
  dataset: DatasetResponse;
  config: PlotConfig;
  onConfigChange: (c: PlotConfig) => void;
  triggerNotification: NotifyFn;
}

const W = 720;
const H = 340;

export const PlotPanel: React.FC<PlotPanelProps> = ({
  dataset,
  config,
  onConfigChange,
  triggerNotification,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const names = columnNames(dataset);
  const set = (patch: Partial<PlotConfig>) =>
    onConfigChange({ ...config, ...patch });

  // --- Histogram data (fetched server-side for correct binning) -------------
  const [hist, setHist] = useState<{ edges: number[]; counts: number[] } | null>(
    null,
  );
  useEffect(() => {
    if (config.kind !== "histogram" || !config.histColumn) return;
    let live = true;
    getHistogram({
      values: columnValues(dataset, config.histColumn),
      bins: config.bins,
    })
      .then((r) => live && setHist({ edges: r.bin_edges, counts: r.counts }))
      .catch(() => live && setHist(null));
    return () => {
      live = false;
    };
  }, [dataset, config.kind, config.histColumn, config.bins]);

  // --- Trendline overlay (scatter) ------------------------------------------
  const [trend, setTrend] = useState<[number, number][] | null>(null);
  const [trendEq, setTrendEq] = useState<string>("");
  useEffect(() => {
    if (
      config.kind !== "scatter" ||
      config.trendline === "none" ||
      !config.xColumn ||
      !config.yColumn
    ) {
      setTrend(null);
      setTrendEq("");
      return;
    }
    let live = true;
    getTrendline({
      x: columnValues(dataset, config.xColumn),
      y: columnValues(dataset, config.yColumn),
      kind: config.trendline,
      degree: config.degree,
    })
      .then((r) => {
        if (!live) return;
        const pts: [number, number][] = r.x_fit.map((x, i) => [x, r.y_fit[i]]);
        setTrend(pts);
        setTrendEq(`${r.equation}  (R²=${r.r_squared.toFixed(4)})`);
      })
      .catch((err) => {
        if (!live) return;
        setTrend(null);
        setTrendEq("");
        triggerNotification(
          `Trendline failed: ${err instanceof Error ? err.message : err}`,
          "error",
        );
      });
    return () => {
      live = false;
    };
    // triggerNotification is intentionally excluded: it is re-created each
    // render, and including it would re-fetch the trendline on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, config.kind, config.trendline, config.degree, config.xColumn, config.yColumn]);

  const exportPlot = async (kind: "png" | "svg") => {
    const svg = svgRef.current;
    if (!svg) return;
    try {
      const blob =
        kind === "svg" ? serializeSvg(svg) : await svgToPngBlob(svg);
      downloadBlob(blob, `explorer-plot.${kind}`);
    } catch (err) {
      triggerNotification(
        `Export failed: ${err instanceof Error ? err.message : err}`,
        "error",
      );
    }
  };

  const lineSeries: LineSeries[] = config.columns.map((name, i) => ({
    name,
    color: colorForIndex(i),
    points: linePoints(dataset.index, columnValues(dataset, name)),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem" }}>
      <Row>
        {(["line", "scatter", "histogram"] as const).map((k) => (
          <Btn
            key={k}
            variant={config.kind === k ? "primary" : "ghost"}
            onClick={() => set({ kind: k })}
          >
            {k}
          </Btn>
        ))}
        <div style={{ flex: 1 }} />
        <Btn onClick={() => void exportPlot("png")}>Export PNG</Btn>
        <Btn onClick={() => void exportPlot("svg")}>Export SVG</Btn>
      </Row>

      {/* Per-kind controls */}
      {config.kind === "line" && (
        <Row>
          <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>
            Series:
          </span>
          {names.map((n) => (
            <Check
              key={n}
              label={n}
              checked={config.columns.includes(n)}
              onChange={(on) =>
                set({
                  columns: on
                    ? [...config.columns, n]
                    : config.columns.filter((c) => c !== n),
                })
              }
            />
          ))}
        </Row>
      )}
      {config.kind === "scatter" && (
        <Row>
          <Field label="x">
            <Select value={config.xColumn} onChange={(e) => set({ xColumn: e.target.value })}>
              {names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="y">
            <Select value={config.yColumn} onChange={(e) => set({ yColumn: e.target.value })}>
              {names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="trendline">
            <Select
              value={config.trendline}
              onChange={(e) =>
                set({ trendline: e.target.value as TrendlineKind | "none" })
              }
            >
              <option value="none">none</option>
              {TRENDLINE_KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </Select>
          </Field>
          {config.trendline === "polynomial" && (
            <Field label="degree">
              <NumInput
                min={1}
                max={10}
                value={config.degree}
                onChange={(e) => set({ degree: Number(e.target.value) || 2 })}
                style={{ width: "4.5rem" }}
              />
            </Field>
          )}
        </Row>
      )}
      {config.kind === "histogram" && (
        <Row>
          <Field label="column">
            <Select
              value={config.histColumn}
              onChange={(e) => set({ histColumn: e.target.value })}
            >
              {names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="bins">
            <NumInput
              min={1}
              max={500}
              value={config.bins}
              onChange={(e) => set({ bins: Number(e.target.value) || 30 })}
              style={{ width: "5rem" }}
            />
          </Field>
        </Row>
      )}

      {/* Axis style (line + scatter) */}
      {config.kind !== "histogram" && (
        <Row>
          <Check label="log X" checked={config.logX} onChange={(v) => set({ logX: v })} />
          <Check label="log Y" checked={config.logY} onChange={(v) => set({ logY: v })} />
          <Check label="grid" checked={config.grid} onChange={(v) => set({ grid: v })} />
          <Check label="legend" checked={config.legend} onChange={(v) => set({ legend: v })} />
        </Row>
      )}

      {trendEq && (
        <span style={{ fontSize: "0.72rem", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>
          {trendEq}
        </span>
      )}

      {/* The chart */}
      <div style={{ overflowX: "auto" }}>
        {config.kind === "line" && (
          <LinePlot
            ref={svgRef}
            width={W}
            height={H}
            series={lineSeries}
            xLabel="t (s)"
            yLabel="value"
            logX={config.logX}
            logY={config.logY}
            grid={config.grid}
            legend={config.legend}
          />
        )}
        {config.kind === "scatter" && (
          <ScatterPlot
            ref={svgRef}
            width={W}
            height={H}
            series={[
              {
                name: `${config.yColumn} vs ${config.xColumn}`,
                color: colorForIndex(0),
                points: scatterPoints(
                  columnValues(dataset, config.xColumn),
                  columnValues(dataset, config.yColumn),
                ),
              },
            ]}
            trendline={trend ? { points: trend, color: "var(--accent-magenta)" } : undefined}
            xLabel={config.xColumn}
            yLabel={config.yColumn}
            logX={config.logX}
            logY={config.logY}
            grid={config.grid}
            legend={config.legend}
          />
        )}
        {config.kind === "histogram" && hist && (
          <Histogram
            ref={svgRef}
            width={W}
            height={H}
            binEdges={hist.edges}
            counts={hist.counts}
            color={colorForIndex(2)}
            xLabel={config.histColumn}
            yLabel="count"
          />
        )}
        {config.kind === "histogram" && !hist && (
          <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
            Select a column to histogram.
          </span>
        )}
      </div>
    </div>
  );
};
