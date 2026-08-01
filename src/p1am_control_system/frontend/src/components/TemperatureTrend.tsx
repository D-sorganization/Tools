import React, { useState, useRef } from "react";
import {
  type AxisRange,
  defaultAxisRange,
  resolveRange,
  axisTicks,
} from "../lib/trendAxis";
import {
  MAX_WINDOW_SECONDS,
  TREND_BACKFILL_MAX_POINTS,
  downsample,
  formatWindow,
  formatClock,
  timeSeriesPath,
  timeToX,
} from "../lib/trendTime";
import { nearestIndexByX } from "../lib/plotCursor";
import { fitSeries, NO_FIT_ID, pointsInLastWindow, type FitPoint } from "../lib/curveFit";
import {
  type TempSample,
  activeSpans,
  formatTcReadout,
  heatUpRateReadout,
  plotPxToTime,
  tcSampleValue,
  timedSegments,
} from "../lib/temperatureTrend";
import { useTrendBackfill } from "../hooks/useTrendBackfill";
import { useTrendViewport } from "../hooks/useTrendViewport";
import { useNonPassiveWheel } from "../hooks/useNonPassiveWheel";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendTimeControls } from "./TrendTimeControls";
import { TrendFitControls } from "./TrendFitControls";
import {
  TrendTimeAxis,
  TrendFitOverlay,
  TrendCrosshair,
  type CrosshairSeries,
} from "./TrendPlotOverlays";
import { SnapshotButton } from "./SnapshotButton";
import type { TcType } from "../types";

/**
 * Compact single-trace temperature trend for the controller screen: measured
 * temperature plotted against a 0–full-scale °C Y axis, with the setpoint and
 * high-high cutoff drawn as horizontal reference lines.
 *
 * Self-contained SVG (no chart lib): the parent accumulates a rolling sample
 * buffer from the live status and passes it in. Split out of
 * TemperatureControl.tsx verbatim — the parent still owns the buffer and every
 * REST call; this file owns only the plot's own view state (axis, fit, viewport,
 * hover), exactly as it did while nested.
 */
const TREND_W = 600;
const TREND_H = 188;
const TREND_PAD_L = 40;
const TREND_PAD_R = 10;
const TREND_PAD_T = 10;
const TREND_PAD_B = 26; // room for the X-axis time labels
const TREND_PLOT_H = TREND_H - TREND_PAD_T - TREND_PAD_B;

const DEFAULT_WINDOW_SECONDS = 3600; // 60 minutes

const K_COLOR = "var(--color-error)"; // Type-K trace (also the historian backfill)
const R_COLOR = "var(--accent-magenta)"; // Type-R trace
const SETPOINT_COLOR = "var(--accent-cyan)";
const HH_COLOR = "var(--color-warning)";
const RELAY_BAND_COLOR = "var(--color-error)";

/** Default look-back (minutes) for the heat-up-rate regression. */
const DEFAULT_FIT_WINDOW_MIN = 5;

interface TrendProps {
  samples: TempSample[];
  /** Tag index backing the temperature (for historian backfill). */
  tagId: number;
  fullScale: number;
  setpoint: number;
  hhLimit: number;
  /** Which thermocouple is controlling — emphasized and used for the fit. */
  activeTcType: TcType;
}

export const TempTrend: React.FC<TrendProps> = ({
  samples,
  tagId,
  fullScale,
  setpoint,
  hhLimit,
  activeTcType,
}) => {
  const [axis, setAxis] = useState<AxisRange>(defaultAxisRange(0, fullScale));
  const [fitMethodId, setFitMethodId] = useState<string>(NO_FIT_ID);
  const [fitWindowMin, setFitWindowMin] = useState<number>(DEFAULT_FIT_WINDOW_MIN);

  // Signal picker: default every trace + the heater-status band visible.
  const [showK, setShowK] = useState(true);
  const [showR, setShowR] = useState(true);
  const [showRelay, setShowRelay] = useState(true);

  // Pan / zoom / pause come from the shared, tested viewport model (DRY). The
  // domain is epoch ms; the visible window (seconds) is DERIVED from its span so
  // it still drives the historian backfill depth and the "last …" label.
  const view = useTrendViewport({
    defaultSpan: DEFAULT_WINDOW_SECONDS * 1000,
    minSpan: 1000,
    maxSpan: MAX_WINDOW_SECONDS * 1000,
  });
  const windowSeconds = view.viewport.span / 1000;

  // While paused, plot a frozen snapshot of the samples captured at pause time
  // so incoming live data never slides the view out from under the operator.
  const [frozen, setFrozen] = useState<TempSample[]>([]);
  // Plot pixel X of the hover crosshair, or null when not hovering.
  const [hoverPx, setHoverPx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Backfill from the historian so widening the window immediately shows past
  // data (stored tag is a 0–100 %, so scale it to °C). The historian carries the
  // ACTIVE channel's tag only, so merge it into that channel of the buffer
  // (anything older than the live buffer) ahead of the live samples.
  const backfill = useTrendBackfill(
    tagId,
    windowSeconds,
    fullScale / 100,
    TREND_BACKFILL_MAX_POINTS,
  );
  // Freeze swaps the live prop for the snapshot captured at pause time; either
  // way backfill merges in ahead of it, so windowing/backfill stay unchanged.
  const source = view.paused ? frozen : samples;
  const liveStart = source.length ? source[0].t : Infinity;
  const older = backfill.filter((b) => b.t < liveStart);
  const series: TempSample[] = older.length
    ? [
        ...older.map((b) => ({
          t: b.t,
          k: activeTcType === "K" ? b.v : null,
          r: activeTcType === "R" ? b.v : null,
          relayOn: false,
        })),
        ...source,
      ]
    : source;

  // The viewport decides which slice of real wall-clock time is on screen. Its
  // bounds are the full data extent; resolve() clamps the visible [t0,t1] for
  // the current span (zoom) and offset (pan), then we window the samples to it.
  const bounds = {
    min: series[0]?.t ?? 0,
    max: series[series.length - 1]?.t ?? Date.now(),
  };
  const { start: t0, end: t1 } = view.resolve(bounds);
  const windowed = series.filter((s) => s.t >= t0 && s.t <= t1);
  const plotted = downsample(windowed);

  // Plot-pixel ↔ time mapping for wheel / drag-zoom. preserveAspectRatio="none"
  // means x scales with the rendered width, so convert client px → viewBox px →
  // plot px, and (via the pure helper) plot px → time within the window.
  const plotW = TREND_W - TREND_PAD_L - TREND_PAD_R;
  const plotPx = (clientX: number): number => {
    const el = svgRef.current;
    if (!el) return 0;
    const r = el.getBoundingClientRect();
    if (!r.width) return 0;
    const x = ((clientX - r.left) / r.width) * TREND_W;
    return Math.max(0, Math.min(plotW, x - TREND_PAD_L));
  };
  const pxToUnit = (px: number): number => plotPxToTime(px, plotW, t0, t1);

  // Wheel zoom about the cursor, attached as a NON-passive native listener (see
  // useNonPassiveWheel) so preventDefault suppresses page scroll — React's
  // onWheel is passive under React 18.
  const handleWheel = (e: WheelEvent): void => {
    e.preventDefault();
    view.zoomBy(e.deltaY > 0 ? 1.15 : 0.87, pxToUnit(plotPx(e.clientX)), bounds);
  };
  useNonPassiveWheel(svgRef, handleWheel);

  // Resolve the Y range against BOTH channels so neither trace clips.
  const plottedC = plotted
    .flatMap((s) => [s.k, s.r])
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const { min, max } = resolveRange(axis, plottedC, { min: 0, max: fullScale });

  // Latest reading of each channel (for the legend), and the active reading.
  const lastSample = windowed.length ? windowed[windowed.length - 1] : undefined;
  const lastK = lastSample?.k ?? null;
  const lastR = lastSample?.r ?? null;

  const refY = (value: number): number => {
    const frac = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return TREND_PAD_T + (1 - frac) * TREND_PLOT_H;
  };

  // Position traces by real timestamp (not array index) so they are time-accurate
  // even when sparse historian backfill is merged with dense live samples. The X
  // axis spans the viewport's resolved [t0,t1] window, so zoom / pan / window
  // changes rescale the axis immediately.
  const geom = {
    t0,
    t1,
    min,
    max,
    x0: TREND_PAD_L,
    x1: TREND_W - TREND_PAD_R,
    yTop: TREND_PAD_T,
    plotH: TREND_PLOT_H,
  } as const;

  // Build each channel as gap-aware path segments (a null skips the line).
  const kSegments = timedSegments(plotted.map((s) => ({ t: s.t, v: s.k })));
  const rSegments = timedSegments(plotted.map((s) => ({ t: s.t, v: s.r })));
  const kPaths = kSegments.map((seg) => timeSeriesPath(seg, geom));
  const rPaths = rSegments.map((seg) => timeSeriesPath(seg, geom));

  // Heater-status band: shaded spans where the relay was closed.
  const relaySpans = activeSpans(plotted, (s) => s.relayOn);

  // Resolve the hover crosshair: snap to the nearest sample by time and read
  // each visible channel's °C there, placing markers with the same refY mapping
  // the traces use (DRY). Channels that are null at that sample are skipped.
  const hover: { px: number; series: CrosshairSeries[]; xLabel: string } | null =
    (() => {
      if (hoverPx === null || plotted.length < 2) return null;
      const idx = nearestIndexByX(
        plotted.map((s) => s.t),
        pxToUnit(hoverPx),
      );
      if (idx === null) return null;
      const s = plotted[idx];
      const series: CrosshairSeries[] = [];
      if (showK && typeof s.k === "number" && Number.isFinite(s.k)) {
        series.push({ label: "K", color: K_COLOR, text: `${s.k.toFixed(1)} °C`, py: refY(s.k) });
      }
      if (showR && typeof s.r === "number" && Number.isFinite(s.r)) {
        series.push({ label: "R", color: R_COLOR, text: `${s.r.toFixed(1)} °C`, py: refY(s.r) });
      }
      if (series.length === 0) return null;
      return {
        px: timeToX(s.t, t0, t1, TREND_PAD_L, TREND_W - TREND_PAD_R),
        series,
        xLabel: formatClock(s.t),
      };
    })();

  // Curve fit runs over the ACTIVE thermocouple series, restricted to the last
  // `fitWindowMin` minutes (the operator-chosen regression window). x is epoch
  // ms so the shared heat-up-rate helpers (called with msPerXUnit = 1) convert
  // the slope to °C/min and °C/hr — no rate math is duplicated here (DRY).
  const activePoints: FitPoint[] = plotted
    .map((s) => ({ x: s.t, y: tcSampleValue(s, activeTcType) }))
    .filter((p): p is FitPoint => typeof p.y === "number" && Number.isFinite(p.y));
  const windowedFitPoints = pointsInLastWindow(activePoints, fitWindowMin * 60000);
  const fit = fitSeries(windowedFitPoints, fitMethodId);
  const rateReadout = heatUpRateReadout(
    plotted,
    activeTcType,
    fitMethodId,
    fitWindowMin,
  );
  // The fit's x IS the epoch-ms timestamp, so both the X position (p.t) and the
  // value fed to fit.predict (p.x) are that same timestamp.
  const fitOverlayPoints = windowedFitPoints.map((p) => ({ t: p.x, x: p.x }));

  const activeColor = activeTcType === "K" ? K_COLOR : R_COLOR;

  // CSV export of the plotted K/R samples (empty cell for a channel gap; relay
  // as 0/1). Hidden (undefined) until the window holds at least one point.
  const snapshotCsv = plotted.length
    ? {
        headers: ["t_ms", "type_k_c", "type_r_c", "relay_on"],
        rows: plotted.map((s) => [
          s.t,
          s.k ?? "",
          s.r ?? "",
          s.relayOn ? 1 : 0,
        ]),
      }
    : undefined;

  return (
    <div className="tc-trend">
      <div className="tc-trend-legend">
        <label
          className="tc-trend-key tc-trend-toggle"
          title="Show / hide the Type-K trace"
        >
          <input
            type="checkbox"
            checked={showK}
            onChange={(e) => setShowK(e.target.checked)}
            aria-label="Show Type K trace"
          />
          <span
            className="swatch"
            style={{
              background: K_COLOR,
              opacity: activeTcType === "K" ? 1 : 0.55,
            }}
          />
          Type K{activeTcType === "K" ? " (active)" : ""}
          <strong>{formatTcReadout(lastK)}</strong>
        </label>
        <label
          className="tc-trend-key tc-trend-toggle"
          title="Show / hide the Type-R trace"
        >
          <input
            type="checkbox"
            checked={showR}
            onChange={(e) => setShowR(e.target.checked)}
            aria-label="Show Type R trace"
          />
          <span
            className="swatch"
            style={{
              background: R_COLOR,
              opacity: activeTcType === "R" ? 1 : 0.55,
            }}
          />
          Type R{activeTcType === "R" ? " (active)" : ""}
          <strong>{formatTcReadout(lastR)}</strong>
        </label>
        <label
          className="tc-trend-key tc-trend-toggle"
          title="Shade the periods when the heater relay was ON"
        >
          <input
            type="checkbox"
            checked={showRelay}
            onChange={(e) => setShowRelay(e.target.checked)}
            aria-label="Show heater ON band"
          />
          <span
            className="swatch"
            style={{ background: RELAY_BAND_COLOR, opacity: 0.25 }}
          />
          Heater ON
        </label>
        <span className="tc-trend-key">
          <span className="swatch" style={{ background: SETPOINT_COLOR }} />
          Setpoint
          <strong>{setpoint.toFixed(1)} °C</strong>
        </span>
        <span className="tc-trend-key">
          <span className="swatch" style={{ background: HH_COLOR }} />
          HH cutoff
          <strong>{hhLimit.toFixed(0)} °C</strong>
        </span>
        {!view.live && (
          <span
            className="tc-trend-key"
            style={{
              color: "var(--color-warning)",
              border: "1px solid var(--color-warning)",
              borderRadius: "3px",
              padding: "0.05rem 0.3rem",
              fontSize: "0.66rem",
              fontWeight: 600,
              letterSpacing: "0.5px",
              textTransform: "uppercase",
            }}
            title={
              view.paused
                ? "Plot frozen — live updates paused"
                : "Scrolled back in time — not following the live edge"
            }
          >
            {view.paused ? "Frozen" : "Panned"}
          </span>
        )}
        <span className="tc-trend-window">
          last {formatWindow(windowSeconds)} · {min.toFixed(0)}–{max.toFixed(0)} °C
        </span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          margin: "0.1rem 0 0.3rem",
          flexWrap: "wrap",
        }}
      >
        <div
          style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}
          role="group"
          aria-label="Trend pan, zoom and pause controls"
        >
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => {
              if (!view.paused) setFrozen(samples);
              view.togglePause();
            }}
            title={view.paused ? "Resume live streaming" : "Pause / freeze the plot"}
            aria-label={view.paused ? "Resume live" : "Pause"}
          >
            {view.paused ? "Live" : "Pause"}
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.panBy((t1 - t0) * 0.3, bounds)}
            title="Scroll back in time"
            aria-label="Pan back in time"
          >
            ◀
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.panBy(-(t1 - t0) * 0.3, bounds)}
            title="Scroll forward in time"
            aria-label="Pan forward in time"
          >
            ▶
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.zoomBy(0.7, (t0 + t1) / 2, bounds)}
            title="Zoom in (narrow the time window)"
            aria-label="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.zoomBy(1.43, (t0 + t1) / 2, bounds)}
            title="Zoom out (widen the time window)"
            aria-label="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.reset()}
            title="Reset zoom / pan and resume live"
            aria-label="Reset view"
          >
            Reset
          </button>
        </div>
        <TrendTimeControls
          value={windowSeconds}
          onChange={(seconds) => view.setSpan(seconds * 1000)}
        />
        <TrendAxisControls value={axis} onChange={setAxis} unit="°C" />
        <TrendFitControls value={fitMethodId} onChange={setFitMethodId} />
        <label
          className="tc-trend-fitwindow"
          title="How many recent minutes the linear regression covers"
        >
          <span>Fit window (min)</span>
          <input
            type="number"
            min={0.1}
            step={0.5}
            value={fitWindowMin}
            onChange={(e) => {
              const v = Number.parseFloat(e.target.value);
              setFitWindowMin(Number.isFinite(v) && v > 0 ? v : DEFAULT_FIT_WINDOW_MIN);
            }}
            aria-label="Fit window in minutes"
          />
        </label>
        <SnapshotButton
          targetRef={svgRef}
          filename="temperature_trend"
          csv={snapshotCsv}
          label="Export temperature trend snapshot"
        />
      </div>

      {/* Heat-up-rate readout box: ramp rate (°/min · °/hr) + fit quality. */}
      <div
        className={`tc-rate-readout ${rateReadout.hasFit ? "has-fit" : ""}`}
        aria-label="Heat-up rate readout"
      >
        <span className="tc-rate-label">
          Heat-up rate (Type {activeTcType}, last {fitWindowMin} min)
        </span>
        <span className="tc-rate-value">{rateReadout.rate}</span>
        <span className="tc-rate-r2">
          {rateReadout.hasFit
            ? rateReadout.r2
            : "select a fit method to read the ramp rate"}
        </span>
      </div>

      <svg
        ref={svgRef}
        className="tc-trend-svg"
        viewBox={`0 0 ${TREND_W} ${TREND_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Temperature trend"
        style={{
          cursor: view.selectionPx ? "ew-resize" : "crosshair",
          touchAction: "none",
        }}
        onPointerDown={(e) => {
          setHoverPx(null); // a drag-zoom gesture supersedes the hover readout
          view.startSelect(plotPx(e.clientX));
        }}
        onPointerMove={(e) => {
          if (view.selectionPx) {
            view.moveSelect(plotPx(e.clientX));
            return;
          }
          setHoverPx(plotPx(e.clientX)); // hover: value readout at the cursor
        }}
        onPointerUp={() => view.endSelect(bounds, pxToUnit)}
        onPointerLeave={() => {
          view.cancelSelect();
          setHoverPx(null);
        }}
      >
        {/* heater-ON status band behind everything else */}
        {showRelay &&
          relaySpans.map((span, i) => {
            const xa = timeToX(span.start, t0, t1, TREND_PAD_L, TREND_W - TREND_PAD_R);
            const xb = timeToX(span.end, t0, t1, TREND_PAD_L, TREND_W - TREND_PAD_R);
            return (
              <rect
                key={i}
                x={xa}
                y={TREND_PAD_T}
                width={Math.max(0.5, xb - xa)}
                height={TREND_PLOT_H}
                className="tc-trend-relay-band"
                fill={RELAY_BAND_COLOR}
              />
            );
          })}

        {/* gridlines + °C axis labels across the resolved [min, max] range */}
        {axisTicks(min, max, 4).map((tick, i) => {
          const y = TREND_PAD_T + (1 - (tick - min) / (max - min)) * TREND_PLOT_H;
          return (
            <g key={i}>
              <line
                x1={TREND_PAD_L}
                y1={y}
                x2={TREND_W - TREND_PAD_R}
                y2={y}
                className="tc-trend-grid"
              />
              <text
                x={TREND_PAD_L - 6}
                y={y + 3}
                className="tc-trend-axis"
                textAnchor="end"
              >
                {tick.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* setpoint + HH reference lines (only when within the visible range) */}
        {max > min && (
          <>
            <line
              x1={TREND_PAD_L}
              y1={refY(setpoint)}
              x2={TREND_W - TREND_PAD_R}
              y2={refY(setpoint)}
              className="tc-trend-ref"
              stroke={SETPOINT_COLOR}
            />
            <line
              x1={TREND_PAD_L}
              y1={refY(hhLimit)}
              x2={TREND_W - TREND_PAD_R}
              y2={refY(hhLimit)}
              className="tc-trend-ref"
              stroke={HH_COLOR}
            />
          </>
        )}

        <TrendTimeAxis
          x0={TREND_PAD_L}
          x1={TREND_W - TREND_PAD_R}
          yBottom={TREND_PAD_T + TREND_PLOT_H}
          spanSeconds={windowSeconds}
        />

        {plotted.length < 2 ? (
          <text
            x={TREND_W / 2}
            y={TREND_H / 2}
            className="tc-trend-empty"
            textAnchor="middle"
          >
            waiting for live data…
          </text>
        ) : (
          <>
            {/* Non-active channel dimmer; active channel bold/solid. Draw the
                inactive one first so the active trace sits on top. */}
            {showR &&
              rPaths.map((d, i) => (
                <path
                  key={`r-${i}`}
                  d={d}
                  className="tc-trend-line"
                  stroke={R_COLOR}
                  opacity={activeTcType === "R" ? 1 : 0.5}
                  strokeWidth={activeTcType === "R" ? 2.4 : 1.4}
                />
              ))}
            {showK &&
              kPaths.map((d, i) => (
                <path
                  key={`k-${i}`}
                  d={d}
                  className="tc-trend-line"
                  stroke={K_COLOR}
                  opacity={activeTcType === "K" ? 1 : 0.5}
                  strokeWidth={activeTcType === "K" ? 2.4 : 1.4}
                />
              ))}
            {fit && (
              <TrendFitOverlay
                fit={fit}
                points={fitOverlayPoints}
                t0={t0}
                t1={t1}
                x0={TREND_PAD_L}
                x1={TREND_W - TREND_PAD_R}
                yScale={refY}
                color={activeColor}
              />
            )}
          </>
        )}

        {/* Hover crosshair + value tooltip (shared overlay). */}
        {hover && (
          <TrendCrosshair
            px={hover.px}
            yTop={TREND_PAD_T}
            yBottom={TREND_PAD_T + TREND_PLOT_H}
            plotLeft={TREND_PAD_L}
            plotRight={TREND_W - TREND_PAD_R}
            series={hover.series}
            xLabel={hover.xLabel}
          />
        )}

        {/* Drag-to-zoom selection: translucent band over the plot area. */}
        {view.selectionPx && (
          <rect
            x={TREND_PAD_L + Math.min(view.selectionPx.fromPx, view.selectionPx.toPx)}
            y={TREND_PAD_T}
            width={Math.abs(view.selectionPx.toPx - view.selectionPx.fromPx)}
            height={TREND_PLOT_H}
            fill="var(--accent-cyan)"
            fillOpacity={0.18}
            stroke="var(--accent-cyan)"
            strokeOpacity={0.6}
          />
        )}
      </svg>
    </div>
  );
};
