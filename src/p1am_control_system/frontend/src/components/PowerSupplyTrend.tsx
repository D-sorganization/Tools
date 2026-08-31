import React, { useRef, useState } from "react";
import {
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import {
  type AxisRange,
  defaultAxisRange,
  resolveRange,
  axisTicks,
} from "../lib/trendAxis";
import {
  downsample,
  formatWindow,
  formatClock,
  timeSeriesPath,
  timeToX,
  valueToY,
  BUFFER_WINDOW_SECONDS,
} from "../lib/trendTime";
import { nearestIndexByX } from "../lib/plotCursor";
import { useTrendViewport } from "../hooks/useTrendViewport";
import { useNonPassiveWheel } from "../hooks/useNonPassiveWheel";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendTimeControls } from "./TrendTimeControls";
import { TrendTimeAxis, TrendCrosshair, type CrosshairSeries } from "./TrendPlotOverlays";
import { SnapshotButton } from "./SnapshotButton";

/**
 * Compact dual-trace trend for the power-supply screen: measured current and
 * voltage coming back from the unit, plotted as a percent of their respective
 * full scales on a shared axis so both fit one clean chart. Live engineering
 * values are shown in the legend.
 *
 * The shared axis is in % of full scale and defaults to 0-100 %, but is
 * rescalable (Auto-fit or manual min/max %) via the shared TrendAxisControls so
 * small signals can be zoomed in.
 *
 * Pan / zoom / pause / drag-to-zoom come from the shared {@link useTrendViewport}
 * model (DRY): the viewport owns "which slice of time is on screen" in epoch-ms,
 * this component only supplies the data bounds and a pixel<->time converter.
 *
 * Self-contained SVG (no chart lib): the parent accumulates a rolling sample
 * buffer from the live status and passes it in.
 */

export interface TrendSample {
  t: number; // epoch ms when sampled
  i: number; // measured current (A)
  v: number; // measured voltage (V)
  p: number; // measured power (W)
}

interface Props {
  samples: TrendSample[];
  currentFullScale: number;
  voltageFullScale: number;
  powerFullScale: number;
  currentLabel?: string;
  voltageLabel?: string;
}

const W = 600;
const H = 192;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 28; // room for the X-axis time labels
const PLOT_H = H - PAD_T - PAD_B;
const PLOT_W = W - PAD_L - PAD_R;

/** Default (and reset) visible window: 60 minutes, matching the original trend. */
const DEFAULT_WINDOW_SECONDS = 3600;

const CURRENT_COLOR = "var(--color-success)";
const VOLTAGE_COLOR = "var(--accent-purple)";
const POWER_COLOR = "var(--accent-cyan)";

const btnStyle: React.CSSProperties = {
  padding: "0.2rem 0.4rem",
  fontSize: "0.7rem",
  display: "inline-flex",
  alignItems: "center",
  gap: "0.2rem",
};

/** Convert a measured value to percent of its full scale (0 when no scale). */
function toPct(value: number, fullScale: number): number {
  return fullScale > 0 ? (value / fullScale) * 100 : 0;
}

export const PowerSupplyTrend: React.FC<Props> = ({
  samples,
  currentFullScale,
  voltageFullScale,
  powerFullScale,
  currentLabel = "Current",
  voltageLabel = "Voltage",
}) => {
  const [axis, setAxis] = useState<AxisRange>(defaultAxisRange(0, 100));
  // Frozen snapshot of the buffer taken when the operator pauses, so live frames
  // arriving afterwards don't move the plot.
  const [frozen, setFrozen] = useState<TrendSample[]>([]);
  // Plot pixel X of the hover crosshair, or null when not hovering.
  const [hoverPx, setHoverPx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // The one place pan/zoom/pause/drag-to-zoom live. Domain is epoch-ms. This
  // trend has no historian backfill, so it caps its zoom-out at the live-buffer
  // depth (BUFFER_WINDOW_SECONDS) rather than the longer backfilled max, so the
  // axis never stretches past the retained data.
  const view = useTrendViewport({
    defaultSpan: DEFAULT_WINDOW_SECONDS * 1000,
    minSpan: 1000,
    maxSpan: BUFFER_WINDOW_SECONDS * 1000,
  });

  // Active buffer: the frozen snapshot while paused, the live samples otherwise.
  const active = view.paused ? frozen : samples;

  // Data bounds in ms. Guard the empty/one-sample case: a degenerate range
  // (max <= min) renders the placeholder instead of a chart.
  const bounds = {
    min: active[0]?.t ?? 0,
    max: active.length ? active[active.length - 1].t : 0,
  };
  const hasData = bounds.max > bounds.min;

  // Visible window (ms) from the viewport, then slice + downsample the buffer.
  const { start, end } = view.resolve(bounds);
  const visible = hasData
    ? active.filter((s) => s.t >= start && s.t <= end)
    : [];
  const down = downsample(visible);
  const iPts = down.map((s) => ({ t: s.t, v: toPct(s.i, currentFullScale) }));
  const vPts = down.map((s) => ({ t: s.t, v: toPct(s.v, voltageFullScale) }));
  const pPts = down.map((s) => ({ t: s.t, v: toPct(s.p, powerFullScale) }));
  const { min, max } = resolveRange(
    axis,
    [...iPts, ...vPts, ...pPts].map((p) => p.v),
    { min: 0, max: 100 },
  );

  // resolve() always returns a window exactly `span` wide, so the shown window
  // (legend + axis span) is the current zoom level.
  const windowSeconds = (end - start) / 1000;

  const geom = {
    t0: start,
    t1: end,
    min,
    max,
    x0: PAD_L,
    x1: W - PAD_R,
    yTop: PAD_T,
    plotH: PLOT_H,
  };

  const currentPath = timeSeriesPath(iPts, geom);
  const voltagePath = timeSeriesPath(vPts, geom);
  const powerPath = timeSeriesPath(pPts, geom);
  const last = active.length ? active[active.length - 1] : undefined;

  // --- Pixel <-> time mapping (viewBox is 0..W, the <svg> stretches to 100%). ---
  const plotPx = (e: { clientX: number }): number => {
    const svg = svgRef.current;
    if (!svg) return 0;
    const r = svg.getBoundingClientRect();
    if (r.width === 0) return 0; // unlaid-out / jsdom: avoid a NaN focus
    const x = ((e.clientX - r.left) / r.width) * W;
    return Math.max(0, Math.min(PLOT_W, x - PAD_L));
  };
  const pxToUnit = (px: number): number =>
    start + (px / PLOT_W) * (end - start);

  // --- Interaction handlers ---
  const onTogglePause = (): void => {
    if (!view.paused) setFrozen(samples); // snapshot exactly what's on screen
    view.togglePause();
  };
  const panStep = (end - start) * 0.3;
  const focusCenter = (start + end) / 2;

  // Non-passive native wheel listener (see useNonPassiveWheel) so preventDefault
  // stops the page scrolling while wheel-zooming; React's onWheel is passive.
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    view.zoomBy(e.deltaY > 0 ? 1.15 : 0.87, pxToUnit(plotPx(e)), bounds);
  };
  useNonPassiveWheel(svgRef, onWheel);
  const onPointerDown = (e: React.PointerEvent): void => {
    setHoverPx(null); // a drag-zoom gesture supersedes the hover readout
    view.startSelect(plotPx(e));
  };
  const onPointerMove = (e: React.PointerEvent): void => {
    if (view.selectionPx) {
      view.moveSelect(plotPx(e));
      return;
    }
    setHoverPx(plotPx(e)); // hover: show the value readout at the cursor
  };
  const onPointerUp = (): void => view.endSelect(bounds, pxToUnit);
  const onPointerLeave = (): void => {
    view.cancelSelect();
    setHoverPx(null);
  };

  // Resolve the hover crosshair: snap to the nearest sample by time, read each
  // trace's engineering value there, and place its marker with the SAME
  // time→x / value→y mapping the paths use (DRY via timeToX / valueToY).
  const hover: { px: number; series: CrosshairSeries[]; xLabel: string } | null =
    (() => {
      if (hoverPx === null || !hasData || down.length === 0) return null;
      const idx = nearestIndexByX(
        down.map((s) => s.t),
        pxToUnit(hoverPx),
      );
      if (idx === null) return null;
      const s = down[idx];
      const kW = s.p / 1000;
      const series: CrosshairSeries[] = [
        {
          label: currentLabel,
          color: CURRENT_COLOR,
          text: `${s.i.toFixed(2)} A`,
          py: valueToY(toPct(s.i, currentFullScale), min, max, PAD_T, PLOT_H),
        },
        {
          label: voltageLabel,
          color: VOLTAGE_COLOR,
          text: `${s.v.toFixed(2)} V`,
          py: valueToY(toPct(s.v, voltageFullScale), min, max, PAD_T, PLOT_H),
        },
        {
          label: "Power",
          color: POWER_COLOR,
          text: `${kW.toFixed(kW >= 10 ? 1 : 2)} kW`,
          py: valueToY(toPct(s.p, powerFullScale), min, max, PAD_T, PLOT_H),
        },
      ];
      return {
        px: timeToX(s.t, start, end, PAD_L, W - PAD_R),
        series,
        xLabel: formatClock(s.t),
      };
    })();

  // CSV export of the raw measured samples (t, current, voltage, power). Hidden
  // (undefined) until at least one sample has been captured.
  const snapshotCsv = samples.length
    ? {
        headers: ["t_ms", "current_a", "voltage_v", "power_w"],
        rows: samples.map((s) => [s.t, s.i, s.v, s.p]),
      }
    : undefined;

  return (
    <div className="ps-trend">
      <div className="ps-trend-legend">
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: CURRENT_COLOR }} />
          {currentLabel}
          <strong>{last ? `${last.i.toFixed(2)} A` : "—"}</strong>
        </span>
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: VOLTAGE_COLOR }} />
          {voltageLabel}
          <strong>{last ? `${last.v.toFixed(2)} V` : "—"}</strong>
        </span>
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: POWER_COLOR }} />
          Power
          <strong>
            {last ? `${(last.p / 1000).toFixed(last.p / 1000 >= 10 ? 1 : 2)} kW` : "—"}
          </strong>
        </span>
        <span className="ps-trend-window">
          last {formatWindow(windowSeconds)} · % of full scale
        </span>
        {!view.live && (
          <span
            style={{
              fontSize: "0.62rem",
              color: "var(--color-warning)",
              border: "1px solid var(--color-warning)",
              borderRadius: "3px",
              padding: "0.02rem 0.25rem",
              fontWeight: 600,
              textTransform: "uppercase",
            }}
          >
            {view.paused ? "Frozen" : "panned"}
          </span>
        )}
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
        {/* Playback + pan + zoom, driven by the shared viewport model. */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
          <button
            type="button"
            className="btn"
            style={btnStyle}
            onClick={onTogglePause}
            title={view.paused ? "Resume live stream" : "Pause / freeze plot"}
            aria-label={view.paused ? "Resume live stream" : "Pause plot"}
          >
            {view.paused ? (
              <Play size={12} color="var(--color-success)" />
            ) : (
              <Pause size={12} />
            )}
            <span>{view.paused ? "Live" : "Freeze"}</span>
          </button>
          <button
            type="button"
            className="btn"
            style={btnStyle}
            onClick={() => view.panBy(panStep, bounds)}
            title="Scroll back in time"
            aria-label="Scroll back in time"
          >
            <ChevronLeft size={12} />
          </button>
          <button
            type="button"
            className="btn"
            style={btnStyle}
            onClick={() => view.panBy(-panStep, bounds)}
            title="Scroll forward in time"
            aria-label="Scroll forward in time"
          >
            <ChevronRight size={12} />
          </button>
          <button
            type="button"
            className="btn"
            style={btnStyle}
            onClick={() => view.zoomBy(0.8, focusCenter, bounds)}
            title="Zoom in (time)"
            aria-label="Zoom in"
          >
            <ZoomIn size={12} />
          </button>
          <button
            type="button"
            className="btn"
            style={btnStyle}
            onClick={() => view.zoomBy(1.25, focusCenter, bounds)}
            title="Zoom out (time)"
            aria-label="Zoom out"
          >
            <ZoomOut size={12} />
          </button>
          <button
            type="button"
            className="btn"
            style={btnStyle}
            onClick={view.reset}
            title="Reset pan / zoom / freeze"
            aria-label="Reset"
          >
            <RotateCcw size={12} />
            <span>Reset</span>
          </button>
        </div>

        <TrendTimeControls
          value={windowSeconds}
          onChange={(seconds) => view.setSpan(seconds * 1000)}
          maxSeconds={BUFFER_WINDOW_SECONDS}
        />
        <TrendAxisControls value={axis} onChange={setAxis} unit="%" />
        <SnapshotButton
          targetRef={svgRef}
          filename="power_supply_trend"
          csv={snapshotCsv}
          label="Export power-supply trend snapshot"
        />
      </div>

      <svg
        ref={svgRef}
        className="ps-trend-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Current and voltage trend"
        style={{ touchAction: "none", cursor: "crosshair" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
      >
        {/* gridlines + % axis labels across the resolved [min, max] range */}
        {axisTicks(min, max, 4).map((tick, i) => {
          const y = PAD_T + (1 - (tick - min) / (max - min)) * PLOT_H;
          return (
            <g key={i}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} className="ps-trend-grid" />
              <text x={PAD_L - 6} y={y + 3} className="ps-trend-axis" textAnchor="end">
                {tick.toFixed(Number.isInteger(tick) ? 0 : 1)}
              </text>
            </g>
          );
        })}

        <TrendTimeAxis
          x0={PAD_L}
          x1={W - PAD_R}
          yBottom={PAD_T + PLOT_H}
          spanSeconds={windowSeconds}
        />

        {!hasData ? (
          <text x={W / 2} y={H / 2} className="ps-trend-empty" textAnchor="middle">
            waiting for live data…
          </text>
        ) : (
          <>
            <path d={powerPath} className="ps-trend-line" stroke={POWER_COLOR} />
            <path d={voltagePath} className="ps-trend-line" stroke={VOLTAGE_COLOR} />
            <path d={currentPath} className="ps-trend-line" stroke={CURRENT_COLOR} />
          </>
        )}

        {/* Hover crosshair + value tooltip (shared overlay). */}
        {hover && (
          <TrendCrosshair
            px={hover.px}
            yTop={PAD_T}
            yBottom={PAD_T + PLOT_H}
            plotLeft={PAD_L}
            plotRight={W - PAD_R}
            series={hover.series}
            xLabel={hover.xLabel}
          />
        )}

        {/* Translucent drag-to-zoom region while a selection is in progress. */}
        {view.selectionPx && (
          <rect
            x={PAD_L + Math.min(view.selectionPx.fromPx, view.selectionPx.toPx)}
            y={PAD_T}
            width={Math.abs(view.selectionPx.toPx - view.selectionPx.fromPx)}
            height={PLOT_H}
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
