import React, { useState, useRef, useMemo } from "react";
import {
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { TAG_INDICES } from "./RoutingMatrix";
import { SnapshotButton } from "./SnapshotButton";
import { type AxisRange, defaultAxisRange } from "../lib/trendAxis";
import {
  SAMPLES_PER_SECOND,
  MAX_TREND_SAMPLES,
  downsample,
  RENDER_MAX_POINTS,
} from "../lib/trendTime";
import { useTrendViewport } from "../hooks/useTrendViewport";
import { useNonPassiveWheel } from "../hooks/useNonPassiveWheel";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendCrosshair, type CrosshairSeries } from "./TrendPlotOverlays";

interface TrendChartProps {
  history: number[][]; // Array of TagValue arrays: number[time][tag_id]
  tagValues: number[];
}

const LINE_COLORS = [
  "#38bdf8", // Cyan
  "#a78bfa", // Purple
  "#10b981", // Green
  "#f472b6", // Pink
  "#f59e0b", // Yellow
  "#0ea5e9", // Blue
  "#c084fc", // Light Purple
  "#f43f5e", // Crimson Red
];

// Window presets, in seconds. Capped at 600 s (10 min) — the depth of the live
// telemetry buffer this multi-tag chart draws from (it has no historian
// backfill, unlike the temperature trend which reaches back hours). Deeper
// history per-tag would mean holding an hour of every tag's frames in the JS
// heap; see MAX_HISTORY in useTelemetryStream for that deliberate tradeoff.
const WINDOW_PRESETS = [10, 30, 60, 120, 300, 600];

export const TrendChart: React.FC<TrendChartProps> = ({ history, tagValues }) => {
  const [selectedTags, setSelectedTags] = useState<number[]>([0, 1, 10]);

  // Y-axis range: "auto" keeps the existing auto-scale + Y-Zoom behavior;
  // manual mode pins the axis to operator-entered min/max.
  const [yAxis, setYAxis] = useState<AxisRange>(defaultAxisRange(0, 100));

  // X-axis pan / zoom / pause / drag-to-zoom all live in the ONE shared,
  // pre-tested viewport model (DRY) instead of bespoke scroll/duration/pause
  // plumbing. The domain here is the SAMPLE INDEX (history rows are per-scan
  // tag arrays broadcast at ~10 Hz). defaultSpan = 60 s, min = 1 s, max = the
  // full retained buffer.
  const view = useTrendViewport({
    defaultSpan: 60 * SAMPLES_PER_SECOND,
    minSpan: SAMPLES_PER_SECOND,
    maxSpan: MAX_TREND_SAMPLES,
  });

  // Y-axis zoom multiplier (Y AXIS ONLY — independent of the viewport's X zoom).
  // 1.0 = Default, < 1.0 = Zoom In, > 1.0 = Zoom Out.
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);

  // Frozen snapshot of the live buffer while paused. Captured in the freeze
  // handler (below) rather than an effect so exactly what's on screen is frozen.
  const [frozenHistory, setFrozenHistory] = useState<number[][]>([]);

  const svgRef = useRef<SVGSVGElement | null>(null);

  // Plot-relative pixel X of the hover crosshair, or null when not hovering.
  // Stored raw here and resolved to the nearest sample during render (where the
  // value→pixel scale lives), so hover never duplicates the projection math.
  const [hoverPx, setHoverPx] = useState<number | null>(null);

  // Freeze/live toggle. Snapshot the CURRENT `history` synchronously BEFORE
  // flipping to paused, so the frame on screen at the click instant is frozen
  // (no frame dropped between the click and a re-render).
  const handlePauseToggle = () => {
    if (!view.paused) setFrozenHistory(history);
    view.togglePause();
  };

  // Active pool: the frozen snapshot while paused, otherwise the live buffer.
  const pool = view.paused ? frozenHistory : history;

  // Slice the pool to the viewport's visible sample-index window. The domain
  // spans the whole pool; the viewport decides which trailing/panned window of
  // it is on screen.
  const bounds = { min: 0, max: pool.length };
  const { start, end } = view.resolve(bounds);
  const clampInt = (n: number) =>
    Math.max(0, Math.min(pool.length, Math.round(n)));
  const sliceStart = clampInt(start);
  const sliceEnd = Math.max(clampInt(end), sliceStart);
  // Full-resolution visible slice — kept intact for the CSV export only.
  const activeHistory = pool.slice(sliceStart, sliceEnd);

  // Render-resolution slice: stride the visible window down to at most
  // RENDER_MAX_POINTS rows BEFORE the extent scan and SVG path build, so a long
  // window (e.g. 300 s = 3000 samples) draws a few hundred points instead of
  // thousands. Memoized on the slice identity so it isn't recomputed on
  // unrelated re-renders.
  const renderHistory = useMemo(
    () => downsample(activeHistory, RENDER_MAX_POINTS),
    [activeHistory],
  );

  // Window (span) in seconds — drives the window-button highlight + the
  // server-smoothing fetch range. How far the right edge sits behind "now",
  // in seconds, drives the "panned -Ns" indicator.
  const windowSeconds = view.viewport.span / SAMPLES_PER_SECOND;
  const pannedSeconds = view.viewport.offset / SAMPLES_PER_SECOND;

  // Time-ago labels for the X axis, derived from the visible index range:
  // secondsAgo(i) = (pool.length - i) / SAMPLES_PER_SECOND.
  const leftSecondsAgo = (pool.length - sliceStart) / SAMPLES_PER_SECOND;
  const rightSecondsAgo = (pool.length - sliceEnd) / SAMPLES_PER_SECOND;

  const toggleTag = (tagId: number) => {
    if (selectedTags.includes(tagId)) {
      setSelectedTags(selectedTags.filter((id) => id !== tagId));
    } else {
      setSelectedTags([...selectedTags, tagId]);
    }
  };

  const selectInputs = () => setSelectedTags([0, 1, 2, 3, 4, 5]);
  const selectOutputs = () => setSelectedTags([10, 11]);
  const clearAll = () => setSelectedTags([]);
  const resetPlotControls = () => {
    view.reset();
    setZoomLevel(1.0);
    setSmoothingMode("none");
    setSmoothedData(null);
    setYAxis(defaultAxisRange(0, 100));
  };

  const [smoothingMode, setSmoothingMode] = useState<string>("none");
  const [smoothedData, setSmoothedData] = useState<{ [tagId: number]: number[] } | null>(null);

  const fetchSmoothedData = async () => {
    if (smoothingMode === "none" || selectedTags.length === 0) return;
    // Freeze the on-screen slice at request time so the live buffer stops
    // advancing under the fetch.
    setFrozenHistory(history);
    view.setPaused(true);

    const nowDate = new Date();
    const windowStartDate = new Date(nowDate.getTime() - windowSeconds * 1000);
    const startIso = windowStartDate.toISOString();
    const endIso = nowDate.toISOString();

    const newSmoothedData: { [tagId: number]: number[] } = {};

    for (const tagId of selectedTags) {
      try {
        const res = await fetch(`/api/trends?tag_id=${tagId}&start_time=${encodeURIComponent(startIso)}&end_time=${encodeURIComponent(endIso)}&smoothing=${smoothingMode}`);
        if (res.ok) {
          const data = await res.json();
          newSmoothedData[tagId] = data.values;
        }
      } catch (e) {
        console.error("Failed to fetch smoothed data for tag", tagId);
      }
    }
    setSmoothedData(newSmoothedData);
  };

  // Dimensions of SVG canvas
  const width = 700;
  const height = 280;
  const paddingLeft = 45;
  const paddingRight = 15;
  const paddingTop = 20;
  const paddingBottom = 30;

  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  // ---- pixel <-> sample-index mapping (the crux of wheel-zoom + drag-zoom) ----
  // The SVG uses viewBox="0 0 width height" with width:100%, so convert a
  // pointer's client X into viewBox space, then to a plot-relative pixel, then
  // to a domain (sample-index) unit via the viewport's current window.
  const plotW = chartWidth;
  const plotPx = (e: { clientX: number }): number => {
    const svg = svgRef.current;
    if (!svg) return 0;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0) return 0; // unlaid-out / jsdom: avoid a NaN focus
    const svgX = ((e.clientX - rect.left) / rect.width) * width;
    return Math.max(0, Math.min(plotW, svgX - paddingLeft));
  };
  const pxToUnit = (px: number): number => {
    const range = view.resolve(bounds);
    return range.start + (px / plotW) * (range.end - range.start);
  };

  // Mouse wheel over the plot zooms about the cursor (X axis, via the viewport).
  // Attached as a NON-passive native listener (see useNonPassiveWheel) so
  // preventDefault suppresses page scroll — React's onWheel is passive.
  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    view.zoomBy(e.deltaY > 0 ? 1.15 : 0.87, pxToUnit(plotPx(e)), bounds);
  };
  useNonPassiveWheel(svgRef, handleWheel);
  // Click-drag to zoom a region: down starts a selection, move grows the
  // overlay rectangle, up zooms to it (release inside), leave cancels it.
  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    setHoverPx(null); // a drag-zoom gesture supersedes the hover readout
    view.startSelect(plotPx(e));
  };
  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (view.selectionPx) {
      view.moveSelect(plotPx(e));
      return;
    }
    setHoverPx(plotPx(e)); // hover: show the value readout at the cursor
  };
  const handlePointerUp = () => {
    view.endSelect(bounds, pxToUnit);
  };
  const handlePointerLeave = () => {
    view.cancelSelect();
    setHoverPx(null);
  };

  // Compute scale boundaries across selected tags. Memoized on the exact inputs
  // that move the axis (selected tags, the render-resolution slice, any server
  // smoothing, zoom, manual override) so the extent scan doesn't re-run on
  // unrelated re-renders. The scan reads `renderHistory` (already strided to
  // RENDER_MAX_POINTS) rather than the full slice.
  const { minVal, maxVal } = useMemo(() => {
    let lo = 0;
    let hi = 100;

    if (selectedTags.length > 0) {
      let realMin = Infinity;
      let realMax = -Infinity;
      let hasData = false;

      if (smoothedData) {
        const series = Object.values(smoothedData);
        for (let i = 0; i < series.length; i++) {
          const values = series[i];
          for (let j = 0; j < values.length; j++) {
            hasData = true;
            const val = values[j];
            if (val < realMin) realMin = val;
            if (val > realMax) realMax = val;
          }
        }
      } else if (renderHistory.length > 0) {
        hasData = true;
        for (let i = 0; i < renderHistory.length; i++) {
          const sample = renderHistory[i];
          for (let j = 0; j < selectedTags.length; j++) {
            const val = sample[selectedTags[j]] ?? 0;
            if (val < realMin) realMin = val;
            if (val > realMax) realMax = val;
          }
        }
      }

      if (hasData) {
        const delta = realMax - realMin;
        const padding = delta > 0 ? delta * 0.1 : 5;
        lo = Math.max(0, realMin - padding);
        hi = realMax + padding;

        // Apply Zoom multiplier
        const center = lo + (hi - lo) / 2;
        const zoomedRange = (hi - lo) * zoomLevel;
        lo = Math.max(0, center - zoomedRange / 2);
        hi = center + zoomedRange / 2;
      }
    }

    // Manual axis override wins over the auto-scale + zoom computed above.
    if (!yAxis.auto) {
      lo = yAxis.min;
      hi = yAxis.max > yAxis.min ? yAxis.max : yAxis.min + 1;
    }

    return { minVal: lo, maxVal: hi };
  }, [selectedTags, smoothedData, renderHistory, zoomLevel, yAxis]);

  const valRange = maxVal - minVal || 1;

  // Convert (index, value) to SVG (X, Y) coordinates
  const getCoordinates = (index: number, value: number, totalPoints: number) => {
    if (totalPoints <= 1) return { x: paddingLeft, y: paddingTop + chartHeight / 2 };
    const x = paddingLeft + (index / (totalPoints - 1)) * chartWidth;
    const y = paddingTop + chartHeight - ((value - minVal) / valRange) * chartHeight;
    return { x, y };
  };

  // Precompute each selected trace's SVG line + area `d` strings ONCE per input
  // change instead of on every render. Reads the render-resolution slice
  // (`renderHistory`) — the expensive per-point string concat now runs over a
  // few hundred points, not thousands, and is skipped entirely when none of the
  // inputs changed.
  const seriesPaths = useMemo(() => {
    const lastX = paddingLeft + chartWidth;
    return selectedTags.map((tagId, activeIdx) => {
      const color = LINE_COLORS[activeIdx % LINE_COLORS.length];
      const useSmoothed =
        smoothedData && smoothedData[tagId] && smoothedData[tagId].length > 0;
      const totalPoints = useSmoothed
        ? smoothedData[tagId].length
        : renderHistory.length;

      if (totalPoints < 2) {
        return { tagId, activeIdx, color, pathD: "", areaD: "", drawable: false };
      }

      let pathD = "";
      let areaD = `M ${paddingLeft} ${paddingTop + chartHeight} `;

      // ⚡ Bolt Optimization: Replace .forEach iterations with a standard for loop to eliminate closure allocation overhead in hot paths.
      if (useSmoothed) {
        const data = smoothedData[tagId];
        for (let sampleIdx = 0; sampleIdx < data.length; sampleIdx++) {
          const val = data[sampleIdx];
          const { x, y } = getCoordinates(sampleIdx, val, totalPoints);
          pathD += `${sampleIdx === 0 ? "M" : "L"} ${x} ${y} `;
          areaD += `L ${x} ${y} `;
        }
      } else {
        for (let sampleIdx = 0; sampleIdx < renderHistory.length; sampleIdx++) {
          const sample = renderHistory[sampleIdx];
          const val = sample[tagId] ?? 0;
          const { x, y } = getCoordinates(sampleIdx, val, totalPoints);
          pathD += `${sampleIdx === 0 ? "M" : "L"} ${x} ${y} `;
          areaD += `L ${x} ${y} `;
        }
      }

      areaD += `L ${lastX} ${paddingTop + chartHeight} Z`;
      return { tagId, activeIdx, color, pathD, areaD, drawable: true };
    });
    // getCoordinates depends only on the memoized minVal/maxVal (valRange) and
    // static chart dims, so the extent inputs below are sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTags, renderHistory, smoothedData, minVal, maxVal]);

  // Resolve the hover crosshair from the raw pointer pixel, snapping to the
  // nearest drawn sample and reading each selected tag's value there. Reuses the
  // same getCoordinates projection as the series paths (DRY) so the marker sits
  // exactly on the line. Computed inline (cheap: a handful of series) rather than
  // memoized because it depends on the full projection state.
  const seriesLen = (tagId: number): number => {
    const sm = smoothedData?.[tagId];
    return sm && sm.length > 0 ? sm.length : renderHistory.length;
  };
  const seriesValueAt = (tagId: number, idx: number): number => {
    const sm = smoothedData?.[tagId];
    if (sm && sm.length > 0) return sm[idx] ?? 0;
    return renderHistory[idx]?.[tagId] ?? 0;
  };
  const hover: { px: number; series: CrosshairSeries[]; xLabel: string } | null =
    (() => {
      if (hoverPx === null) return null;
      const drawables = seriesPaths.filter((s) => s.drawable);
      if (drawables.length === 0) return null;
      const f = chartWidth > 0 ? Math.max(0, Math.min(1, hoverPx / chartWidth)) : 0;
      const firstLen = seriesLen(drawables[0].tagId);
      if (firstLen < 2) return null;
      const snapIdx = Math.max(0, Math.min(firstLen - 1, Math.round(f * (firstLen - 1))));
      const snappedX = getCoordinates(snapIdx, 0, firstLen).x;
      const series: CrosshairSeries[] = drawables.map(({ tagId, color }) => {
        const n = seriesLen(tagId);
        const idx = Math.max(0, Math.min(n - 1, Math.round(f * (n - 1))));
        const value = seriesValueAt(tagId, idx);
        return {
          label: `#${tagId}`,
          color,
          text: value.toFixed(1),
          py: getCoordinates(idx, value, n).y,
        };
      });
      const secondsAgo = leftSecondsAgo + f * (rightSecondsAgo - leftSecondsAgo);
      const xLabel =
        secondsAgo < 0.5
          ? "now"
          : secondsAgo < 60
            ? `-${secondsAgo.toFixed(1)}s`
            : `-${(secondsAgo / 60).toFixed(1)}m`;
      return { px: snappedX, series, xLabel };
    })();

  // Snapshot/export data for the shared SnapshotButton: PNG + SVG come straight
  // from the <svg> ref; CSV is the full-resolution visible slice of the selected
  // tags. `undefined` (no tags / no data) hides the CSV button.
  const snapshotCsv =
    selectedTags.length > 0 && activeHistory.length > 0
      ? {
          headers: ["index", ...selectedTags.map((tagId) => `tag_${tagId}`)],
          rows: activeHistory.map((sample, idx) => [
            idx,
            ...selectedTags.map((tagId) => sample[tagId] ?? 0),
          ]),
        }
      : undefined;

  const gridLinesY = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      <div className="panel-header" style={{ marginBottom: "0.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span>Trends</span>
          {!view.live && (
            <span
              style={{
                fontSize: "0.7rem",
                color: "var(--color-warning)",
                border: "1px solid var(--color-warning)",
                borderRadius: "3px",
                padding: "0.05rem 0.25rem",
                fontWeight: 600,
                textTransform: "uppercase",
                fontFamily: "var(--font-mono)",
              }}
            >
              {view.paused ? "FROZEN" : `panned -${Math.round(pannedSeconds)}s`}
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: "0.3rem" }}>
          {WINDOW_PRESETS.map((t) => {
            const active = Math.round(windowSeconds) === t;
            return (
              <button
                key={t}
                type="button"
                onClick={() => view.setSpan(t * SAMPLES_PER_SECOND)}
                className="btn"
                style={{
                  padding: "0.15rem 0.4rem",
                  fontSize: "0.7rem",
                  border: "1px solid",
                  borderColor: active ? "var(--accent-cyan)" : "var(--panel-border)",
                  background: active ? "var(--cell-hover-bg)" : "var(--input-bg)",
                  color: active ? "var(--accent-cyan)" : "var(--text-secondary)",
                }}
              >
                {t}s
              </button>
            );
          })}
        </div>
      </div>

      {/* Advanced Control Dashboard Toolbar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "1rem",
          background: "var(--input-bg)",
          padding: "0.4rem 0.6rem",
          borderRadius: "4px",
          border: "1px solid var(--panel-border)",
          flexWrap: "wrap",
        }}
      >
        {/* Playback + pan (X-axis viewport) Controls */}
        <div style={{ display: "flex", gap: "0.3rem" }}>
          <button
            type="button"
            onClick={handlePauseToggle}
            className="btn"
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
            title={view.paused ? "Resume Live Stream" : "Pause / Freeze Plot"}
            aria-label={view.paused ? "Resume live stream" : "Pause / freeze plot"}
          >
            {view.paused ? <Play size={12} color="var(--color-success)" /> : <Pause size={12} />}
            <span>{view.paused ? "Live" : "Freeze"}</span>
          </button>
          <button
            type="button"
            onClick={() => view.panBy((end - start) * 0.3, bounds)}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Scroll back in time (older)"
            aria-label="Scroll back in time (older)"
          >
            <ChevronLeft size={12} />
          </button>
          <button
            type="button"
            onClick={() => view.panBy(-(end - start) * 0.3, bounds)}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Scroll toward now (newer)"
            aria-label="Scroll toward now (newer)"
          >
            <ChevronRight size={12} />
          </button>
          <button
            type="button"
            onClick={resetPlotControls}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Reset Plot Zoom/Pan Settings"
            aria-label="Reset Plot Zoom/Pan Settings"
          >
            <RotateCcw size={12} />
          </button>
        </div>

        {/* Zoom Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Y-Zoom</span>
          <button
            type="button"
            onClick={() => setZoomLevel((prev) => Math.max(0.2, prev - 0.15))}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Zoom In"
            aria-label="Zoom In"
          >
            <ZoomIn size={12} />
          </button>
          <button
            type="button"
            onClick={() => setZoomLevel((prev) => Math.min(3.0, prev + 0.15))}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Zoom Out"
            aria-label="Zoom Out"
          >
            <ZoomOut size={12} />
          </button>
        </div>

        {/* Server-Side Smoothing */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Smoothing</span>
          <select
            value={smoothingMode}
            onChange={(e) => setSmoothingMode(e.target.value)}
            className="form-input"
            style={{ padding: "0.15rem 0.4rem", fontSize: "0.75rem", height: "auto" }}
          >
            <option value="none">None</option>
            <option value="moving_average">Moving Avg</option>
            <option value="exponential_smoothing">Exp Smooth</option>
          </select>
          {smoothingMode !== "none" && (
            <button
              type="button"
              onClick={fetchSmoothedData}
              className="btn btn-primary"
              style={{ padding: "0.15rem 0.5rem", fontSize: "0.75rem" }}
            >
              Apply
            </button>
          )}
        </div>

        {/* Snapshot / export (PNG + SVG + CSV) via the shared control */}
        <SnapshotButton
          targetRef={svgRef}
          filename="p1am_trend"
          csv={snapshotCsv}
          label="Export trend snapshot"
        />
      </div>

      {/* Manual Y-axis range override (auto-scale + Y-Zoom still apply when "Auto Y" is on) */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          background: "var(--input-bg)",
          padding: "0.35rem 0.6rem",
          borderRadius: "4px",
          border: "1px solid var(--panel-border)",
        }}
      >
        <span
          style={{
            fontSize: "0.7rem",
            color: "var(--text-secondary)",
            minWidth: "75px",
            textTransform: "uppercase",
          }}
        >
          Y Axis:
        </span>
        <TrendAxisControls value={yAxis} onChange={setYAxis} />
      </div>

      {/* Macro Controls */}
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", alignItems: "center" }}>
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginRight: "0.25rem" }}>
          Quick Toggles:
        </span>
        <button
          type="button"
          onClick={selectInputs}
          className="btn"
          style={{ padding: "0.2rem 0.5rem", fontSize: "0.7rem" }}
        >
          Inputs (0-5)
        </button>
        <button
          type="button"
          onClick={selectOutputs}
          className="btn"
          style={{ padding: "0.2rem 0.5rem", fontSize: "0.7rem" }}
        >
          Outputs (10-11)
        </button>
        <button
          type="button"
          onClick={clearAll}
          className="btn"
          style={{ padding: "0.2rem 0.5rem", fontSize: "0.7rem" }}
        >
          Clear
        </button>
      </div>

      {/* Tag Multi-Select Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(72px, 1fr))",
          gap: "0.2rem",
          maxHeight: "82px",
          overflowY: "auto",
          padding: "0.2rem",
          background: "var(--input-bg)",
          borderRadius: "4px",
          border: "1px solid var(--panel-border)",
        }}
      >
        {TAG_INDICES.map((i) => {
          const isSelected = selectedTags.includes(i);
          const colorIdx = selectedTags.indexOf(i);
          const buttonColor = isSelected ? LINE_COLORS[colorIdx % LINE_COLORS.length] : "transparent";

          return (
            <button
              key={i}
              type="button"
              onClick={() => toggleTag(i)}
              className="btn"
              style={{
                padding: "0.15rem 0.3rem",
                fontSize: "0.7rem",
                fontFamily: "var(--font-mono)",
                borderColor: isSelected ? buttonColor : "var(--panel-border)",
                background: isSelected ? `${buttonColor}12` : "transparent",
                color: isSelected ? buttonColor : "var(--text-muted)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>#{i}</span>
              <span style={{ fontSize: "0.65rem", opacity: 0.8 }}>
                {tagValues[i]?.toFixed(1) ?? "0"}
              </span>
            </button>
          );
        })}
      </div>

      {/* SVG Canvas */}
      <div style={{ position: "relative", width: "100%" }}>
        {selectedTags.length === 0 ? (
          <div
            className="flex-center"
            style={{
              height: `${height}px`,
              color: "var(--text-muted)",
              fontSize: "0.85rem",
              border: "1px dashed var(--panel-border)",
              borderRadius: "4px",
            }}
          >
            No active tags selected for graphing.
          </div>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${width} ${height}`}
            style={{
              width: "100%",
              height: "auto",
              overflow: "visible",
              touchAction: "none",
              cursor: "crosshair",
            }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
          >
            <defs>
              {/* Linear Gradients under lines */}
              {LINE_COLORS.map((color, idx) => (
                <linearGradient key={idx} id={`grad-${idx}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity="0.08" />
                  <stop offset="100%" stopColor={color} stopOpacity="0" />
                </linearGradient>
              ))}
            </defs>

            {/* Grid Lines */}
            {gridLinesY.map((ratio, idx) => {
              const y = paddingTop + ratio * chartHeight;
              const labelVal = maxVal - ratio * valRange;
              return (
                <g key={idx}>
                  <line
                    x1={paddingLeft}
                    y1={y}
                    x2={width - paddingRight}
                    y2={y}
                    stroke="var(--panel-border)"
                    strokeDasharray="2 3"
                  />
                  <text
                    x={paddingLeft - 8}
                    y={y + 3}
                    fill="var(--text-muted)"
                    fontSize="9"
                    fontFamily="var(--font-mono)"
                    textAnchor="end"
                  >
                    {labelVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Time axis helper */}
            <line
              x1={paddingLeft}
              y1={paddingTop + chartHeight}
              x2={width - paddingRight}
              y2={paddingTop + chartHeight}
              stroke="var(--panel-border)"
            />
            <text
              x={paddingLeft}
              y={height - 8}
              fill="var(--text-muted)"
              fontSize="9"
              textAnchor="start"
            >
              {`-${Math.round(leftSecondsAgo)}s`}
            </text>
            <text
              x={width - paddingRight}
              y={height - 8}
              fill="var(--text-muted)"
              fontSize="9"
              textAnchor="end"
            >
              {rightSecondsAgo < 0.5 ? "now" : `-${Math.round(rightSecondsAgo)}s`}
            </text>

            {/* Paths for active tag series (precomputed in `seriesPaths`) */}
            {seriesPaths.map(({ tagId, activeIdx, color, pathD, areaD, drawable }) =>
              drawable ? (
                <g key={tagId}>
                  {/* Area Gradient */}
                  <path d={areaD} fill={`url(#grad-${activeIdx})`} />
                  {/* Line Path */}
                  <path
                    d={pathD}
                    fill="none"
                    stroke={color}
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
              ) : null,
            )}

            {/* Hover crosshair + value tooltip (shared overlay) */}
            {hover && (
              <TrendCrosshair
                px={hover.px}
                yTop={paddingTop}
                yBottom={paddingTop + chartHeight}
                plotLeft={paddingLeft}
                plotRight={width - paddingRight}
                series={hover.series}
                xLabel={hover.xLabel}
              />
            )}

            {/* Click-drag zoom-region overlay */}
            {view.selectionPx && (
              <rect
                x={paddingLeft + Math.min(view.selectionPx.fromPx, view.selectionPx.toPx)}
                y={paddingTop}
                width={Math.abs(view.selectionPx.toPx - view.selectionPx.fromPx)}
                height={chartHeight}
                fill="var(--accent-cyan)"
                fillOpacity="0.15"
                stroke="var(--accent-cyan)"
                strokeWidth="1"
                pointerEvents="none"
              />
            )}
          </svg>
        )}
      </div>

      {/* Legend Indicators */}
      {selectedTags.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", marginTop: "0.1rem" }}>
          {selectedTags.map((tagId, activeIdx) => {
            const color = LINE_COLORS[activeIdx % LINE_COLORS.length];
            return (
              <div key={tagId} style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                <span
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    backgroundColor: color,
                  }}
                />
                <span style={{ fontSize: "0.75rem", fontWeight: 600 }}>
                  Tag {tagId}
                </span>
                <span
                  style={{
                    fontSize: "0.7rem",
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-secondary)",
                    background: "rgba(0,0,0,0.1)",
                    padding: "0.05rem 0.2rem",
                    borderRadius: "2px",
                    border: "1px solid var(--panel-border)",
                  }}
                >
                  {tagValues[tagId]?.toFixed(2) ?? "0.00"}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
