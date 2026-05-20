import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, ZoomIn, ZoomOut, Image, FileSpreadsheet, RotateCcw } from "lucide-react";

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

export const TrendChart: React.FC<TrendChartProps> = ({ history, tagValues }) => {
  const [selectedTags, setSelectedTags] = useState<number[]>([0, 1, 10]);
  const [duration, setDuration] = useState<number>(60); // Time window in seconds

  // Advanced Plot Control States
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [frozenHistory, setFrozenHistory] = useState<number[][]>([]);
  const [zoomLevel, setZoomLevel] = useState<number>(1.0); // 1.0 = Default, < 1.0 = Zoom In, > 1.0 = Zoom Out
  const [scrollOffset, setScrollOffset] = useState<number>(0); // Scroll offset back in samples

  const svgRef = useRef<SVGSVGElement | null>(null);

  // Freeze history when paused
  useEffect(() => {
    if (isPaused) {
      setFrozenHistory([...history]);
    }
  }, [isPaused]);

  // Adjust scroll offset bounds when history changes
  const activeHistoryPool = isPaused ? frozenHistory : history;
  const maxSamples = duration * 10;
  
  // Bound scroll offset so we don't scroll past the start of the buffer
  const maxPossibleOffset = Math.max(0, activeHistoryPool.length - maxSamples);
  const safeOffset = Math.min(scrollOffset, maxPossibleOffset);

  // Slice history pool based on duration and offset
  const sliceEnd = activeHistoryPool.length - safeOffset;
  const sliceStart = Math.max(0, sliceEnd - maxSamples);
  const activeHistory = activeHistoryPool.slice(sliceStart, sliceEnd);

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
    setIsPaused(false);
    setZoomLevel(1.0);
    setScrollOffset(0);
    setSmoothingMode("none");
    setSmoothedData(null);
  };

  const [smoothingMode, setSmoothingMode] = useState<string>("none");
  const [smoothedData, setSmoothedData] = useState<{ [tagId: number]: number[] } | null>(null);

  const fetchSmoothedData = async () => {
    if (smoothingMode === "none" || selectedTags.length === 0) return;
    setIsPaused(true);
    
    const end = new Date();
    const start = new Date(end.getTime() - duration * 1000);
    const startIso = start.toISOString();
    const endIso = end.toISOString();

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

  // Compute scale boundaries across selected tags in history
  let minVal = 0;
  let maxVal = 100;

  if (selectedTags.length > 0) {
    let activeValues: number[] = [];
    if (smoothedData) {
      activeValues = Object.values(smoothedData).flat();
    } else if (activeHistory.length > 0) {
      activeValues = activeHistory.flatMap((sample) =>
        selectedTags.map((tagId) => sample[tagId] ?? 0)
      );
    }

    if (activeValues.length > 0) {
      const realMin = Math.min(...activeValues);
      const realMax = Math.max(...activeValues);
      const delta = realMax - realMin;
      
      const padding = delta > 0 ? delta * 0.1 : 5;
      minVal = Math.max(0, realMin - padding);
      maxVal = realMax + padding;

      // Apply Zoom multiplier
      const center = minVal + (maxVal - minVal) / 2;
      const zoomedRange = (maxVal - minVal) * zoomLevel;
      minVal = Math.max(0, center - zoomedRange / 2);
      maxVal = center + zoomedRange / 2;
    }
  }

  const valRange = maxVal - minVal || 1;

  // Convert (index, value) to SVG (X, Y) coordinates
  const getCoordinates = (index: number, value: number, totalPoints: number) => {
    if (totalPoints <= 1) return { x: paddingLeft, y: paddingTop + chartHeight / 2 };
    const x = paddingLeft + (index / (totalPoints - 1)) * chartWidth;
    const y = paddingTop + chartHeight - ((value - minVal) / valRange) * chartHeight;
    return { x, y };
  };

  // Snapshot functionality: Download SVG screenshot
  const downloadSnapshotSVG = () => {
    if (!svgRef.current) return;
    const svgEl = svgRef.current.cloneNode(true) as SVGSVGElement;
    
    // Add explicitly styling inline for standalone viewing
    svgEl.setAttribute("style", "background-color: #0f172a; color: #f8fafc; font-family: sans-serif;");
    const svgString = new XMLSerializer().serializeToString(svgEl);
    const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement("a");
    a.href = url;
    a.download = `SCADA_Trend_Snapshot_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Snapshot functionality: Export active chart history to CSV
  const downloadSnapshotCSV = () => {
    if (activeHistory.length === 0 || selectedTags.length === 0) return;
    
    let csvContent = "data:text/csv;charset=utf-8,";
    // Header
    csvContent += "Index," + selectedTags.map(tagId => `Tag_${tagId}`).join(",") + "\n";
    
    // Body
    activeHistory.forEach((sample, idx) => {
      const row = [idx];
      selectedTags.forEach(tagId => {
        row.push(sample[tagId] ?? 0.0);
      });
      csvContent += row.join(",") + "\n";
    });

    const encodedUri = encodeURI(csvContent);
    const a = document.createElement("a");
    a.href = encodedUri;
    a.download = `Trend_Data_Export_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const gridLinesY = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      <div className="panel-header" style={{ marginBottom: "0.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span>Trends</span>
          {isPaused && (
            <span
              style={{
                fontSize: "0.7rem",
                color: "var(--color-warning)",
                border: "1px solid var(--color-warning)",
                borderRadius: "3px",
                padding: "0.05rem 0.25rem",
                fontWeight: 600,
                textTransform: "uppercase",
              }}
            >
              Frozen
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: "0.3rem" }}>
          {[10, 30, 60, 120].map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setDuration(t)}
              className="btn"
              style={{
                padding: "0.15rem 0.4rem",
                fontSize: "0.7rem",
                border: "1px solid",
                borderColor: duration === t ? "var(--accent-cyan)" : "var(--panel-border)",
                background: duration === t ? "var(--cell-hover-bg)" : "var(--input-bg)",
                color: duration === t ? "var(--accent-cyan)" : "var(--text-secondary)",
              }}
            >
              {t}s
            </button>
          ))}
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
        {/* Playback Controls */}
        <div style={{ display: "flex", gap: "0.3rem" }}>
          <button
            type="button"
            onClick={() => setIsPaused(!isPaused)}
            className="btn"
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
            title={isPaused ? "Resume Live Stream" : "Pause / Freeze Plot"}
          >
            {isPaused ? <Play size={12} color="var(--color-success)" /> : <Pause size={12} />}
            <span>{isPaused ? "Live" : "Freeze"}</span>
          </button>
          <button
            type="button"
            onClick={resetPlotControls}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Reset Plot Zoom/Pan Settings"
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
          >
            <ZoomIn size={12} />
          </button>
          <button
            type="button"
            onClick={() => setZoomLevel((prev) => Math.min(3.0, prev + 0.15))}
            className="btn"
            style={{ padding: "0.25rem 0.5rem" }}
            title="Zoom Out"
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

        {/* Snapshot Download Buttons */}
        <div style={{ display: "flex", gap: "0.3rem" }}>
          <button
            type="button"
            onClick={downloadSnapshotSVG}
            className="btn"
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.7rem" }}
            title="Download Standalone SVG Graphic"
          >
            <Image size={12} />
            <span>SVG</span>
          </button>
          <button
            type="button"
            onClick={downloadSnapshotCSV}
            className="btn"
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
            title="Download trend view data as CSV"
          >
            <FileSpreadsheet size={12} />
            <span>CSV</span>
          </button>
        </div>
      </div>

      {/* History scroll panning slider */}
      {maxPossibleOffset > 0 && (
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
          <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)", minWidth: "75px" }}>
            Pan History:
          </span>
          <input
            type="range"
            min={0}
            max={maxPossibleOffset}
            value={safeOffset}
            onChange={(e) => setScrollOffset(Number(e.target.value))}
            style={{
              flex: 1,
              cursor: "pointer",
              accentColor: "var(--accent-cyan)",
              height: "4px",
              background: "var(--panel-border)",
            }}
          />
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)", minWidth: "50px", textAlign: "right" }}>
            -{Math.round(safeOffset / 10)}s
          </span>
        </div>
      )}

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
        {Array.from({ length: 32 }).map((_, i) => {
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
            style={{ width: "100%", height: "auto", overflow: "visible" }}
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
              -{Math.round(duration + safeOffset / 10)}s
            </text>
            <text
              x={width - paddingRight}
              y={height - 8}
              fill="var(--text-muted)"
              fontSize="9"
              textAnchor="end"
            >
              -{Math.round(safeOffset / 10)}s
            </text>

            {/* Paths for active tag series */}
            {selectedTags.map((tagId, activeIdx) => {
              const color = LINE_COLORS[activeIdx % LINE_COLORS.length];
              const useSmoothed = smoothedData && smoothedData[tagId] && smoothedData[tagId].length > 0;
              const totalPoints = useSmoothed ? smoothedData[tagId].length : activeHistory.length;

              if (totalPoints < 2) return null;

              // Generate Path commands
              let pathD = "";
              let areaD = `M ${paddingLeft} ${paddingTop + chartHeight} `;

              if (useSmoothed) {
                smoothedData[tagId].forEach((val, sampleIdx) => {
                  const { x, y } = getCoordinates(sampleIdx, val, totalPoints);
                  if (sampleIdx === 0) {
                    pathD += `M ${x} ${y} `;
                  } else {
                    pathD += `L ${x} ${y} `;
                  }
                  areaD += `L ${x} ${y} `;
                });
              } else {
                activeHistory.forEach((sample, sampleIdx) => {
                  const val = sample[tagId] ?? 0;
                  const { x, y } = getCoordinates(sampleIdx, val, totalPoints);
                  if (sampleIdx === 0) {
                    pathD += `M ${x} ${y} `;
                  } else {
                    pathD += `L ${x} ${y} `;
                  }
                  areaD += `L ${x} ${y} `;
                });
              }

              // Close the area path
              const lastX = paddingLeft + chartWidth;
              areaD += `L ${lastX} ${paddingTop + chartHeight} Z`;

              return (
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
              );
            })}
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
