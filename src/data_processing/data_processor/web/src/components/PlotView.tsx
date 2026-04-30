/**
 * PlotView — Plotly.js-powered chart component replacing Recharts.
 *
 * Accepts either raw DataRow[] + selectedSignals (backward-compatible)
 * or PlotlyData (data + layout) from the Python PlotlyConverter.
 *
 * No 1000-point downsampling limit — Plotly handles large datasets via WebGL.
 */

import { useMemo, memo, useRef, useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { BarChart2 } from 'lucide-react';
import type { DataRow } from '../types';
import type { PlotlyData, PlotlyTrace, PlotlyLayout } from '../types/plot-specs';

const CHART_COLORS = [
  '#3b82f6', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
];

// ── Dark theme layout defaults ──────────────────────────────────────────────

const DARK_LAYOUT: Partial<PlotlyLayout> = {
  paper_bgcolor: '#1e1e1e',
  plot_bgcolor: '#252526',
  font: { color: '#d4d4d4', family: 'Segoe UI, sans-serif', size: 12 },
  xaxis: {
    gridcolor: '#343541',
    zerolinecolor: '#4a4a5a',
    tickfont: { color: '#8e8ea0' },
  },
  yaxis: {
    gridcolor: '#343541',
    zerolinecolor: '#4a4a5a',
    tickfont: { color: '#8e8ea0' },
  },
  legend: {
    font: { color: '#d4d4d4' },
    bgcolor: 'rgba(37, 37, 38, 0.8)',
  },
  margin: { t: 40, r: 20, b: 40, l: 60 },
};

// ── Props ───────────────────────────────────────────────────────────────────

interface PlotViewProps {
  /** Raw data rows (backward-compatible mode) */
  data?: DataRow[];
  /** Selected signal names (backward-compatible mode) */
  selectedSignals?: string[];
  /** Pre-computed Plotly data from Python PlotlyConverter */
  plotlyData?: PlotlyData;
  /** Chart title */
  title?: string;
  /** Chart height in pixels (default: 400, min on mobile: 300) */
  height?: number;
  /** Whether to adapt height on mobile (default: true) */
  responsiveHeight?: boolean;
}

// ── Component ───────────────────────────────────────────────────────────────

export const PlotView = memo(function PlotView({
  data,
  selectedSignals,
  plotlyData,
  title = 'Signal Plot',
  height = 400,
  responsiveHeight = true,
}: PlotViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // Track viewport changes for responsive behavior
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    handleResize();
    const resizeObserver = new ResizeObserver(handleResize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    window.addEventListener('resize', handleResize);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Adjust height for mobile devices: reduce on very small screens
  const displayHeight = responsiveHeight && isMobile ? Math.max(250, height - 100) : height;
  // Build traces from raw data (backward-compatible with DataChart)
  const traces: PlotlyTrace[] = useMemo(() => {
    if (plotlyData) return plotlyData.data;

    if (!data || data.length === 0 || !selectedSignals || selectedSignals.length === 0) {
      return [];
    }

    // ⚡ Bolt: Optimize WebGL plotting by pre-allocating Float64Array buffers instead of
    // using array.map() which creates standard JS arrays and causes massive O(N) GC overhead.
    // We use assertions (as unknown as number[]) because Plotly accepts typed arrays at runtime.
    // Performance impact: Eliminates intermediate array allocations, reducing GC pauses and speeding up rendering.
    const len = data.length;
    const xValues = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      xValues[i] = i;
    }

    return selectedSignals.map((signal, i) => {
      const yValues = new Float64Array(len);
      for (let j = 0; j < len; j++) {
        const val = data[j][signal];
        yValues[j] = typeof val === 'number' ? val : NaN;
      }

      return {
        type: 'scattergl' as const,
        x: xValues as unknown as number[],
        y: yValues as unknown as number[],
        name: signal,
        mode: 'lines' as const,
        line: {
          color: CHART_COLORS[i % CHART_COLORS.length],
          width: 1.5,
        },
        hovertemplate: `%{fullData.name}: %{y:.4f}<extra></extra>`,
      };
    });
  }, [data, selectedSignals, plotlyData]);

  // Build layout with responsive margins and legend positioning for mobile
  const layout: Partial<PlotlyLayout> = useMemo(() => {
    const base = plotlyData?.layout ?? {};
    const responsiveMargin = isMobile
      ? { t: 25, r: 8, b: 25, l: 35 }
      : { t: 40, r: 20, b: 40, l: 60 };
    const responsiveLegendOrientation = isMobile
      ? { x: 0, y: -0.15, orientation: 'h' as const, yanchor: 'top' as const, xanchor: 'left' as const }
      : { x: 1.05, y: 1, orientation: 'v' as const };
    return {
      ...DARK_LAYOUT,
      ...base,
      title: base.title ?? title,
      height: displayHeight,
      autosize: true,
      showlegend: traces.length > 0,
      margin: responsiveMargin,
      legend: {
        ...(DARK_LAYOUT.legend || {}),
        ...responsiveLegendOrientation,
      },
    };
  }, [plotlyData, title, displayHeight, traces.length, isMobile]);

  // Empty states
  if (!plotlyData && (!data || data.length === 0)) {
    return (
      <div className="card h-full">
        <div className="card-header flex items-center gap-2">
          <BarChart2 className="w-4 h-4" />
          {title}
        </div>
        <div className="card-body flex items-center justify-center h-64 text-dark-400">
          Load data to visualize signals
        </div>
      </div>
    );
  }

  if (!plotlyData && (!selectedSignals || selectedSignals.length === 0)) {
    return (
      <div className="card h-full">
        <div className="card-header flex items-center gap-2">
          <BarChart2 className="w-4 h-4" />
          {title}
        </div>
        <div className="card-body flex items-center justify-center h-64 text-dark-400">
          Select signals to display
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="card h-full w-full">
      <div className="card-header flex items-center gap-2">
        <BarChart2 className="w-4 h-4 flex-shrink-0" />
        <span className="truncate">{title}</span>
      </div>
      <div className="card-body overflow-x-auto">
        <Plot
          data={traces}
          layout={layout as Record<string, unknown>}
          config={{
            responsive: true,
            displayModeBar: !isMobile,
            modeBarButtonsToRemove: isMobile
              ? ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'resetScale2d']
              : ['lasso2d', 'select2d'],
            displaylogo: false,
            toImageButtonOptions: {
              format: 'png',
              filename: title.replace(/\s+/g, '_'),
              height: 800,
              width: 1200,
              scale: 2,
            },
          }}
          useResizeHandler
          style={{
            width: '100%',
            height: `${displayHeight}px`,
            minHeight: '300px',
          }}
        />
      </div>
    </div>
  );
});

export default PlotView;
