/**
 * PlotView — Plotly.js-powered chart component replacing Recharts.
 *
 * Accepts either raw DataRow[] + selectedSignals (backward-compatible)
 * or PlotlyData (data + layout) from the Python PlotlyConverter.
 *
 * No 1000-point downsampling limit — Plotly handles large datasets via WebGL.
 */

import { useMemo } from 'react';
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
  /** Chart height in pixels */
  height?: number;
}

// ── Component ───────────────────────────────────────────────────────────────

export function PlotView({
  data,
  selectedSignals,
  plotlyData,
  title = 'Signal Plot',
  height = 400,
}: PlotViewProps) {
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

  // Build layout
  const layout: Partial<PlotlyLayout> = useMemo(() => {
    const base = plotlyData?.layout ?? {};
    return {
      ...DARK_LAYOUT,
      ...base,
      title: base.title ?? title,
      height,
      autosize: true,
      showlegend: traces.length > 0,
    };
  }, [plotlyData, title, height, traces.length]);

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
    <div className="card h-full">
      <div className="card-header flex items-center gap-2">
        <BarChart2 className="w-4 h-4" />
        {title}
      </div>
      <div className="card-body">
        <Plot
          data={traces}
          layout={layout as Record<string, unknown>}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
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
          style={{ width: '100%', height: `${height}px` }}
        />
      </div>
    </div>
  );
}

export default PlotView;
