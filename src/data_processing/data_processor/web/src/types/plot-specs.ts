/**
 * TypeScript interfaces matching the Python PlotSpec Pydantic contracts.
 *
 * These types mirror src/shared/python/plot_engine/specs.py and are used
 * by PlotView.tsx to pass type-safe plot specifications to Plotly.js.
 *
 * When specs.py changes, update these types to match.
 */

// ── Series styling ──────────────────────────────────────────────────────────

export type LineStyle = 'solid' | 'dashed' | 'dotted' | 'dashdot';
export type MarkerShape =
  | 'none'
  | 'circle'
  | 'square'
  | 'triangle'
  | 'diamond'
  | 'cross'
  | 'plus'
  | 'star';
export type DisplayMode = 'line' | 'scatter' | 'line+scatter';
export type TrendlineType = 'linear' | 'polynomial' | 'exponential' | 'power';
export type LegendPosition = 'right' | 'left' | 'top' | 'bottom' | 'none';
export type InterpolationMethod =
  | 'linear'
  | 'cubic'
  | 'nearest'
  | 'multiquadric'
  | 'inverse'
  | 'gaussian';

export interface SeriesStyle {
  color?: string | null;
  line_style: LineStyle;
  line_width: number;
  marker: MarkerShape;
  marker_size: number;
  opacity: number;
  display_mode: DisplayMode;
}

export interface TrendlineSpec {
  type: TrendlineType;
  degree: number;
  show_equation: boolean;
  show_r_squared: boolean;
  color?: string | null;
  line_style: LineStyle;
}

export interface AxisSpec {
  label: string;
  min?: number | null;
  max?: number | null;
  log_scale: boolean;
  grid: boolean;
}

export interface LegendSpec {
  visible: boolean;
  position: LegendPosition;
  labels: Record<string, string>;
}

// ── Series data ─────────────────────────────────────────────────────────────

export interface SeriesData {
  name: string;
  x: number[];
  y: number[];
  style: SeriesStyle;
  trendline?: TrendlineSpec | null;
}

// ── Plot specifications ─────────────────────────────────────────────────────

export interface PlotSpec {
  title: string;
  series: SeriesData[];
  x_axis: AxisSpec;
  y_axis: AxisSpec;
  legend: LegendSpec;
  width: number;
  height: number;
}

export interface SurfacePlotSpec extends PlotSpec {
  z_data: number[][];
  x_grid: number[];
  y_grid: number[];
  z_axis: AxisSpec;
  colormap: string;
  opacity: number;
  show_wireframe: boolean;
  show_scatter: boolean;
  interpolation: InterpolationMethod;
}

export interface ContourPlotSpec extends PlotSpec {
  z_data: number[][];
  x_grid: number[];
  y_grid: number[];
  levels: number;
  filled: boolean;
  colormap: string;
  show_colorbar: boolean;
  show_labels: boolean;
}

export interface HeatmapSpec extends PlotSpec {
  z_data: number[][];
  x_labels: string[];
  y_labels: string[];
  colormap: string;
  annotate: boolean;
  show_colorbar: boolean;
}

export interface HistogramSpec extends PlotSpec {
  bins: number;
  density: boolean;
  cumulative: boolean;
  stacked: boolean;
}

export interface FilterComparisonSpec extends PlotSpec {
  original_series: SeriesData[];
  filtered_series: SeriesData[];
  show_difference: boolean;
  difference_color: string;
}

// ── Plotly.js output format ─────────────────────────────────────────────────

export interface PlotlyTrace {
  type?: string;
  x?: number[] | string[];
  y?: number[] | string[];
  z?: number[][] | number[];
  name?: string;
  mode?: string;
  line?: Record<string, unknown>;
  marker?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PlotlyLayout {
  title?: string | { text: string };
  xaxis?: Record<string, unknown>;
  yaxis?: Record<string, unknown>;
  width?: number;
  height?: number;
  showlegend?: boolean;
  legend?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PlotlyData {
  data: PlotlyTrace[];
  layout: PlotlyLayout;
}

// ── Default factories ───────────────────────────────────────────────────────

export const DEFAULT_SERIES_STYLE: SeriesStyle = {
  color: null,
  line_style: 'solid',
  line_width: 1.5,
  marker: 'none',
  marker_size: 6.0,
  opacity: 1.0,
  display_mode: 'line',
};

export const DEFAULT_AXIS_SPEC: AxisSpec = {
  label: '',
  min: null,
  max: null,
  log_scale: false,
  grid: true,
};

export const DEFAULT_LEGEND_SPEC: LegendSpec = {
  visible: true,
  position: 'right',
  labels: {},
};
