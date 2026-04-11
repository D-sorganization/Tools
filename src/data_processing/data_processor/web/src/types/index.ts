/**
 * Type definitions for Data Processor Web UI
 */

export interface DataRow {
  [key: string]: number | string;
}

export interface Signal {
  name: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  median: number;
}

export interface FilterConfig {
  type: FilterType;
  parameters: FilterParameters;
}

export type FilterType =
  | "Moving Average"
  | "Butterworth Low-pass"
  | "Butterworth High-pass"
  | "Butterworth Band-pass"
  | "Median Filter"
  | "Gaussian Filter"
  | "Hampel Filter"
  | "Z-Score Filter"
  | "Savitzky-Golay"
  | "FFT Low-pass"
  | "FFT High-pass";

export interface FilterParameters {
  ma_window?: number;
  bw_order?: number;
  bw_cutoff?: number;
  bw_low?: number;
  bw_high?: number;
  median_kernel?: number;
  gaussian_sigma?: number;
  hampel_window?: number;
  hampel_threshold?: number;
  zscore_threshold?: number;
  savgol_window?: number;
  savgol_polyorder?: number;
  fft_cutoff?: number;
}

export interface ProcessingResult {
  success: boolean;
  data?: DataRow[];
  error?: string;
  signals?: string[];
}

export interface Statistics {
  [signal: string]: {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
  };
}

export type ExportFormat = "csv" | "json" | "excel";

// Integration types
export type IntegrationMethod = "trapezoidal" | "simpson" | "rectangular";

export interface IntegrationConfig {
  method: IntegrationMethod;
  signals: string[];
  timeColumn: string;
}

// Differentiation types
export type DifferentiationMethod = "spline" | "rolling_polynomial";

export interface DifferentiationConfig {
  method: DifferentiationMethod;
  signals: string[];
  timeColumn: string;
  order: number;
  windowSize?: number;
  polyOrder?: number;
}

// Resampling types
export type ResampleMethod =
  | "mean"
  | "median"
  | "first"
  | "last"
  | "min"
  | "max"
  | "sum";

export interface ResampleConfig {
  targetFrequency: string;
  method: ResampleMethod;
  interpolate: boolean;
  timeColumn: string;
}

// Time range types
export interface TimeRangeConfig {
  startTime?: number | string;
  endTime?: number | string;
  timeColumn: string;
}

// Trendline types
export type TrendlineType = "linear" | "polynomial" | "exponential" | "power";

export interface TrendlineConfig {
  type: TrendlineType;
  xColumn: string;
  yColumn: string;
  degree?: number;
  xMin?: number;
  xMax?: number;
}

export interface TrendlineResult {
  type: TrendlineType;
  equation: string;
  rSquared: number;
  coefficients: number[];
  predictions?: number[];
}

// Custom formula types
export interface FormulaConfig {
  name: string;
  formula: string;
}

// Plot configuration types
export interface PlotConfig {
  name: string;
  xAxis: string;
  ySignals: string[];
  trendline?: TrendlineConfig;
  colors?: Record<string, string>;
}

// App configuration types
export interface AppConfig {
  outputDirectory?: string;
  exportFormat?: ExportFormat;
  includeTimestamp?: boolean;
  resampleRule?: string;
  resampleMethod?: ResampleMethod;
  integrationMethod?: IntegrationMethod;
  differentiationMethod?: DifferentiationMethod;
}

// ---------------------------------------------------------------------------
// Advanced analytics types -- See issue #607
// ---------------------------------------------------------------------------

/** Correlation matrix between selected signals. */
export interface CorrelationMatrix {
  signals: string[];
  matrix: number[][];
}

/** Result of a PCA (Principal Component Analysis). */
export interface PCAResult {
  /** Proportion of variance explained by each component. */
  explainedVariance: number[];
  /** Cumulative variance explained. */
  cumulativeVariance: number[];
  /** Number of components retained. */
  numComponents: number;
  /** PCA scores for each data row (rows x numComponents). */
  scores: number[][];
  /** Loading vectors (signals x numComponents). */
  loadings: number[][];
  /** Signal names in the same order as loadings rows. */
  signals: string[];
}

/** Result of a regression analysis. */
export interface RegressionResult {
  type: "linear" | "polynomial";
  equation: string;
  rSquared: number;
  adjustedRSquared: number;
  coefficients: number[];
  residuals: number[];
  predictions: number[];
  xSignal: string;
  ySignal: string;
}
