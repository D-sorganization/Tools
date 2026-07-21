import { z } from "zod";

/**
 * Runtime contracts + request types for the Data Explorer API.
 *
 * Mirrors the backend pydantic models in `backend/data_explorer_models.py` and
 * the enums in `backend/data_explorer_enums.py`. Response schemas are parsed in
 * {@link apiFetch} so a backend contract drift fails loudly; request shapes are
 * plain TypeScript types (no runtime validation needed on the way out).
 */

// --- Enums (string-literal unions matching the backend StrEnums) -------------

export const FILTER_TYPES = [
  "moving_average",
  "exponential",
  "median",
  "gaussian",
  "savgol",
  "hampel",
  "zscore",
  "fft_lowpass",
  "fft_highpass",
  "fft_bandpass",
  "integrate",
  "differentiate",
] as const;
export type FilterType = (typeof FILTER_TYPES)[number];

export const AGG_METHODS = [
  "mean",
  "median",
  "first",
  "last",
  "min",
  "max",
  "sum",
] as const;
export type AggMethod = (typeof AGG_METHODS)[number];

export const CORRELATION_METHODS = ["pearson", "spearman"] as const;
export type CorrelationMethod = (typeof CORRELATION_METHODS)[number];

export const SPECTRUM_METHODS = ["fft", "welch"] as const;
export type SpectrumMethod = (typeof SPECTRUM_METHODS)[number];

export const WINDOW_KINDS = ["none", "hanning", "hamming", "blackman"] as const;
export type WindowKind = (typeof WINDOW_KINDS)[number];

export const TRENDLINE_KINDS = [
  "linear",
  "polynomial",
  "exponential",
  "power",
] as const;
export type TrendlineKind = (typeof TRENDLINE_KINDS)[number];

export const EXPORT_FORMATS = ["csv", "json"] as const;
export type ExportFormat = (typeof EXPORT_FORMATS)[number];

// --- Core dataset shape ------------------------------------------------------

/** A numeric sample may be `null` to mark a gap / non-finite value. */
export const sampleSchema = z.number().nullable();

export const columnSchema = z.object({
  name: z.string(),
  values: z.array(sampleSchema),
});
export type Column = z.infer<typeof columnSchema>;

export const datasetResponseSchema = z.object({
  index: z.array(z.number()),
  columns: z.array(columnSchema),
  row_count: z.number(),
  truncated: z.boolean(),
  sample_rate_hz: z.number().nullable(),
});
export type DatasetResponse = z.infer<typeof datasetResponseSchema>;

export const signalInfoSchema = z.object({
  name: z.string(),
  count: z.number(),
  start_time: z.string().nullable(),
  end_time: z.string().nullable(),
});
export type SignalInfo = z.infer<typeof signalInfoSchema>;

export const signalListResponseSchema = z.object({
  signals: z.array(signalInfoSchema),
});
export type SignalListResponse = z.infer<typeof signalListResponseSchema>;

// --- Analysis responses ------------------------------------------------------

export const columnStatisticsSchema = z.object({
  name: z.string(),
  count: z.number(),
  mean: z.number(),
  std: z.number(),
  min: z.number(),
  max: z.number(),
  median: z.number(),
  p25: z.number(),
  p75: z.number(),
  rms: z.number(),
});
export type ColumnStatistics = z.infer<typeof columnStatisticsSchema>;

export const statisticsResponseSchema = z.object({
  stats: z.array(columnStatisticsSchema),
});
export type StatisticsResponse = z.infer<typeof statisticsResponseSchema>;

export const correlationResponseSchema = z.object({
  labels: z.array(z.string()),
  matrix: z.array(z.array(z.number())),
  method: z.enum(CORRELATION_METHODS),
});
export type CorrelationResponse = z.infer<typeof correlationResponseSchema>;

export const spectrumResponseSchema = z.object({
  freqs: z.array(z.number()),
  power: z.array(z.number()),
  method: z.enum(SPECTRUM_METHODS),
});
export type SpectrumResponse = z.infer<typeof spectrumResponseSchema>;

export const trendlineResponseSchema = z.object({
  kind: z.enum(TRENDLINE_KINDS),
  coefficients: z.array(z.number()),
  equation: z.string(),
  r_squared: z.number(),
  x_fit: z.array(z.number()),
  y_fit: z.array(z.number()),
});
export type TrendlineResponse = z.infer<typeof trendlineResponseSchema>;

export const pcaResponseSchema = z.object({
  explained_variance_ratio: z.array(z.number()),
  cumulative_variance: z.array(z.number()),
  singular_values: z.array(z.number()),
  component_labels: z.array(z.string()),
  loadings: z.array(z.array(z.number())),
  scores_pc1: z.array(z.number()),
  scores_pc2: z.array(z.number()),
});
export type PcaResponse = z.infer<typeof pcaResponseSchema>;

export const histogramResponseSchema = z.object({
  bin_edges: z.array(z.number()),
  counts: z.array(z.number()),
});
export type HistogramResponse = z.infer<typeof histogramResponseSchema>;

// --- Request shapes (sent to the backend; no runtime validation needed) ------

export interface FilterSpec {
  target: string;
  type: FilterType;
  params: Record<string, number>;
  output?: string | null;
}

export interface DerivedColumn {
  name: string;
  expression: string;
}

export interface ResampleSpec {
  interval_s: number;
  agg: AggMethod;
  interpolate: boolean;
}

export interface TrimSpec {
  start_ms?: number | null;
  end_ms?: number | null;
}

export interface HistorianSource {
  tags: string[];
  start_time: string;
  end_time: string;
  max_points?: number;
}

export interface InlineData {
  index: number[];
  columns: Column[];
}

export interface DatasetRequest {
  historian?: HistorianSource;
  inline?: InlineData;
  resample?: ResampleSpec;
  filters?: FilterSpec[];
  derived?: DerivedColumn[];
  trim?: TrimSpec;
  max_points?: number;
}

export interface SpectrumRequest {
  values: Array<number | null>;
  sample_rate_hz: number;
  method?: SpectrumMethod;
  window?: WindowKind;
  segment_size?: number | null;
  detrend?: boolean;
}

export interface TrendlineRequest {
  x: Array<number | null>;
  y: Array<number | null>;
  kind?: TrendlineKind;
  degree?: number;
  num_points?: number;
}

export interface PcaRequest {
  columns: Column[];
  standardize?: boolean;
  n_components?: number;
}

export interface HistogramRequest {
  values: Array<number | null>;
  bins?: number;
  density?: boolean;
}

export interface ExportRequest {
  index?: number[] | null;
  columns: Column[];
  format?: ExportFormat;
  filename?: string | null;
}
