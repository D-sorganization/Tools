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
  | 'Moving Average'
  | 'Butterworth Low-pass'
  | 'Butterworth High-pass'
  | 'Butterworth Band-pass'
  | 'Median Filter'
  | 'Gaussian Filter'
  | 'Hampel Filter'
  | 'Z-Score Filter'
  | 'Savitzky-Golay'
  | 'FFT Low-pass'
  | 'FFT High-pass';

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

export type ExportFormat = 'csv' | 'json' | 'excel';
