/**
 * Type definitions for the Data Processor API.
 */

// File types
export interface FileInfo {
  filename: string;
  path: string;
  size_bytes: number;
  row_count: number;
  column_count: number;
  loaded_at: string;
}

export interface FileUploadResponse {
  success: boolean;
  file_id: string;
  file_info: FileInfo | null;
  error: string | null;
}

export interface FileListResponse {
  files: FileInfo[];
  total_count: number;
}

export interface SignalInfo {
  name: string;
  dtype: string;
  is_numeric: boolean;
  non_null_count: number;
  min_value: number | null;
  max_value: number | null;
}

export interface SignalListResponse {
  file_id: string;
  signals: SignalInfo[];
  numeric_count: number;
  total_count: number;
}

// Filter types
export type FilterType =
  | 'Moving Average'
  | 'Butterworth Low-pass'
  | 'Butterworth High-pass'
  | 'Median Filter'
  | 'Hampel Filter'
  | 'Z-Score Filter'
  | 'Savitzky-Golay'
  | 'Gaussian Filter'
  | 'FFT Low-pass'
  | 'FFT High-pass';

export interface FilterParameters {
  ma_window?: number;
  bw_order?: number;
  bw_cutoff?: number;
  median_kernel?: number;
  hampel_window?: number;
  hampel_threshold?: number;
  zscore_threshold?: number;
  savgol_window?: number;
  savgol_polyorder?: number;
  gaussian_sigma?: number;
}

export interface FilterRequest {
  file_id: string;
  filter_type: FilterType;
  signals: string[];
  parameters: FilterParameters;
}

export interface FilterResponse {
  success: boolean;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  file_id: string;
  filter_type: string;
  signals_processed: string[];
  row_count: number;
  error: string | null;
}

// Statistics types
export interface SignalStatistics {
  name: string;
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  median: number | null;
  q25: number | null;
  q75: number | null;
}

export interface StatisticsResponse {
  file_id: string;
  statistics: SignalStatistics[];
  error: string | null;
}

// Preview types
export interface DataPreviewResponse {
  file_id: string;
  columns: string[];
  data: (string | number | null)[][];
  total_rows: number;
  offset: number;
  limit: number;
}

// Export types
export type ExportFormat = 'csv' | 'excel' | 'parquet' | 'hdf5' | 'feather';

export interface ExportRequest {
  file_id: string;
  format: ExportFormat;
  signals: string[];
  filename?: string;
}

export interface ExportResponse {
  success: boolean;
  filename: string;
  format: string;
  size_bytes: number;
  error: string | null;
}
