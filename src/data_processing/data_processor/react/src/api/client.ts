/**
 * API client for the Data Processor backend.
 */

import axios, { AxiosInstance } from 'axios';
import type {
  DataPreviewResponse,
  ExportFormat,
  ExportResponse,
  FileListResponse,
  FileUploadResponse,
  FilterParameters,
  FilterResponse,
  FilterType,
  SignalListResponse,
  StatisticsResponse,
} from '../types';

const API_BASE_URL = '/api/v1';

/**
 * Create configured axios instance.
 */
function createAxiosInstance(): AxiosInstance {
  return axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
}

const api = createAxiosInstance();

// File operations

export async function loadFile(path: string): Promise<FileUploadResponse> {
  const response = await api.post<FileUploadResponse>('/files/load', { path });
  return response.data;
}

export async function listFiles(): Promise<FileListResponse> {
  const response = await api.get<FileListResponse>('/files');
  return response.data;
}

export async function getSignals(fileId: string): Promise<SignalListResponse> {
  const response = await api.get<SignalListResponse>(`/files/${fileId}/signals`);
  return response.data;
}

export async function deleteFile(fileId: string): Promise<void> {
  await api.delete(`/files/${fileId}`);
}

// Processing operations

export async function applyFilter(
  fileId: string,
  filterType: FilterType,
  signals: string[],
  parameters: FilterParameters
): Promise<FilterResponse> {
  const response = await api.post<FilterResponse>('/processing/filter', {
    file_id: fileId,
    filter_type: filterType,
    signals,
    parameters,
  });
  return response.data;
}

export async function getStatistics(
  fileId: string,
  signals: string[]
): Promise<StatisticsResponse> {
  const response = await api.post<StatisticsResponse>('/processing/statistics', {
    file_id: fileId,
    signals,
  });
  return response.data;
}

export async function getPreview(
  fileId: string,
  signals: string[] = [],
  offset: number = 0,
  limit: number = 100
): Promise<DataPreviewResponse> {
  const response = await api.post<DataPreviewResponse>('/processing/preview', {
    file_id: fileId,
    signals,
    offset,
    limit,
  });
  return response.data;
}

export async function exportData(
  fileId: string,
  format: ExportFormat,
  signals: string[] = [],
  filename?: string
): Promise<ExportResponse> {
  const response = await api.post<ExportResponse>('/processing/export', {
    file_id: fileId,
    format,
    signals,
    filename,
  });
  return response.data;
}

// Health check

export async function checkHealth(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>('/health');
  return response.data;
}

// Export the API client
export const dataProcessorApi = {
  loadFile,
  listFiles,
  getSignals,
  deleteFile,
  applyFilter,
  getStatistics,
  getPreview,
  exportData,
  checkHealth,
};

export default dataProcessorApi;
