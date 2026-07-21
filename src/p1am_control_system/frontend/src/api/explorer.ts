import { API_BASE, ApiError, apiFetch } from "./client";
import {
  correlationResponseSchema,
  datasetResponseSchema,
  histogramResponseSchema,
  pcaResponseSchema,
  signalListResponseSchema,
  spectrumResponseSchema,
  statisticsResponseSchema,
  trendlineResponseSchema,
  type Column,
  type CorrelationMethod,
  type CorrelationResponse,
  type DatasetRequest,
  type DatasetResponse,
  type ExportRequest,
  type HistogramRequest,
  type HistogramResponse,
  type PcaRequest,
  type PcaResponse,
  type SignalListResponse,
  type SpectrumRequest,
  type SpectrumResponse,
  type StatisticsResponse,
  type TrendlineRequest,
  type TrendlineResponse,
} from "./explorerSchemas";

/**
 * Typed endpoint functions for the Data Explorer API (`/api/explorer/*`).
 *
 * Every read validates its response against the matching zod schema so backend
 * contract drift fails loudly rather than corrupting an analysis.
 */

export function getSignals(): Promise<SignalListResponse> {
  return apiFetch("/explorer/signals", { schema: signalListResponseSchema });
}

export function buildDataset(req: DatasetRequest): Promise<DatasetResponse> {
  return apiFetch("/explorer/dataset", {
    method: "POST",
    json: req,
    schema: datasetResponseSchema,
  });
}

export function getStatistics(columns: Column[]): Promise<StatisticsResponse> {
  return apiFetch("/explorer/statistics", {
    method: "POST",
    json: { columns },
    schema: statisticsResponseSchema,
  });
}

export function getCorrelation(
  columns: Column[],
  method: CorrelationMethod = "pearson",
): Promise<CorrelationResponse> {
  return apiFetch("/explorer/correlation", {
    method: "POST",
    json: { columns, method },
    schema: correlationResponseSchema,
  });
}

export function getSpectrum(req: SpectrumRequest): Promise<SpectrumResponse> {
  return apiFetch("/explorer/spectrum", {
    method: "POST",
    json: req,
    schema: spectrumResponseSchema,
  });
}

export function getTrendline(req: TrendlineRequest): Promise<TrendlineResponse> {
  return apiFetch("/explorer/trendline", {
    method: "POST",
    json: req,
    schema: trendlineResponseSchema,
  });
}

export function getPca(req: PcaRequest): Promise<PcaResponse> {
  return apiFetch("/explorer/pca", {
    method: "POST",
    json: req,
    schema: pcaResponseSchema,
  });
}

export function getHistogram(req: HistogramRequest): Promise<HistogramResponse> {
  return apiFetch("/explorer/histogram", {
    method: "POST",
    json: req,
    schema: histogramResponseSchema,
  });
}

/**
 * Export a processed dataset. Returns the raw file as a Blob (CSV or JSON) so
 * the caller can trigger a browser download. `apiFetch` is JSON-only, so this
 * uses `fetch` directly against the same proxied `/api` base.
 */
export async function exportDataset(req: ExportRequest): Promise<Blob> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/explorer/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch (cause) {
    throw new ApiError(
      "Network error calling /explorer/export",
      0,
      cause instanceof Error ? cause.message : cause,
    );
  }
  if (!res.ok) {
    throw new ApiError(
      `Export failed with status ${res.status}`,
      res.status,
    );
  }
  return res.blob();
}
