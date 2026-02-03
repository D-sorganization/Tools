import { useState, useCallback, useMemo } from 'react';
import Papa from 'papaparse';
import type { DataRow, Signal, FilterConfig, Statistics, ProcessingResult } from '../types';

interface DataProcessorState {
  data: DataRow[];
  filteredData: DataRow[];
  signals: string[];
  selectedSignals: string[];
  statistics: Statistics;
  isLoading: boolean;
  error: string | null;
  fileName: string | null;
}

const initialState: DataProcessorState = {
  data: [],
  filteredData: [],
  signals: [],
  selectedSignals: [],
  statistics: {},
  isLoading: false,
  error: null,
  fileName: null,
};

export function useDataProcessor() {
  const [state, setState] = useState<DataProcessorState>(initialState);

  const calculateStatistics = useCallback((data: DataRow[], signals: string[]): Statistics => {
    const stats: Statistics = {};

    for (const signal of signals) {
      const values = data
        .map((row) => row[signal])
        .filter((v): v is number => typeof v === 'number' && !isNaN(v));

      if (values.length === 0) continue;

      const sorted = [...values].sort((a, b) => a - b);
      const sum = values.reduce((a, b) => a + b, 0);
      const mean = sum / values.length;
      const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;

      stats[signal] = {
        mean,
        std: Math.sqrt(variance),
        min: sorted[0],
        max: sorted[sorted.length - 1],
        median: sorted[Math.floor(sorted.length / 2)],
      };
    }

    return stats;
  }, []);

  const loadFile = useCallback(
    async (file: File): Promise<ProcessingResult> => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      return new Promise((resolve) => {
        Papa.parse(file, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const data = results.data as DataRow[];
            const signals = Object.keys(data[0] || {}).filter((key) => {
              const firstValue = data[0]?.[key];
              return typeof firstValue === 'number';
            });

            const statistics = calculateStatistics(data, signals);

            setState((prev) => ({
              ...prev,
              data,
              filteredData: data,
              signals,
              selectedSignals: signals.slice(0, 3),
              statistics,
              isLoading: false,
              fileName: file.name,
            }));

            resolve({ success: true, data, signals });
          },
          error: (error) => {
            const errorMsg = `Failed to parse CSV: ${error.message}`;
            setState((prev) => ({ ...prev, isLoading: false, error: errorMsg }));
            resolve({ success: false, error: errorMsg });
          },
        });
      });
    },
    [calculateStatistics]
  );

  const applyFilter = useCallback(
    (config: FilterConfig): ProcessingResult => {
      try {
        const { data, selectedSignals } = state;
        if (data.length === 0 || selectedSignals.length === 0) {
          return { success: false, error: 'No data or signals selected' };
        }

        const filteredData = data.map((row) => {
          const newRow = { ...row };
          return newRow;
        });

        // Apply filter based on type
        for (const signal of selectedSignals) {
          const values = filteredData.map((row) => row[signal] as number);
          const filtered = applyFilterToSignal(values, config);
          filteredData.forEach((row, i) => {
            row[signal] = filtered[i];
          });
        }

        const statistics = calculateStatistics(filteredData, selectedSignals);

        setState((prev) => ({
          ...prev,
          filteredData,
          statistics,
        }));

        return { success: true, data: filteredData };
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Filter application failed';
        setState((prev) => ({ ...prev, error: errorMsg }));
        return { success: false, error: errorMsg };
      }
    },
    [state, calculateStatistics]
  );

  const setSelectedSignals = useCallback((signals: string[]) => {
    setState((prev) => ({ ...prev, selectedSignals: signals }));
  }, []);

  const resetData = useCallback(() => {
    setState((prev) => ({
      ...prev,
      filteredData: prev.data,
      statistics: calculateStatistics(prev.data, prev.signals),
    }));
  }, [calculateStatistics]);

  const clearAll = useCallback(() => {
    setState(initialState);
  }, []);

  return {
    ...state,
    loadFile,
    applyFilter,
    setSelectedSignals,
    resetData,
    clearAll,
  };
}

// Filter implementations
function applyFilterToSignal(values: number[], config: FilterConfig): number[] {
  switch (config.type) {
    case 'Moving Average':
      return movingAverage(values, config.parameters.ma_window || 5);
    case 'Median Filter':
      return medianFilter(values, config.parameters.median_kernel || 5);
    case 'Gaussian Filter':
      return gaussianFilter(values, config.parameters.gaussian_sigma || 1.0);
    case 'Z-Score Filter':
      return zScoreFilter(values, config.parameters.zscore_threshold || 3.0);
    case 'Savitzky-Golay':
      return savitzkyGolay(
        values,
        config.parameters.savgol_window || 5,
        config.parameters.savgol_polyorder || 2
      );
    default:
      return values;
  }
}

function movingAverage(values: number[], windowSize: number): number[] {
  const result: number[] = [];
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(values.length, i + halfWindow + 1);
    const window = values.slice(start, end);
    const avg = window.reduce((a, b) => a + b, 0) / window.length;
    result.push(avg);
  }

  return result;
}

function medianFilter(values: number[], kernelSize: number): number[] {
  const result: number[] = [];
  const halfKernel = Math.floor(kernelSize / 2);

  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - halfKernel);
    const end = Math.min(values.length, i + halfKernel + 1);
    const window = values.slice(start, end).sort((a, b) => a - b);
    result.push(window[Math.floor(window.length / 2)]);
  }

  return result;
}

function gaussianFilter(values: number[], sigma: number): number[] {
  const kernelSize = Math.ceil(sigma * 6) | 1;
  const halfKernel = Math.floor(kernelSize / 2);
  const kernel: number[] = [];

  let sum = 0;
  for (let i = -halfKernel; i <= halfKernel; i++) {
    const g = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(g);
    sum += g;
  }

  // Normalize kernel
  for (let i = 0; i < kernel.length; i++) {
    kernel[i] /= sum;
  }

  // Apply convolution
  const result: number[] = [];
  for (let i = 0; i < values.length; i++) {
    let val = 0;
    for (let j = 0; j < kernel.length; j++) {
      const idx = Math.min(Math.max(0, i - halfKernel + j), values.length - 1);
      val += values[idx] * kernel[j];
    }
    result.push(val);
  }

  return result;
}

function zScoreFilter(values: number[], threshold: number): number[] {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(
    values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length
  );

  if (std === 0) return values;

  return values.map((v) => {
    const zScore = Math.abs((v - mean) / std);
    return zScore > threshold ? mean : v;
  });
}

function savitzkyGolay(values: number[], windowSize: number, polyOrder: number): number[] {
  // Simplified Savitzky-Golay using moving average as fallback
  // Full implementation would require matrix operations
  if (windowSize < polyOrder + 1) {
    windowSize = polyOrder + 1;
  }

  return movingAverage(values, windowSize);
}

export default useDataProcessor;
