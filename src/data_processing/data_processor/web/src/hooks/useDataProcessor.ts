import { useState, useCallback } from 'react';
import Papa from 'papaparse';
import type {
  DataRow,
  FilterConfig,
  Statistics,
  ProcessingResult,
  IntegrationConfig,
  DifferentiationConfig,
  TimeRangeConfig,
  TrendlineConfig,
  TrendlineResult,
  FormulaConfig,
  PlotConfig,
} from '../types';

interface DataProcessorState {
  data: DataRow[];
  filteredData: DataRow[];
  signals: string[];
  selectedSignals: string[];
  statistics: Statistics;
  isLoading: boolean;
  error: string | null;
  fileName: string | null;
  timeColumn: string | null;
  savedPlotConfigs: Record<string, PlotConfig>;
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
  timeColumn: null,
  savedPlotConfigs: {},
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
            const allColumns = Object.keys(data[0] || {});
            const signals = allColumns.filter((key) => {
              const firstValue = data[0]?.[key];
              return typeof firstValue === 'number';
            });

            // Detect time column
            const timeKeywords = ['time', 'timestamp', 'date', 't', 'seconds', 'datetime'];
            const timeColumn = allColumns.find((col) =>
              timeKeywords.some((kw) => col.toLowerCase().includes(kw))
            ) || (signals.length > 0 ? signals[0] : null);

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
              timeColumn,
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

  // Integration
  const integrateSignals = useCallback(
    (config: IntegrationConfig): ProcessingResult => {
      try {
        const { filteredData, timeColumn } = state;
        if (filteredData.length === 0 || !timeColumn) {
          return { success: false, error: 'No data or time column' };
        }

        const result = filteredData.map((row, i) => {
          const newRow = { ...row };
          if (i === 0) {
            // Initialize cumulative values to 0
            for (const signal of config.signals) {
              newRow[`cumulative_${signal}`] = 0;
            }
          } else {
            const dt = getTimeDelta(row[timeColumn], filteredData[i - 1][timeColumn]);
            for (const signal of config.signals) {
              const prevCum = (filteredData[i - 1] as any)[`cumulative_${signal}`] || 0;
              const y0 = filteredData[i - 1][signal] as number;
              const y1 = row[signal] as number;

              let integral = 0;
              if (config.method === 'trapezoidal') {
                integral = ((y0 + y1) / 2) * dt;
              } else if (config.method === 'rectangular') {
                integral = y0 * dt;
              } else {
                integral = ((y0 + y1) / 2) * dt; // Default to trapezoidal
              }

              newRow[`cumulative_${signal}`] = prevCum + integral;
            }
          }
          return newRow;
        });

        // Update signals list
        const newSignals = [
          ...state.signals,
          ...config.signals.map((s) => `cumulative_${s}`),
        ];

        setState((prev) => ({
          ...prev,
          filteredData: result,
          signals: [...new Set(newSignals)],
        }));

        return { success: true, data: result };
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Integration failed';
        setState((prev) => ({ ...prev, error: errorMsg }));
        return { success: false, error: errorMsg };
      }
    },
    [state]
  );

  // Differentiation
  const differentiateSignals = useCallback(
    (config: DifferentiationConfig): ProcessingResult => {
      try {
        const { filteredData, timeColumn } = state;
        if (filteredData.length === 0 || !timeColumn) {
          return { success: false, error: 'No data or time column' };
        }

        const windowSize = config.windowSize || 11;
        const result = filteredData.map((row, i) => {
          const newRow = { ...row };
          for (const signal of config.signals) {
            const suffix = config.order === 1 ? 'd' : config.order === 2 ? 'd2' : `d${config.order}`;
            const derivName = `${signal}_${suffix}`;

            if (i === 0 || i === filteredData.length - 1) {
              newRow[derivName] = 0;
            } else if (config.method === 'spline') {
              // Simple central difference for spline approximation
              const dt = getTimeDelta(filteredData[i + 1][timeColumn], filteredData[i - 1][timeColumn]);
              const dy = (filteredData[i + 1][signal] as number) - (filteredData[i - 1][signal] as number);
              newRow[derivName] = dy / dt;
            } else {
              // Rolling polynomial using moving average approximation
              const halfWindow = Math.floor(windowSize / 2);
              const start = Math.max(0, i - halfWindow);
              const end = Math.min(filteredData.length, i + halfWindow + 1);

              if (end - start >= 3) {
                const dt = getTimeDelta(filteredData[end - 1][timeColumn], filteredData[start][timeColumn]);
                const dy = (filteredData[end - 1][signal] as number) - (filteredData[start][signal] as number);
                newRow[derivName] = dy / dt;
              } else {
                newRow[derivName] = 0;
              }
            }
          }
          return newRow;
        });

        // Update signals list
        const suffix = config.order === 1 ? 'd' : config.order === 2 ? 'd2' : `d${config.order}`;
        const newSignals = [
          ...state.signals,
          ...config.signals.map((s) => `${s}_${suffix}`),
        ];

        setState((prev) => ({
          ...prev,
          filteredData: result,
          signals: [...new Set(newSignals)],
        }));

        return { success: true, data: result };
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Differentiation failed';
        setState((prev) => ({ ...prev, error: errorMsg }));
        return { success: false, error: errorMsg };
      }
    },
    [state]
  );

  // Time Range Trimming
  const trimTimeRange = useCallback(
    (config: TimeRangeConfig): ProcessingResult => {
      try {
        const { filteredData } = state;
        if (filteredData.length === 0) {
          return { success: false, error: 'No data' };
        }

        let result = [...filteredData];

        if (config.startTime !== undefined) {
          const startVal = typeof config.startTime === 'string'
            ? parseFloat(config.startTime) || config.startTime
            : config.startTime;
          result = result.filter((row) => {
            const time = row[config.timeColumn];
            if (typeof time === 'number' && typeof startVal === 'number') {
              return time >= startVal;
            }
            return String(time) >= String(startVal);
          });
        }

        if (config.endTime !== undefined) {
          const endVal = typeof config.endTime === 'string'
            ? parseFloat(config.endTime) || config.endTime
            : config.endTime;
          result = result.filter((row) => {
            const time = row[config.timeColumn];
            if (typeof time === 'number' && typeof endVal === 'number') {
              return time <= endVal;
            }
            return String(time) <= String(endVal);
          });
        }

        setState((prev) => ({
          ...prev,
          filteredData: result,
          statistics: calculateStatistics(result, prev.selectedSignals),
        }));

        return { success: true, data: result };
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Time range trim failed';
        setState((prev) => ({ ...prev, error: errorMsg }));
        return { success: false, error: errorMsg };
      }
    },
    [state, calculateStatistics]
  );

  // Trendline Calculation
  const calculateTrendline = useCallback(
    (config: TrendlineConfig): TrendlineResult | null => {
      try {
        const { filteredData } = state;
        if (filteredData.length === 0) return null;

        let xData = filteredData.map((row) => row[config.xColumn] as number);
        let yData = filteredData.map((row) => row[config.yColumn] as number);

        // Filter by range if specified
        if (config.xMin !== undefined || config.xMax !== undefined) {
          const filtered = xData.map((x, i) => ({ x, y: yData[i] }))
            .filter(({ x }) => {
              if (config.xMin !== undefined && x < config.xMin) return false;
              if (config.xMax !== undefined && x > config.xMax) return false;
              return true;
            });
          xData = filtered.map((d) => d.x);
          yData = filtered.map((d) => d.y);
        }

        // Filter out NaN values
        const validData = xData.map((x, i) => ({ x, y: yData[i] }))
          .filter(({ x, y }) => !isNaN(x) && !isNaN(y));
        xData = validData.map((d) => d.x);
        yData = validData.map((d) => d.y);

        if (xData.length < 2) return null;

        let result: TrendlineResult;

        if (config.type === 'linear') {
          const { slope, intercept, rSquared } = linearRegression(xData, yData);
          result = {
            type: 'linear',
            equation: `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`,
            rSquared,
            coefficients: [slope, intercept],
          };
        } else if (config.type === 'polynomial') {
          const degree = config.degree || 2;
          const { coefficients, rSquared } = polynomialRegression(xData, yData, degree);
          const terms = coefficients.map((c, i) =>
            i === 0 ? c.toFixed(4) : `${c >= 0 ? '+' : ''}${c.toFixed(4)}x${i > 1 ? `^${i}` : ''}`
          ).reverse().join('');
          result = {
            type: 'polynomial',
            equation: `y = ${terms}`,
            rSquared,
            coefficients,
          };
        } else if (config.type === 'exponential') {
          // y = a * e^(bx), linearize: ln(y) = ln(a) + bx
          const lnY = yData.filter((y) => y > 0).map((y) => Math.log(y));
          const xFiltered = xData.filter((_, i) => yData[i] > 0);
          if (lnY.length < 2) return null;

          const { slope: b, intercept: lnA, rSquared } = linearRegression(xFiltered, lnY);
          const a = Math.exp(lnA);
          result = {
            type: 'exponential',
            equation: `y = ${a.toFixed(4)}e^(${b.toFixed(4)}x)`,
            rSquared,
            coefficients: [a, b],
          };
        } else {
          // Power: y = a * x^b, linearize: ln(y) = ln(a) + b*ln(x)
          const validPower = xData.map((x, i) => ({ x, y: yData[i] }))
            .filter(({ x, y }) => x > 0 && y > 0);
          if (validPower.length < 2) return null;

          const lnX = validPower.map((d) => Math.log(d.x));
          const lnY = validPower.map((d) => Math.log(d.y));

          const { slope: b, intercept: lnA, rSquared } = linearRegression(lnX, lnY);
          const a = Math.exp(lnA);
          result = {
            type: 'power',
            equation: `y = ${a.toFixed(4)}x^${b.toFixed(4)}`,
            rSquared,
            coefficients: [a, b],
          };
        }

        return result;
      } catch (error) {
        console.error('Trendline calculation error:', error);
        return null;
      }
    },
    [state]
  );

  // Custom Formula
  const applyFormula = useCallback(
    (config: FormulaConfig): ProcessingResult => {
      try {
        const { filteredData, signals } = state;
        if (filteredData.length === 0) {
          return { success: false, error: 'No data' };
        }

        // Simple formula evaluation
        const result = filteredData.map((row) => {
          const newRow = { ...row };
          try {
            // Replace signal names with values
            let expr = config.formula;
            for (const signal of signals) {
              const value = row[signal];
              if (typeof value === 'number') {
                expr = expr.replace(new RegExp(`\\b${signal}\\b`, 'g'), String(value));
              }
            }

            // Replace math functions
            expr = expr
              .replace(/\bsqrt\b/g, 'Math.sqrt')
              .replace(/\bsin\b/g, 'Math.sin')
              .replace(/\bcos\b/g, 'Math.cos')
              .replace(/\btan\b/g, 'Math.tan')
              .replace(/\babs\b/g, 'Math.abs')
              .replace(/\blog\b/g, 'Math.log')
              .replace(/\blog10\b/g, 'Math.log10')
              .replace(/\bexp\b/g, 'Math.exp')
              .replace(/\*\*/g, '**');

            // Evaluate (in production, use a proper expression parser)
            const evalResult = Function(`"use strict"; return (${expr})`)();
            newRow[config.name] = typeof evalResult === 'number' ? evalResult : NaN;
          } catch {
            newRow[config.name] = NaN;
          }
          return newRow;
        });

        const newSignals = [...new Set([...signals, config.name])];

        setState((prev) => ({
          ...prev,
          filteredData: result,
          signals: newSignals,
          statistics: calculateStatistics(result, prev.selectedSignals),
        }));

        return { success: true, data: result };
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Formula application failed';
        setState((prev) => ({ ...prev, error: errorMsg }));
        return { success: false, error: errorMsg };
      }
    },
    [state, calculateStatistics]
  );

  // Plot Config Management
  const savePlotConfig = useCallback((config: PlotConfig) => {
    setState((prev) => ({
      ...prev,
      savedPlotConfigs: {
        ...prev.savedPlotConfigs,
        [config.name]: config,
      },
    }));
  }, []);

  const loadPlotConfig = useCallback(
    (name: string): PlotConfig | null => {
      return state.savedPlotConfigs[name] || null;
    },
    [state.savedPlotConfigs]
  );

  return {
    ...state,
    loadFile,
    applyFilter,
    setSelectedSignals,
    resetData,
    clearAll,
    integrateSignals,
    differentiateSignals,
    trimTimeRange,
    calculateTrendline,
    applyFormula,
    savePlotConfig,
    loadPlotConfig,
    savedPlotConfigNames: Object.keys(state.savedPlotConfigs),
  };
}

// Helper function to calculate time delta in seconds
function getTimeDelta(t1: string | number, t2: string | number): number {
  if (typeof t1 === 'number' && typeof t2 === 'number') {
    return t1 - t2;
  }
  // For date strings, calculate difference in seconds
  const d1 = new Date(t1).getTime();
  const d2 = new Date(t2).getTime();
  return (d1 - d2) / 1000;
}

// Linear regression helper
function linearRegression(x: number[], y: number[]): { slope: number; intercept: number; rSquared: number } {
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Calculate R²
  const meanY = sumY / n;
  const ssTotal = y.reduce((sum, yi) => sum + (yi - meanY) ** 2, 0);
  const ssResidual = y.reduce((sum, yi, i) => sum + (yi - (slope * x[i] + intercept)) ** 2, 0);
  const rSquared = 1 - ssResidual / ssTotal;

  return { slope, intercept, rSquared };
}

// Polynomial regression helper (simplified)
function polynomialRegression(x: number[], y: number[], degree: number): { coefficients: number[]; rSquared: number } {
  // For simplicity, use linear regression for degree 1
  if (degree === 1) {
    const { slope, intercept, rSquared } = linearRegression(x, y);
    return { coefficients: [intercept, slope], rSquared };
  }

  // For higher degrees, use a simplified quadratic for degree 2
  // Full implementation would require matrix operations
  if (degree === 2) {
    const n = x.length;
    const x2 = x.map((xi) => xi * xi);
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumX2 = x2.reduce((a, b) => a + b, 0);
    const sumX3 = x.reduce((sum, xi) => sum + xi ** 3, 0);
    const sumX4 = x.reduce((sum, xi) => sum + xi ** 4, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2Y = x2.reduce((sum, xi, i) => sum + xi * y[i], 0);

    // Solve system of equations using Cramer's rule (simplified)
    const a2 = (n * sumX2Y - sumX2 * sumY) / (n * sumX4 - sumX2 * sumX2);
    const a1 = (sumXY - a2 * sumX3) / sumX2;
    const a0 = (sumY - a1 * sumX - a2 * sumX2) / n;

    // Calculate R²
    const meanY = sumY / n;
    const ssTotal = y.reduce((sum, yi) => sum + (yi - meanY) ** 2, 0);
    const ssResidual = y.reduce((sum, yi, i) => sum + (yi - (a0 + a1 * x[i] + a2 * x[i] ** 2)) ** 2, 0);
    const rSquared = 1 - ssResidual / ssTotal;

    return { coefficients: [a0, a1, a2], rSquared };
  }

  // Fallback to linear for higher degrees
  const { slope, intercept, rSquared } = linearRegression(x, y);
  return { coefficients: [intercept, slope], rSquared };
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
