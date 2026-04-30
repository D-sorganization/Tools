import { useState, useCallback, useMemo, memo } from 'react';
import { Filter, Play, RotateCcw, AlertCircle } from 'lucide-react';
import type { FilterConfig, FilterType, FilterParameters } from '../types';

interface FilterPanelProps {
  onApply: (config: FilterConfig) => void;
  onReset: () => void;
  disabled: boolean;
}

const FILTER_TYPES: FilterType[] = [
  'Moving Average',
  'Butterworth Low-pass',
  'Butterworth High-pass',
  'Butterworth Band-pass',
  'Median Filter',
  'Gaussian Filter',
  'Hampel Filter',
  'Z-Score Filter',
  'Savitzky-Golay',
  'FFT Low-pass',
  'FFT High-pass',
];

const DEFAULT_PARAMETERS: FilterParameters = {
  ma_window: 5,
  bw_order: 4,
  bw_cutoff: 0.1,
  bw_low: 0.05,
  bw_high: 0.3,
  median_kernel: 5,
  gaussian_sigma: 1.0,
  hampel_window: 5,
  hampel_threshold: 3.0,
  zscore_threshold: 3.0,
  savgol_window: 5,
  savgol_polyorder: 2,
  fft_cutoff: 0.1,
};

// ⚡ Bolt: Wrapped FilterPanel in React.memo() to prevent unnecessary O(N) re-render
// cascades when parent (App.tsx) UI state changes (like switching tabs).
// Performance impact: Eliminates UI stuttering during tab navigation.
export const FilterPanel = memo(function FilterPanel({ onApply, onReset, disabled }: FilterPanelProps) {
  const [filterType, setFilterType] = useState<FilterType>('Moving Average');
  const [parameters, setParameters] = useState<FilterParameters>(DEFAULT_PARAMETERS);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const validateParameter = useCallback((key: keyof FilterParameters, value: number): string | null => {
    // Validate ranges based on parameter type
    switch (key) {
      case 'ma_window':
        if (value < 1 || value > 100) return 'Window size must be between 1 and 100';
        break;
      case 'bw_order':
        if (value < 1 || value > 10) return 'Order must be between 1 and 10';
        break;
      case 'bw_cutoff':
      case 'bw_low':
      case 'bw_high':
      case 'fft_cutoff':
        if (value < 0.01 || value > 0.99) return 'Frequency must be between 0.01 and 0.99';
        break;
      case 'median_kernel':
        if (value < 3 || value > 51 || value % 2 === 0) return 'Kernel size must be odd and between 3 and 51';
        break;
      case 'gaussian_sigma':
        if (value < 0.1 || value > 10) return 'Sigma must be between 0.1 and 10';
        break;
      case 'hampel_window':
        if (value < 3 || value > 51) return 'Window size must be between 3 and 51';
        break;
      case 'hampel_threshold':
        if (value < 0.5 || value > 10) return 'Threshold must be between 0.5 and 10';
        break;
      case 'zscore_threshold':
        if (value < 1 || value > 10) return 'Threshold must be between 1 and 10';
        break;
      case 'savgol_window':
        if (value < 5 || value > 51 || value % 2 === 0) return 'Window size must be odd and between 5 and 51';
        break;
      case 'savgol_polyorder':
        if (value < 1 || value > 5) return 'Polynomial order must be between 1 and 5';
        break;
    }
    return null;
  }, []);

  const handleParamChange = useCallback((key: keyof FilterParameters, value: number) => {
    const error = validateParameter(key, value);
    setValidationErrors((prev) => {
      if (error) {
        return { ...prev, [key]: error };
      } else {
        const newErrors = { ...prev };
        delete newErrors[key];
        return newErrors;
      }
    });
    setParameters((prev) => ({ ...prev, [key]: value }));
  }, [validateParameter]);

  const handleApply = useCallback(() => {
    const hasErrors = Object.keys(validationErrors).length > 0;
    if (!hasErrors) {
      onApply({ type: filterType, parameters });
    }
  }, [filterType, parameters, onApply, validationErrors]);

  // Helper to render input field with validation error display
  const renderInput = useCallback((label: string, paramKey: keyof FilterParameters, min: number, max: number, step: number = 1) => {
    const error = validationErrors[paramKey];
    return (
      <div>
        <label className="label">{label}</label>
        <input
          type="number"
          className={`input ${error ? 'border-red-500 focus:ring-red-500' : ''}`}
          value={parameters[paramKey] ?? 0}
          onChange={(e) => handleParamChange(paramKey, parseFloat(e.target.value) || (parameters[paramKey] ?? 0))}
          min={min}
          max={max}
          step={step}
        />
        {error && (
          <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
            <AlertCircle className="w-3 h-3" />
            {error}
          </p>
        )}
      </div>
    );
  }, [parameters, validationErrors, handleParamChange]);

  const parameterInputs = useMemo(() => {
    switch (filterType) {
      case 'Moving Average':
        return (
          <div>
            {renderInput('Window Size', 'ma_window', 1, 100, 1)}
          </div>
        );

      case 'Butterworth Low-pass':
      case 'Butterworth High-pass':
        return (
          <>
            {renderInput('Order', 'bw_order', 1, 10, 1)}
            {renderInput('Cutoff Frequency (0-1)', 'bw_cutoff', 0.01, 0.99, 0.01)}
          </>
        );

      case 'Butterworth Band-pass':
        return (
          <>
            {renderInput('Order', 'bw_order', 1, 10, 1)}
            {renderInput('Low Cutoff (0-1)', 'bw_low', 0.01, 0.99, 0.01)}
            {renderInput('High Cutoff (0-1)', 'bw_high', 0.01, 0.99, 0.01)}
          </>
        );

      case 'Median Filter':
        return (
          <div>
            {renderInput('Kernel Size (odd)', 'median_kernel', 3, 51, 2)}
          </div>
        );

      case 'Gaussian Filter':
        return (
          <div>
            {renderInput('Sigma', 'gaussian_sigma', 0.1, 10, 0.1)}
          </div>
        );

      case 'Hampel Filter':
        return (
          <>
            {renderInput('Window Size', 'hampel_window', 3, 51, 1)}
            {renderInput('Threshold', 'hampel_threshold', 0.5, 10, 0.1)}
          </>
        );

      case 'Z-Score Filter':
        return (
          <div>
            {renderInput('Threshold', 'zscore_threshold', 1, 10, 0.1)}
          </div>
        );

      case 'Savitzky-Golay':
        return (
          <>
            {renderInput('Window Size (odd)', 'savgol_window', 5, 51, 2)}
            {renderInput('Polynomial Order', 'savgol_polyorder', 1, 5, 1)}
          </>
        );

      case 'FFT Low-pass':
      case 'FFT High-pass':
        return (
          <div>
            {renderInput('Cutoff Frequency (0-1)', 'fft_cutoff', 0.01, 0.99, 0.01)}
          </div>
        );

      default:
        return null;
    }
  }, [filterType, parameters, handleParamChange, validationErrors, renderInput]);

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Filter className="w-4 h-4" />
        Filter Configuration
      </div>
      <div className="card-body space-y-4">
        <div>
          <label className="label">Filter Type</label>
          <select
            className="select"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value as FilterType)}
            disabled={disabled}
          >
            {FILTER_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        {parameterInputs}

        <div className="flex gap-2 pt-2">
          <button
            onClick={handleApply}
            disabled={disabled || Object.keys(validationErrors).length > 0}
            className="btn btn-primary flex items-center gap-2 flex-1"
            title={Object.keys(validationErrors).length > 0 ? 'Fix validation errors before applying' : ''}
          >
            <Play className="w-4 h-4" />
            Apply
          </button>
          <button
            onClick={onReset}
            disabled={disabled}
            className="btn btn-secondary flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>
    </div>
  );
});

export default FilterPanel;
