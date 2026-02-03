/**
 * Filter configuration panel component.
 */

import React, { useState, useCallback } from 'react';
import type { FilterType, FilterParameters } from '../types';

interface FilterPanelProps {
  onApply: (
    filterType: FilterType,
    parameters: FilterParameters
  ) => void;
  isLoading: boolean;
  disabled: boolean;
}

const FILTER_CONFIGS: Record<FilterType, { params: { key: keyof FilterParameters; label: string; type: 'int' | 'float'; default: number; min: number; max: number }[] }> = {
  'Moving Average': {
    params: [{ key: 'ma_window', label: 'Window Size', type: 'int', default: 10, min: 3, max: 1000 }],
  },
  'Butterworth Low-pass': {
    params: [
      { key: 'bw_order', label: 'Order', type: 'int', default: 3, min: 1, max: 10 },
      { key: 'bw_cutoff', label: 'Cutoff', type: 'float', default: 0.1, min: 0.01, max: 0.99 },
    ],
  },
  'Butterworth High-pass': {
    params: [
      { key: 'bw_order', label: 'Order', type: 'int', default: 3, min: 1, max: 10 },
      { key: 'bw_cutoff', label: 'Cutoff', type: 'float', default: 0.1, min: 0.01, max: 0.99 },
    ],
  },
  'Median Filter': {
    params: [{ key: 'median_kernel', label: 'Kernel Size', type: 'int', default: 5, min: 3, max: 101 }],
  },
  'Hampel Filter': {
    params: [
      { key: 'hampel_window', label: 'Window Size', type: 'int', default: 5, min: 3, max: 100 },
      { key: 'hampel_threshold', label: 'Threshold', type: 'float', default: 3.0, min: 1.0, max: 10.0 },
    ],
  },
  'Z-Score Filter': {
    params: [{ key: 'zscore_threshold', label: 'Threshold', type: 'float', default: 3.0, min: 1.0, max: 10.0 }],
  },
  'Savitzky-Golay': {
    params: [
      { key: 'savgol_window', label: 'Window Size', type: 'int', default: 5, min: 3, max: 101 },
      { key: 'savgol_polyorder', label: 'Poly Order', type: 'int', default: 2, min: 1, max: 6 },
    ],
  },
  'Gaussian Filter': {
    params: [{ key: 'gaussian_sigma', label: 'Sigma', type: 'float', default: 1.0, min: 0.1, max: 100.0 }],
  },
  'FFT Low-pass': { params: [] },
  'FFT High-pass': { params: [] },
};

const FILTER_TYPES = Object.keys(FILTER_CONFIGS) as FilterType[];

export function FilterPanel({ onApply, isLoading, disabled }: FilterPanelProps) {
  const [filterType, setFilterType] = useState<FilterType>('Moving Average');
  const [parameters, setParameters] = useState<FilterParameters>({});

  const handleFilterTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const newType = e.target.value as FilterType;
      setFilterType(newType);
      // Reset parameters to defaults
      const config = FILTER_CONFIGS[newType];
      const defaults: FilterParameters = {};
      config.params.forEach((p) => {
        (defaults as Record<string, number>)[p.key] = p.default;
      });
      setParameters(defaults);
    },
    []
  );

  const handleParamChange = useCallback(
    (key: keyof FilterParameters, value: string) => {
      const numValue = parseFloat(value);
      if (!isNaN(numValue)) {
        setParameters((prev) => ({ ...prev, [key]: numValue }));
      }
    },
    []
  );

  const handleApply = useCallback(() => {
    onApply(filterType, parameters);
  }, [filterType, parameters, onApply]);

  const config = FILTER_CONFIGS[filterType];

  return (
    <div className="panel">
      <h3 className="panel-title">Filter Configuration</h3>

      <div className="form-group">
        <label className="form-label">Filter Type</label>
        <select
          className="select"
          value={filterType}
          onChange={handleFilterTypeChange}
          disabled={disabled || isLoading}
        >
          {FILTER_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </div>

      {config.params.map((param) => (
        <div key={param.key} className="form-group">
          <label className="form-label">{param.label}</label>
          <input
            type="number"
            className="input"
            value={(parameters as Record<string, number>)[param.key] ?? param.default}
            onChange={(e) => handleParamChange(param.key, e.target.value)}
            min={param.min}
            max={param.max}
            step={param.type === 'float' ? 0.01 : 1}
            disabled={disabled || isLoading}
          />
        </div>
      ))}

      <button
        className="btn"
        onClick={handleApply}
        disabled={disabled || isLoading}
      >
        {isLoading ? 'Applying...' : 'Apply Filter'}
      </button>
    </div>
  );
}

export default FilterPanel;
