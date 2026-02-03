import React, { useState, useCallback, useMemo } from 'react';
import { Filter, Play, RotateCcw } from 'lucide-react';
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

export function FilterPanel({ onApply, onReset, disabled }: FilterPanelProps) {
  const [filterType, setFilterType] = useState<FilterType>('Moving Average');
  const [parameters, setParameters] = useState<FilterParameters>(DEFAULT_PARAMETERS);

  const handleParamChange = useCallback((key: keyof FilterParameters, value: number) => {
    setParameters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleApply = useCallback(() => {
    onApply({ type: filterType, parameters });
  }, [filterType, parameters, onApply]);

  const parameterInputs = useMemo(() => {
    switch (filterType) {
      case 'Moving Average':
        return (
          <div>
            <label className="label">Window Size</label>
            <input
              type="number"
              className="input"
              value={parameters.ma_window}
              onChange={(e) => handleParamChange('ma_window', parseInt(e.target.value) || 5)}
              min={1}
              max={100}
            />
          </div>
        );

      case 'Butterworth Low-pass':
      case 'Butterworth High-pass':
        return (
          <>
            <div>
              <label className="label">Order</label>
              <input
                type="number"
                className="input"
                value={parameters.bw_order}
                onChange={(e) => handleParamChange('bw_order', parseInt(e.target.value) || 4)}
                min={1}
                max={10}
              />
            </div>
            <div>
              <label className="label">Cutoff Frequency (0-1)</label>
              <input
                type="number"
                className="input"
                value={parameters.bw_cutoff}
                onChange={(e) => handleParamChange('bw_cutoff', parseFloat(e.target.value) || 0.1)}
                min={0.01}
                max={0.99}
                step={0.01}
              />
            </div>
          </>
        );

      case 'Butterworth Band-pass':
        return (
          <>
            <div>
              <label className="label">Order</label>
              <input
                type="number"
                className="input"
                value={parameters.bw_order}
                onChange={(e) => handleParamChange('bw_order', parseInt(e.target.value) || 4)}
                min={1}
                max={10}
              />
            </div>
            <div>
              <label className="label">Low Cutoff (0-1)</label>
              <input
                type="number"
                className="input"
                value={parameters.bw_low}
                onChange={(e) => handleParamChange('bw_low', parseFloat(e.target.value) || 0.05)}
                min={0.01}
                max={0.99}
                step={0.01}
              />
            </div>
            <div>
              <label className="label">High Cutoff (0-1)</label>
              <input
                type="number"
                className="input"
                value={parameters.bw_high}
                onChange={(e) => handleParamChange('bw_high', parseFloat(e.target.value) || 0.3)}
                min={0.01}
                max={0.99}
                step={0.01}
              />
            </div>
          </>
        );

      case 'Median Filter':
        return (
          <div>
            <label className="label">Kernel Size</label>
            <input
              type="number"
              className="input"
              value={parameters.median_kernel}
              onChange={(e) => handleParamChange('median_kernel', parseInt(e.target.value) || 5)}
              min={3}
              max={51}
              step={2}
            />
          </div>
        );

      case 'Gaussian Filter':
        return (
          <div>
            <label className="label">Sigma</label>
            <input
              type="number"
              className="input"
              value={parameters.gaussian_sigma}
              onChange={(e) => handleParamChange('gaussian_sigma', parseFloat(e.target.value) || 1.0)}
              min={0.1}
              max={10}
              step={0.1}
            />
          </div>
        );

      case 'Hampel Filter':
        return (
          <>
            <div>
              <label className="label">Window Size</label>
              <input
                type="number"
                className="input"
                value={parameters.hampel_window}
                onChange={(e) => handleParamChange('hampel_window', parseInt(e.target.value) || 5)}
                min={3}
                max={51}
              />
            </div>
            <div>
              <label className="label">Threshold</label>
              <input
                type="number"
                className="input"
                value={parameters.hampel_threshold}
                onChange={(e) => handleParamChange('hampel_threshold', parseFloat(e.target.value) || 3.0)}
                min={0.5}
                max={10}
                step={0.1}
              />
            </div>
          </>
        );

      case 'Z-Score Filter':
        return (
          <div>
            <label className="label">Threshold</label>
            <input
              type="number"
              className="input"
              value={parameters.zscore_threshold}
              onChange={(e) => handleParamChange('zscore_threshold', parseFloat(e.target.value) || 3.0)}
              min={1}
              max={10}
              step={0.1}
            />
          </div>
        );

      case 'Savitzky-Golay':
        return (
          <>
            <div>
              <label className="label">Window Size</label>
              <input
                type="number"
                className="input"
                value={parameters.savgol_window}
                onChange={(e) => handleParamChange('savgol_window', parseInt(e.target.value) || 5)}
                min={5}
                max={51}
                step={2}
              />
            </div>
            <div>
              <label className="label">Polynomial Order</label>
              <input
                type="number"
                className="input"
                value={parameters.savgol_polyorder}
                onChange={(e) => handleParamChange('savgol_polyorder', parseInt(e.target.value) || 2)}
                min={1}
                max={5}
              />
            </div>
          </>
        );

      case 'FFT Low-pass':
      case 'FFT High-pass':
        return (
          <div>
            <label className="label">Cutoff Frequency (0-1)</label>
            <input
              type="number"
              className="input"
              value={parameters.fft_cutoff}
              onChange={(e) => handleParamChange('fft_cutoff', parseFloat(e.target.value) || 0.1)}
              min={0.01}
              max={0.99}
              step={0.01}
            />
          </div>
        );

      default:
        return null;
    }
  }, [filterType, parameters, handleParamChange]);

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
            disabled={disabled}
            className="btn btn-primary flex items-center gap-2 flex-1"
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
}

export default FilterPanel;
