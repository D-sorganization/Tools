import { useState } from 'react';
import { Clock } from 'lucide-react';
import type { ResampleMethod, ResampleConfig } from '../types';

interface ResamplePanelProps {
  timeColumn: string | null;
  disabled: boolean;
  onResample: (config: ResampleConfig) => void;
}

const frequencyOptions = [
  '100ms',
  '250ms',
  '500ms',
  '1s',
  '2s',
  '5s',
  '10s',
  '30s',
  '1min',
  '5min',
  '10min',
  '15min',
  '30min',
  '1h',
];

export function ResamplePanel({
  timeColumn,
  disabled,
  onResample,
}: ResamplePanelProps) {
  const [targetFrequency, setTargetFrequency] = useState('1s');
  const [customFrequency, setCustomFrequency] = useState('');
  const [method, setMethod] = useState<ResampleMethod>('mean');
  const [interpolate, setInterpolate] = useState(true);

  const handleResample = () => {
    if (!timeColumn) return;

    const freq = customFrequency || targetFrequency;
    onResample({
      targetFrequency: freq,
      method,
      interpolate,
      timeColumn,
    });
  };

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Clock className="w-4 h-4" />
        Time Resampling
      </div>
      <div className="card-body space-y-4">
        <div className="space-y-2">
          <label className="block text-xs text-dark-400">Target Frequency</label>
          <select
            className="select w-full"
            value={targetFrequency}
            onChange={(e) => setTargetFrequency(e.target.value)}
            disabled={disabled}
          >
            {frequencyOptions.map((freq) => (
              <option key={freq} value={freq}>
                {freq}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-xs text-dark-400">Custom Frequency</label>
          <input
            type="text"
            className="input w-full"
            value={customFrequency}
            onChange={(e) => setCustomFrequency(e.target.value)}
            placeholder="e.g., 2.5s, 100ms"
            disabled={disabled}
          />
          <p className="text-xs text-dark-500">Leave empty to use selection above</p>
        </div>

        <div className="space-y-2">
          <label className="block text-xs text-dark-400">Aggregation Method</label>
          <select
            className="select w-full"
            value={method}
            onChange={(e) => setMethod(e.target.value as ResampleMethod)}
            disabled={disabled}
          >
            <option value="mean">Mean</option>
            <option value="median">Median</option>
            <option value="first">First</option>
            <option value="last">Last</option>
            <option value="min">Minimum</option>
            <option value="max">Maximum</option>
            <option value="sum">Sum</option>
          </select>
        </div>

        <label className="flex items-center gap-2 text-sm text-dark-300">
          <input
            type="checkbox"
            className="checkbox"
            checked={interpolate}
            onChange={(e) => setInterpolate(e.target.checked)}
            disabled={disabled}
          />
          Interpolate missing values
        </label>

        <button
          className="btn btn-primary w-full"
          onClick={handleResample}
          disabled={disabled || !timeColumn}
        >
          Apply Resampling
        </button>

        {!timeColumn && (
          <p className="text-xs text-amber-400">
            No time column detected. Please load data with a time column.
          </p>
        )}
      </div>
    </div>
  );
}

export default ResamplePanel;
