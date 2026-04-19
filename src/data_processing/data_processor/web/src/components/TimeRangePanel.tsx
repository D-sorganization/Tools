import { useState, useEffect, memo } from 'react';
import { Scissors, Copy } from 'lucide-react';
import type { TimeRangeConfig, DataRow } from '../types';

interface TimeRangePanelProps {
  data: DataRow[];
  timeColumn: string | null;
  disabled: boolean;
  onTrimRange: (config: TimeRangeConfig) => void;
}

// Avoid re-rendering the O(N) range scan when props are unchanged.
export const TimeRangePanel = memo(function TimeRangePanel({
  data,
  timeColumn,
  disabled,
  onTrimRange,
}: TimeRangePanelProps) {
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [dataStart, setDataStart] = useState<string | number | null>(null);
  const [dataEnd, setDataEnd] = useState<string | number | null>(null);

  // Calculate data range when data or time column changes
  useEffect(() => {
    if (!data.length || !timeColumn) {
      setDataStart(null);
      setDataEnd(null);
      return;
    }

    // Use a single pass and avoid Math.min(...array) / Math.max(...array),
    // which can overflow the call stack on large datasets.
    let minVal: number | null = null;
    let maxVal: number | null = null;
    let firstValid: string | number | null = null;
    let lastValid: string | number | null = null;

    for (let i = 0; i < data.length; i++) {
      const val = data[i][timeColumn];
      if (val !== null && val !== undefined) {
        const typedVal = val as string | number;
        if (firstValid === null) {
          firstValid = typedVal;
        }
        lastValid = typedVal;

        if (typeof val === 'number') {
          if (minVal === null || val < minVal) {
            minVal = val;
          }
          if (maxVal === null || val > maxVal) {
            maxVal = val;
          }
        } else {
          // Preserve string/date columns as first and last valid values.
          continue;
        }
      }
    }

    if (firstValid === null) {
      setDataStart(null);
      setDataEnd(null);
      return;
    }

    setDataStart(minVal !== null ? minVal : firstValid);
    setDataEnd(maxVal !== null ? maxVal : lastValid);
  }, [data, timeColumn]);

  const handleTrimRange = () => {
    if (!timeColumn) return;
    onTrimRange({
      startTime: startTime || undefined,
      endTime: endTime || undefined,
      timeColumn,
    });
  };

  const handleCopyRange = () => {
    if (dataStart !== null) setStartTime(String(dataStart));
    if (dataEnd !== null) setEndTime(String(dataEnd));
  };

  const formatValue = (value: string | number | null): string => {
    if (value === null) return '-';
    if (typeof value === 'number') return value.toFixed(4);
    return String(value);
  };

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Scissors className="w-4 h-4" />
        Time Range
      </div>
      <div className="card-body space-y-4">
        {/* Current Data Range */}
        <div className="bg-dark-700/50 rounded-lg p-3 space-y-2">
          <h4 className="text-xs font-medium text-dark-400">Current Data Range</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-dark-500">Start:</span>
              <span className="ml-2 text-dark-200">{formatValue(dataStart)}</span>
            </div>
            <div>
              <span className="text-dark-500">End:</span>
              <span className="ml-2 text-dark-200">{formatValue(dataEnd)}</span>
            </div>
          </div>
          <button
            className="btn btn-secondary btn-sm w-full flex items-center justify-center gap-2"
            onClick={handleCopyRange}
            disabled={disabled || dataStart === null}
          >
            <Copy className="w-3 h-3" />
            Copy to Inputs
          </button>
        </div>

        {/* Time Range Inputs */}
        <div className="space-y-3">
          <div className="space-y-2">
            <label className="block text-xs text-dark-400">Start Time</label>
            <input
              type="text"
              className="input w-full"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              placeholder="e.g., 0.0 or 2024-01-01 10:00:00"
              disabled={disabled}
            />
          </div>

          <div className="space-y-2">
            <label className="block text-xs text-dark-400">End Time</label>
            <input
              type="text"
              className="input w-full"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              placeholder="e.g., 100.0 or 2024-01-01 11:00:00"
              disabled={disabled}
            />
          </div>
        </div>

        <button
          className="btn btn-primary w-full"
          onClick={handleTrimRange}
          disabled={disabled || !timeColumn || (!startTime && !endTime)}
        >
          Trim Data to Range
        </button>

        {!timeColumn && (
          <p className="text-xs text-amber-400">
            No time column detected. Please load data with a time column.
          </p>
        )}
      </div>
    </div>
  );
});

export default TimeRangePanel;
