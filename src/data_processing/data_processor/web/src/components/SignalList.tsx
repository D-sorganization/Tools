import React, { useCallback } from 'react';
import { Activity, Check } from 'lucide-react';

interface SignalListProps {
  signals: string[];
  selectedSignals: string[];
  onSelectionChange: (signals: string[]) => void;
}

const SIGNAL_COLORS = [
  '#3b82f6', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
];

export function SignalList({ signals, selectedSignals, onSelectionChange }: SignalListProps) {
  const toggleSignal = useCallback(
    (signal: string) => {
      if (selectedSignals.includes(signal)) {
        onSelectionChange(selectedSignals.filter((s) => s !== signal));
      } else {
        onSelectionChange([...selectedSignals, signal]);
      }
    },
    [selectedSignals, onSelectionChange]
  );

  const selectAll = useCallback(() => {
    onSelectionChange([...signals]);
  }, [signals, onSelectionChange]);

  const deselectAll = useCallback(() => {
    onSelectionChange([]);
  }, [onSelectionChange]);

  if (signals.length === 0) {
    return (
      <div className="card">
        <div className="card-header flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Signals
        </div>
        <div className="card-body text-dark-400 text-center py-8">
          Load a CSV file to see available signals
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Signals ({selectedSignals.length}/{signals.length})
        </div>
        <div className="flex gap-2">
          <button onClick={selectAll} className="text-xs text-blue-500 hover:text-blue-400">
            All
          </button>
          <button onClick={deselectAll} className="text-xs text-dark-400 hover:text-dark-300">
            None
          </button>
        </div>
      </div>
      <div className="card-body max-h-64 overflow-y-auto">
        <div className="space-y-1">
          {signals.map((signal, index) => {
            const isSelected = selectedSignals.includes(signal);
            const color = SIGNAL_COLORS[index % SIGNAL_COLORS.length];

            return (
              <button
                key={signal}
                onClick={() => toggleSignal(signal)}
                className={`
                  w-full flex items-center gap-2 px-3 py-2 rounded-lg
                  text-left text-sm transition-colors duration-150
                  ${isSelected ? 'bg-dark-700' : 'hover:bg-dark-700/50'}
                `}
              >
                <div
                  className={`
                    w-4 h-4 rounded border-2 flex items-center justify-center
                    ${isSelected ? 'border-transparent' : 'border-dark-500'}
                  `}
                  style={{ backgroundColor: isSelected ? color : 'transparent' }}
                >
                  {isSelected && <Check className="w-3 h-3 text-white" />}
                </div>
                <span className={isSelected ? 'text-dark-100' : 'text-dark-400'}>
                  {signal}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default SignalList;
