import { useCallback, useMemo, useState } from 'react';
import { Activity, Check, Search, Download, Upload } from 'lucide-react';

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
  const [searchTerm, setSearchTerm] = useState('');

  const filteredSignals = useMemo(() => {
    if (!searchTerm.trim()) return signals;
    const term = searchTerm.toLowerCase();
    return signals.filter((s) => s.toLowerCase().includes(term));
  }, [signals, searchTerm]);

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
    onSelectionChange([...filteredSignals]);
  }, [filteredSignals, onSelectionChange]);

  const deselectAll = useCallback(() => {
    onSelectionChange([]);
  }, [onSelectionChange]);

  const saveSignalSet = useCallback(() => {
    if (selectedSignals.length === 0) {
      alert('Please select signals to save.');
      return;
    }
    const signalSet = {
      selected_signals: selectedSignals,
      total_available: signals.length,
    };
    const blob = new Blob([JSON.stringify(signalSet, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'signal_set.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedSignals, signals.length]);

  const loadSignalSet = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const content = event.target?.result as string;
          const signalSet = JSON.parse(content);
          if (signalSet.selected_signals && Array.isArray(signalSet.selected_signals)) {
            const validSignals = signalSet.selected_signals.filter((s: string) =>
              signals.includes(s)
            );
            onSelectionChange(validSignals);
          }
        } catch (err) {
          alert('Invalid signal set file.');
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }, [signals, onSelectionChange]);

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
          <button
            onClick={loadSignalSet}
            className="text-xs text-blue-500 hover:text-blue-400"
            title="Load Signal Set"
          >
            <Upload className="w-3 h-3" />
          </button>
          <button
            onClick={saveSignalSet}
            className="text-xs text-blue-500 hover:text-blue-400"
            title="Save Signal Set"
          >
            <Download className="w-3 h-3" />
          </button>
          <button onClick={selectAll} className="text-xs text-blue-500 hover:text-blue-400">
            All
          </button>
          <button onClick={deselectAll} className="text-xs text-dark-400 hover:text-dark-300">
            None
          </button>
        </div>
      </div>
      <div className="card-body">
        {/* Search field */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search signals..."
            className="w-full pl-9 pr-3 py-2 bg-dark-700 border border-dark-600 rounded-lg text-sm text-dark-100 placeholder-dark-400 focus:outline-none focus:border-blue-500"
          />
        </div>
        {/* Signal list */}
        <div className="max-h-56 overflow-y-auto">
          <div className="space-y-1">
            {filteredSignals.map((signal) => {
              const isSelected = selectedSignals.includes(signal);
              const originalIndex = signals.indexOf(signal);
              const color = SIGNAL_COLORS[originalIndex % SIGNAL_COLORS.length];

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
        {searchTerm && filteredSignals.length === 0 && (
          <div className="text-dark-400 text-center py-4 text-sm">
            No signals match "{searchTerm}"
          </div>
        )}
      </div>
    </div>
  );
}

export default SignalList;
