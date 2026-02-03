/**
 * Signal selection panel component.
 */

import React, { useCallback } from 'react';
import type { SignalInfo } from '../types';

interface SignalPanelProps {
  signals: SignalInfo[];
  selectedSignals: string[];
  onSelectionChange: (signals: string[]) => void;
}

export function SignalPanel({
  signals,
  selectedSignals,
  onSelectionChange,
}: SignalPanelProps) {
  const handleToggle = useCallback(
    (signalName: string) => {
      const isSelected = selectedSignals.includes(signalName);
      if (isSelected) {
        onSelectionChange(selectedSignals.filter((s) => s !== signalName));
      } else {
        onSelectionChange([...selectedSignals, signalName]);
      }
    },
    [selectedSignals, onSelectionChange]
  );

  const handleSelectAll = useCallback(() => {
    const numericSignals = signals
      .filter((s) => s.is_numeric)
      .map((s) => s.name);
    onSelectionChange(numericSignals);
  }, [signals, onSelectionChange]);

  const handleClearSelection = useCallback(() => {
    onSelectionChange([]);
  }, [onSelectionChange]);

  const numericSignals = signals.filter((s) => s.is_numeric);

  return (
    <div className="panel">
      <h3 className="panel-title">Signal Selection</h3>

      <div className="btn-row">
        <button className="btn btn-secondary" onClick={handleSelectAll}>
          Select All
        </button>
        <button className="btn btn-secondary" onClick={handleClearSelection}>
          Clear
        </button>
      </div>

      <div className="list-container">
        {numericSignals.length === 0 ? (
          <div className="list-item" style={{ color: 'var(--text-secondary)' }}>
            No signals available
          </div>
        ) : (
          numericSignals.map((signal) => (
            <div
              key={signal.name}
              className={`list-item ${
                selectedSignals.includes(signal.name) ? 'selected' : ''
              }`}
              onClick={() => handleToggle(signal.name)}
            >
              {signal.name}
              <span style={{ color: 'var(--text-secondary)', marginLeft: 8 }}>
                ({signal.dtype})
              </span>
            </div>
          ))
        )}
      </div>

      <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: 12 }}>
        {selectedSignals.length} of {numericSignals.length} selected
      </div>
    </div>
  );
}

export default SignalPanel;
