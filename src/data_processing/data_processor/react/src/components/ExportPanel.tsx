/**
 * Data export panel component.
 */

import React, { useState, useCallback } from 'react';
import type { ExportFormat } from '../types';

interface ExportPanelProps {
  onExport: (format: ExportFormat) => void;
  isLoading: boolean;
  disabled: boolean;
}

const EXPORT_FORMATS: ExportFormat[] = ['csv', 'excel', 'parquet', 'hdf5', 'feather'];

export function ExportPanel({ onExport, isLoading, disabled }: ExportPanelProps) {
  const [format, setFormat] = useState<ExportFormat>('csv');

  const handleFormatChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setFormat(e.target.value as ExportFormat);
    },
    []
  );

  const handleExport = useCallback(() => {
    onExport(format);
  }, [format, onExport]);

  return (
    <div className="panel">
      <h3 className="panel-title">Export Data</h3>

      <div className="form-group">
        <label className="form-label">Format</label>
        <select
          className="select"
          value={format}
          onChange={handleFormatChange}
          disabled={disabled || isLoading}
        >
          {EXPORT_FORMATS.map((fmt) => (
            <option key={fmt} value={fmt}>
              {fmt.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <button
        className="btn"
        onClick={handleExport}
        disabled={disabled || isLoading}
      >
        {isLoading ? 'Exporting...' : 'Export'}
      </button>
    </div>
  );
}

export default ExportPanel;
