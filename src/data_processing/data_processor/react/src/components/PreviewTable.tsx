/**
 * Data preview table component.
 */

import React from 'react';
import type { DataPreviewResponse } from '../types';

interface PreviewTableProps {
  preview: DataPreviewResponse | null;
}

export function PreviewTable({ preview }: PreviewTableProps) {
  if (!preview) {
    return (
      <div className="panel">
        <h3 className="panel-title">Data Preview</h3>
        <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: 20 }}>
          Load a file to see data preview
        </div>
      </div>
    );
  }

  const formatValue = (value: string | number | null): string => {
    if (value === null || value === undefined) {
      return '';
    }
    if (typeof value === 'number') {
      return Number.isInteger(value) ? value.toString() : value.toFixed(6);
    }
    return String(value);
  };

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 className="panel-title">
        Data Preview
        <span style={{ fontWeight: 'normal', marginLeft: 8, color: 'var(--text-secondary)' }}>
          Showing {preview.data.length} of {preview.total_rows} rows
        </span>
      </h3>

      <div className="table-container" style={{ flex: 1 }}>
        <table>
          <thead>
            <tr>
              <th>#</th>
              {preview.columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.data.map((row, rowIdx) => (
              <tr key={rowIdx}>
                <td style={{ color: 'var(--text-secondary)' }}>
                  {preview.offset + rowIdx + 1}
                </td>
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx}>{formatValue(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default PreviewTable;
