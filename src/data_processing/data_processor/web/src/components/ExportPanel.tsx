import { useCallback, useState } from 'react';
import { Download, FileJson, FileSpreadsheet, FileText } from 'lucide-react';
import Papa from 'papaparse';
import type { DataRow, ExportFormat } from '../types';

interface ExportPanelProps {
  data: DataRow[];
  fileName: string | null;
  disabled: boolean;
}

export function ExportPanel({ data, fileName, disabled }: ExportPanelProps) {
  const [exportFormat, setExportFormat] = useState<ExportFormat>('csv');

  const handleExport = useCallback(() => {
    if (data.length === 0) return;

    const baseName = fileName?.replace(/\.[^/.]+$/, '') || 'data';
    let content: string;
    let mimeType: string;
    let extension: string;

    switch (exportFormat) {
      case 'csv':
        content = Papa.unparse(data);
        mimeType = 'text/csv';
        extension = 'csv';
        break;

      case 'json':
        content = JSON.stringify(data, null, 2);
        mimeType = 'application/json';
        extension = 'json';
        break;

      case 'excel':
        // Export as CSV with .xls extension for basic Excel compatibility
        content = Papa.unparse(data);
        mimeType = 'application/vnd.ms-excel';
        extension = 'xls';
        break;

      default:
        return;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${baseName}_processed.${extension}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [data, fileName, exportFormat]);

  const formatOptions: { value: ExportFormat; label: string; icon: React.ReactNode }[] = [
    { value: 'csv', label: 'CSV', icon: <FileText className="w-4 h-4" /> },
    { value: 'json', label: 'JSON', icon: <FileJson className="w-4 h-4" /> },
    { value: 'excel', label: 'Excel', icon: <FileSpreadsheet className="w-4 h-4" /> },
  ];

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Download className="w-4 h-4" />
        Export Data
      </div>
      <div className="card-body space-y-4">
        <div>
          <label className="label">Format</label>
          <div className="grid grid-cols-3 gap-2">
            {formatOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => setExportFormat(option.value)}
                disabled={disabled}
                className={`
                  flex flex-col items-center gap-1 p-3 rounded-lg border transition-colors
                  ${
                    exportFormat === option.value
                      ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                      : 'border-dark-600 bg-dark-800 text-dark-400 hover:border-dark-500'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
              >
                {option.icon}
                <span className="text-xs">{option.label}</span>
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleExport}
          disabled={disabled || data.length === 0}
          className="btn btn-success w-full flex items-center justify-center gap-2"
        >
          <Download className="w-4 h-4" />
          Export {data.length > 0 ? `(${data.length} rows)` : ''}
        </button>
      </div>
    </div>
  );
}

export default ExportPanel;
