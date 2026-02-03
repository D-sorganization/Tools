/**
 * Main Data Processor application component.
 */

import React, { useState, useCallback } from 'react';
import {
  FilePanel,
  SignalPanel,
  FilterPanel,
  PreviewTable,
  ExportPanel,
  StatisticsPanel,
} from './components';
import { useDataProcessor } from './hooks/useDataProcessor';
import type { ExportFormat, FilterParameters, FilterType } from './types';
import './styles/theme.css';

type TabId = 'filter' | 'export' | 'statistics';

export function App() {
  const {
    fileId,
    signals,
    preview,
    statistics,
    isLoading,
    error,
    loadFile,
    applyFilter,
    calculateStatistics,
    exportData,
    clearError,
  } = useDataProcessor();

  const [selectedSignals, setSelectedSignals] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<TabId>('filter');

  const handleLoadFile = useCallback(
    (path: string) => {
      loadFile(path);
      setSelectedSignals([]);
    },
    [loadFile]
  );

  const handleApplyFilter = useCallback(
    (filterType: FilterType, parameters: FilterParameters) => {
      applyFilter(filterType, selectedSignals, parameters);
    },
    [selectedSignals, applyFilter]
  );

  const handleCalculateStatistics = useCallback(() => {
    calculateStatistics(selectedSignals);
  }, [selectedSignals, calculateStatistics]);

  const handleExport = useCallback(
    async (format: ExportFormat) => {
      const response = await exportData(format, selectedSignals);
      if (response?.success) {
        alert(`Data exported to: ${response.filename}`);
      }
    },
    [selectedSignals, exportData]
  );

  const hasData = fileId !== null;

  return (
    <div className="app-container">
      {/* Main content */}
      <div className="main-content">
        {/* Sidebar */}
        <div className="sidebar">
          {/* Error message */}
          {error && (
            <div className="message message-error">
              {error}
              <button
                onClick={clearError}
                style={{
                  float: 'right',
                  background: 'none',
                  border: 'none',
                  color: 'inherit',
                  cursor: 'pointer',
                }}
              >
                x
              </button>
            </div>
          )}

          {/* File panel */}
          <FilePanel onLoad={handleLoadFile} isLoading={isLoading} />

          {/* Signal panel */}
          <SignalPanel
            signals={signals}
            selectedSignals={selectedSignals}
            onSelectionChange={setSelectedSignals}
          />

          {/* Tabs for filter/export/statistics */}
          <div className="tabs">
            <div
              className={`tab ${activeTab === 'filter' ? 'active' : ''}`}
              onClick={() => setActiveTab('filter')}
            >
              Filter
            </div>
            <div
              className={`tab ${activeTab === 'export' ? 'active' : ''}`}
              onClick={() => setActiveTab('export')}
            >
              Export
            </div>
            <div
              className={`tab ${activeTab === 'statistics' ? 'active' : ''}`}
              onClick={() => setActiveTab('statistics')}
            >
              Statistics
            </div>
          </div>

          {/* Tab content */}
          {activeTab === 'filter' && (
            <FilterPanel
              onApply={handleApplyFilter}
              isLoading={isLoading}
              disabled={!hasData}
            />
          )}
          {activeTab === 'export' && (
            <ExportPanel
              onExport={handleExport}
              isLoading={isLoading}
              disabled={!hasData}
            />
          )}
          {activeTab === 'statistics' && (
            <StatisticsPanel
              statistics={statistics}
              onCalculate={handleCalculateStatistics}
              isLoading={isLoading}
              disabled={!hasData}
            />
          )}
        </div>

        {/* Content area */}
        <div className="content-area">
          <PreviewTable preview={preview} />
        </div>
      </div>

      {/* Status bar */}
      <div className="status-bar">
        {isLoading ? (
          <>
            <span className="spinner" style={{ marginRight: 8 }} />
            Processing...
          </>
        ) : hasData ? (
          `Loaded: ${preview?.total_rows ?? 0} rows, ${signals.length} signals | ${selectedSignals.length} selected`
        ) : (
          'Ready - Load a CSV file to begin'
        )}
      </div>
    </div>
  );
}

export default App;
