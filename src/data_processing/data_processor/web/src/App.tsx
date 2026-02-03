import React, { useCallback, useState } from 'react';
import { Database, Settings, BarChart3, Table } from 'lucide-react';
import {
  FileUpload,
  SignalList,
  FilterPanel,
  DataChart,
  StatisticsPanel,
  ExportPanel,
} from './components';
import { useDataProcessor } from './hooks';
import type { FilterConfig } from './types';

type TabType = 'chart' | 'table';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('chart');
  const {
    data,
    filteredData,
    signals,
    selectedSignals,
    statistics,
    isLoading,
    error,
    fileName,
    loadFile,
    applyFilter,
    setSelectedSignals,
    resetData,
    clearAll,
  } = useDataProcessor();

  const handleFileSelect = useCallback(
    async (file: File) => {
      await loadFile(file);
    },
    [loadFile]
  );

  const handleApplyFilter = useCallback(
    (config: FilterConfig) => {
      applyFilter(config);
    },
    [applyFilter]
  );

  const hasData = data.length > 0;

  return (
    <div className="min-h-screen bg-dark-900 text-dark-100">
      {/* Header */}
      <header className="bg-dark-800 border-b border-dark-700 px-6 py-4">
        <div className="flex items-center justify-between max-w-screen-2xl mx-auto">
          <div className="flex items-center gap-3">
            <Database className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-xl font-bold">Data Processor</h1>
              <p className="text-sm text-dark-400">Signal processing and analysis</p>
            </div>
          </div>
          {fileName && (
            <div className="flex items-center gap-2 text-sm text-dark-400">
              <Settings className="w-4 h-4" />
              {data.length.toLocaleString()} rows · {signals.length} signals
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-screen-2xl mx-auto p-6">
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <aside className="col-span-12 lg:col-span-3 space-y-6">
            <FileUpload
              onFileSelect={handleFileSelect}
              fileName={fileName}
              onClear={clearAll}
              isLoading={isLoading}
            />

            <SignalList
              signals={signals}
              selectedSignals={selectedSignals}
              onSelectionChange={setSelectedSignals}
            />

            <FilterPanel
              onApply={handleApplyFilter}
              onReset={resetData}
              disabled={!hasData}
            />
          </aside>

          {/* Main Content Area */}
          <div className="col-span-12 lg:col-span-6 space-y-6">
            {/* Tabs */}
            <div className="flex border-b border-dark-700">
              <button
                onClick={() => setActiveTab('chart')}
                className={`tab ${activeTab === 'chart' ? 'tab-active' : ''}`}
              >
                <span className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Chart View
                </span>
              </button>
              <button
                onClick={() => setActiveTab('table')}
                className={`tab ${activeTab === 'table' ? 'tab-active' : ''}`}
              >
                <span className="flex items-center gap-2">
                  <Table className="w-4 h-4" />
                  Table View
                </span>
              </button>
            </div>

            {/* Tab Content */}
            {activeTab === 'chart' ? (
              <DataChart
                data={filteredData}
                selectedSignals={selectedSignals}
                title="Filtered Data"
              />
            ) : (
              <DataTableView data={filteredData} selectedSignals={selectedSignals} />
            )}

            {/* Original vs Filtered Comparison */}
            {hasData && filteredData !== data && (
              <DataChart
                data={data}
                selectedSignals={selectedSignals}
                title="Original Data"
              />
            )}
          </div>

          {/* Right Sidebar */}
          <aside className="col-span-12 lg:col-span-3 space-y-6">
            <StatisticsPanel
              statistics={statistics}
              selectedSignals={selectedSignals}
            />

            <ExportPanel
              data={filteredData}
              fileName={fileName}
              disabled={!hasData}
            />
          </aside>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-dark-800 border-t border-dark-700 px-6 py-4 mt-8">
        <div className="max-w-screen-2xl mx-auto text-center text-sm text-dark-500">
          Data Processor v1.0 · Built with React + TypeScript
        </div>
      </footer>
    </div>
  );
}

// Simple table view component
interface DataTableViewProps {
  data: Array<Record<string, number | string>>;
  selectedSignals: string[];
}

function DataTableView({ data, selectedSignals }: DataTableViewProps) {
  if (data.length === 0 || selectedSignals.length === 0) {
    return (
      <div className="card">
        <div className="card-header flex items-center gap-2">
          <Table className="w-4 h-4" />
          Data Table
        </div>
        <div className="card-body flex items-center justify-center h-64 text-dark-400">
          {data.length === 0 ? 'Load data to view table' : 'Select signals to display'}
        </div>
      </div>
    );
  }

  // Show first 100 rows
  const displayData = data.slice(0, 100);

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Table className="w-4 h-4" />
          Data Table
        </div>
        <span className="text-xs text-dark-400">
          Showing {displayData.length} of {data.length} rows
        </span>
      </div>
      <div className="card-body overflow-x-auto max-h-96">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-dark-800">
            <tr className="text-dark-400 border-b border-dark-700">
              <th className="text-left py-2 px-3">#</th>
              {selectedSignals.map((signal) => (
                <th key={signal} className="text-right py-2 px-3">
                  {signal}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayData.map((row, index) => (
              <tr key={index} className="border-b border-dark-800 hover:bg-dark-700/30">
                <td className="py-1 px-3 text-dark-500">{index + 1}</td>
                {selectedSignals.map((signal) => (
                  <td key={signal} className="text-right py-1 px-3 font-mono text-dark-300">
                    {typeof row[signal] === 'number'
                      ? (row[signal] as number).toFixed(4)
                      : row[signal]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
