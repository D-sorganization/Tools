import React, { useCallback, useState } from 'react';
import { Database, Settings, BarChart3, Table, Calculator, Clock, Scissors, LineChart, HelpCircle } from 'lucide-react';
import {
  FileUpload,
  SignalList,
  FilterPanel,
  DataChart,
  StatisticsPanel,
  ExportPanel,
  AdvancedPanel,
  ResamplePanel,
  TimeRangePanel,
  TrendlinePanel,
  HelpPanel,
} from './components';
import { useDataProcessor } from './hooks';
import type {
  FilterConfig,
  IntegrationConfig,
  DifferentiationConfig,
  ResampleConfig,
  TimeRangeConfig,
  TrendlineConfig,
  FormulaConfig,
  PlotConfig,
} from './types';

type TabType = 'chart' | 'table';
type LeftPanelTab = 'signals' | 'advanced' | 'resample' | 'timerange';
type RightPanelTab = 'stats' | 'trendline' | 'export' | 'help';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('chart');
  const [leftPanelTab, setLeftPanelTab] = useState<LeftPanelTab>('signals');
  const [rightPanelTab, setRightPanelTab] = useState<RightPanelTab>('stats');
  const {
    data,
    filteredData,
    signals,
    selectedSignals,
    statistics,
    isLoading,
    error,
    fileName,
    timeColumn,
    loadFile,
    applyFilter,
    setSelectedSignals,
    resetData,
    clearAll,
    integrateSignals,
    differentiateSignals,
    trimTimeRange,
    calculateTrendline,
    applyFormula,
    savePlotConfig,
    loadPlotConfig,
    savedPlotConfigNames,
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

  const handleIntegrate = useCallback(
    (config: IntegrationConfig) => {
      integrateSignals(config);
    },
    [integrateSignals]
  );

  const handleDifferentiate = useCallback(
    (config: DifferentiationConfig) => {
      differentiateSignals(config);
    },
    [differentiateSignals]
  );

  const handleResample = useCallback(
    (config: ResampleConfig) => {
      // Resample functionality (simplified for now)
      console.log('Resample config:', config);
    },
    []
  );

  const handleTrimRange = useCallback(
    (config: TimeRangeConfig) => {
      trimTimeRange(config);
    },
    [trimTimeRange]
  );

  const handleApplyFormula = useCallback(
    (config: FormulaConfig) => {
      applyFormula(config);
    },
    [applyFormula]
  );

  const handleCalculateTrendline = useCallback(
    (config: TrendlineConfig) => {
      return calculateTrendline(config);
    },
    [calculateTrendline]
  );

  const handleSavePlotConfig = useCallback(
    (config: PlotConfig) => {
      savePlotConfig(config);
    },
    [savePlotConfig]
  );

  const handleLoadPlotConfig = useCallback(
    (name: string) => {
      return loadPlotConfig(name);
    },
    [loadPlotConfig]
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
          <aside className="col-span-12 lg:col-span-3 space-y-4">
            <FileUpload
              onFileSelect={handleFileSelect}
              fileName={fileName}
              onClear={clearAll}
              isLoading={isLoading}
            />

            {/* Left Panel Tabs */}
            <div className="flex border-b border-dark-700 text-xs">
              <button
                onClick={() => setLeftPanelTab('signals')}
                className={`px-3 py-2 ${leftPanelTab === 'signals' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                Signals
              </button>
              <button
                onClick={() => setLeftPanelTab('advanced')}
                className={`px-3 py-2 ${leftPanelTab === 'advanced' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <Calculator className="w-3 h-3 inline mr-1" />
                Advanced
              </button>
              <button
                onClick={() => setLeftPanelTab('resample')}
                className={`px-3 py-2 ${leftPanelTab === 'resample' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <Clock className="w-3 h-3 inline mr-1" />
                Resample
              </button>
              <button
                onClick={() => setLeftPanelTab('timerange')}
                className={`px-3 py-2 ${leftPanelTab === 'timerange' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <Scissors className="w-3 h-3 inline mr-1" />
                Time
              </button>
            </div>

            {/* Left Panel Content */}
            {leftPanelTab === 'signals' && (
              <>
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
              </>
            )}

            {leftPanelTab === 'advanced' && (
              <AdvancedPanel
                signals={signals}
                selectedSignals={selectedSignals}
                timeColumn={timeColumn}
                disabled={!hasData}
                onIntegrate={handleIntegrate}
                onDifferentiate={handleDifferentiate}
                onApplyFormula={handleApplyFormula}
              />
            )}

            {leftPanelTab === 'resample' && (
              <ResamplePanel
                timeColumn={timeColumn}
                disabled={!hasData}
                onResample={handleResample}
              />
            )}

            {leftPanelTab === 'timerange' && (
              <TimeRangePanel
                data={filteredData}
                timeColumn={timeColumn}
                disabled={!hasData}
                onTrimRange={handleTrimRange}
              />
            )}
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
          <aside className="col-span-12 lg:col-span-3 space-y-4">
            {/* Right Panel Tabs */}
            <div className="flex border-b border-dark-700 text-xs">
              <button
                onClick={() => setRightPanelTab('stats')}
                className={`px-3 py-2 ${rightPanelTab === 'stats' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                Stats
              </button>
              <button
                onClick={() => setRightPanelTab('trendline')}
                className={`px-3 py-2 ${rightPanelTab === 'trendline' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <LineChart className="w-3 h-3 inline mr-1" />
                Trendline
              </button>
              <button
                onClick={() => setRightPanelTab('export')}
                className={`px-3 py-2 ${rightPanelTab === 'export' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                Export
              </button>
              <button
                onClick={() => setRightPanelTab('help')}
                className={`px-3 py-2 ${rightPanelTab === 'help' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <HelpCircle className="w-3 h-3 inline mr-1" />
                Help
              </button>
            </div>

            {/* Right Panel Content */}
            {rightPanelTab === 'stats' && (
              <StatisticsPanel
                statistics={statistics}
                selectedSignals={selectedSignals}
              />
            )}

            {rightPanelTab === 'trendline' && (
              <TrendlinePanel
                signals={signals}
                selectedSignals={selectedSignals}
                timeColumn={timeColumn}
                disabled={!hasData}
                onCalculateTrendline={handleCalculateTrendline}
                onSavePlotConfig={handleSavePlotConfig}
                savedConfigs={savedPlotConfigNames}
                onLoadPlotConfig={handleLoadPlotConfig}
              />
            )}

            {rightPanelTab === 'export' && (
              <ExportPanel
                data={filteredData}
                fileName={fileName}
                disabled={!hasData}
              />
            )}

            {rightPanelTab === 'help' && <HelpPanel />}
          </aside>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-dark-800 border-t border-dark-700 px-6 py-4 mt-8">
        <div className="max-w-screen-2xl mx-auto text-center text-sm text-dark-500">
          Data Processor v2.0 · Built with React + TypeScript + Tauri
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
