import { useCallback, useState, memo, useEffect } from 'react';
import { Database, Settings, BarChart3, Table, Calculator, Clock, Scissors, LineChart, HelpCircle, FlaskConical, Menu, X } from 'lucide-react';
import {
  FileUpload,
  SignalList,
  FilterPanel,
  PlotView,
  StatisticsPanel,
  ExportPanel,
  AdvancedPanel,
  ResamplePanel,
  TimeRangePanel,
  TrendlinePanel,
  HelpPanel,
  AnalyticsSuite,
  ToastContainer,
} from './components';
import { useDataProcessor, useNotification } from './hooks';
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
type RightPanelTab = 'stats' | 'analytics' | 'trendline' | 'export' | 'help';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('chart');
  const [leftPanelTab, setLeftPanelTab] = useState<LeftPanelTab>('signals');
  const [rightPanelTab, setRightPanelTab] = useState<RightPanelTab>('stats');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const { notifications, showNotification, removeNotification } =
    useNotification();
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
      // Validate file before upload
      const maxSize = 50 * 1024 * 1024; // 50MB
      const validExtensions = ['.csv'];

      if (!validExtensions.some((ext) => file.name.endsWith(ext))) {
        showNotification({
          type: 'error',
          message: `Invalid file type. Please upload a CSV file.`,
          duration: 5000,
        });
        return;
      }

      if (file.size > maxSize) {
        showNotification({
          type: 'error',
          message: `File is too large. Maximum size is 50MB.`,
          duration: 5000,
        });
        return;
      }

      try {
        const result = await loadFile(file);
        if (result.success) {
          showNotification({
            type: 'success',
            message: `File loaded successfully. Found ${result.signals?.length || 0} signals.`,
            duration: 4000,
          });
        } else {
          showNotification({
            type: 'error',
            message: result.error || 'Failed to load file.',
            duration: 5000,
          });
        }
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : 'An unexpected error occurred';
        showNotification({
          type: 'error',
          message: `Error loading file: ${errorMsg}`,
          duration: 5000,
        });
      }
    },
    [loadFile, showNotification]
  );

  const handleApplyFilter = useCallback(
    (config: FilterConfig) => {
      try {
        const result = applyFilter(config);
        if (result.success) {
          showNotification({
            type: 'success',
            message: 'Filter applied successfully.',
            duration: 3000,
          });
        } else {
          showNotification({
            type: 'error',
            message: result.error || 'Failed to apply filter.',
            duration: 5000,
          });
        }
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : 'An unexpected error occurred';
        showNotification({
          type: 'error',
          message: `Filter error: ${errorMsg}`,
          duration: 5000,
        });
      }
    },
    [applyFilter, showNotification]
  );

  const handleIntegrate = useCallback(
    (config: IntegrationConfig) => {
      try {
        const result = integrateSignals(config);
        if (result.success) {
          showNotification({
            type: 'success',
            message: 'Signals integrated successfully.',
            duration: 3000,
          });
        } else {
          showNotification({
            type: 'error',
            message: result.error || 'Failed to integrate signals.',
            duration: 5000,
          });
        }
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : 'An unexpected error occurred';
        showNotification({
          type: 'error',
          message: `Integration error: ${errorMsg}`,
          duration: 5000,
        });
      }
    },
    [integrateSignals, showNotification]
  );

  const handleDifferentiate = useCallback(
    (config: DifferentiationConfig) => {
      try {
        const result = differentiateSignals(config);
        if (result.success) {
          showNotification({
            type: 'success',
            message: 'Signals differentiated successfully.',
            duration: 3000,
          });
        } else {
          showNotification({
            type: 'error',
            message: result.error || 'Failed to differentiate signals.',
            duration: 5000,
          });
        }
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : 'An unexpected error occurred';
        showNotification({
          type: 'error',
          message: `Differentiation error: ${errorMsg}`,
          duration: 5000,
        });
      }
    },
    [differentiateSignals, showNotification]
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
      try {
        const result = trimTimeRange(config);
        if (result.success) {
          showNotification({
            type: 'success',
            message: 'Time range trimmed successfully.',
            duration: 3000,
          });
        } else {
          showNotification({
            type: 'error',
            message: result.error || 'Failed to trim time range.',
            duration: 5000,
          });
        }
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : 'An unexpected error occurred';
        showNotification({
          type: 'error',
          message: `Time range error: ${errorMsg}`,
          duration: 5000,
        });
      }
    },
    [trimTimeRange, showNotification]
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

  // Track viewport size for responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Close sidebar on mobile when tab changes
  useEffect(() => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  }, [activeTab, leftPanelTab, rightPanelTab, isMobile]);

  const hasData = data.length > 0;

  return (
    <div className="min-h-screen bg-dark-900 text-dark-100 flex flex-col">
      {/* Header */}
      <header className="bg-dark-800 border-b border-dark-700 px-4 md:px-6 py-4 sticky top-0 z-50">
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center gap-2 md:gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="md:hidden min-h-[48px] min-w-[48px] flex items-center justify-center rounded-lg hover:bg-dark-700 transition-colors"
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
            <Database className="w-6 md:w-8 h-6 md:h-8 text-blue-500 flex-shrink-0" />
            <div className="hidden sm:block">
              <h1 className="text-lg md:text-xl font-bold">Data Processor</h1>
              <p className="text-xs md:text-sm text-dark-400 hidden md:block">Signal processing and analysis</p>
            </div>
          </div>
          {fileName && (
            <div className="flex items-center gap-2 text-xs md:text-sm text-dark-400 whitespace-nowrap">
              <Settings className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:inline">{data.length.toLocaleString()} rows · {signals.length} signals</span>
              <span className="sm:hidden">{signals.length}S</span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full overflow-hidden">
        {error && (
          <div className="m-4 md:m-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        <div className="flex flex-col lg:grid lg:grid-cols-12 lg:gap-6 h-full p-4 md:p-6 overflow-y-auto lg:overflow-auto">
          {/* Mobile Sidebar Overlay */}
          {isMobile && sidebarOpen && (
            <div
              className="fixed inset-0 bg-black/50 z-40 md:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          )}

          {/* Left Sidebar - Responsive: Mobile hamburger drawer, tablet/desktop visible */}
          <aside
            className={`${
              isMobile
                ? `fixed top-20 left-0 bottom-0 z-40 bg-dark-800 border-r border-dark-700 overflow-y-auto transition-transform duration-300 ${
                    sidebarOpen ? 'translate-x-0' : '-translate-x-full'
                  } w-80 max-w-[90vw]`
                : 'col-span-3 hidden md:block'
            } space-y-4`}
          >
            <FileUpload
              onFileSelect={handleFileSelect}
              fileName={fileName}
              onClear={clearAll}
              isLoading={isLoading}
            />

            {/* Left Panel Tabs */}
            <div className="flex overflow-x-auto border-b border-dark-700 text-xs">
              <button
                onClick={() => setLeftPanelTab('signals')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${leftPanelTab === 'signals' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                Signals
              </button>
              <button
                onClick={() => setLeftPanelTab('advanced')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${leftPanelTab === 'advanced' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <Calculator className="w-3 h-3 mr-1 flex-shrink-0" />
                <span className="hidden sm:inline">Advanced</span>
                <span className="sm:hidden">Adv</span>
              </button>
              <button
                onClick={() => setLeftPanelTab('resample')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${leftPanelTab === 'resample' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <Clock className="w-3 h-3 mr-1 flex-shrink-0" />
                <span className="hidden sm:inline">Resample</span>
                <span className="sm:hidden">Rsp</span>
              </button>
              <button
                onClick={() => setLeftPanelTab('timerange')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${leftPanelTab === 'timerange' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <Scissors className="w-3 h-3 mr-1 flex-shrink-0" />
                <span className="hidden sm:inline">Time</span>
                <span className="sm:hidden">Tm</span>
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

          {/* Main Content Area - Responsive: Full width on mobile, centered on desktop */}
          <div className="col-span-12 lg:col-span-6 space-y-4 md:space-y-6 min-h-0 flex-1 lg:flex-auto">
            {/* Tabs - Responsive: Flexible on mobile, wrapped on small screens */}
            <div className="flex flex-wrap border-b border-dark-700 gap-1 sm:gap-0">
              <button
                onClick={() => setActiveTab('chart')}
                className={`tab min-h-[48px] flex items-center gap-2 transition-colors flex-1 sm:flex-none ${activeTab === 'chart' ? 'tab-active' : ''}`}
              >
                <BarChart3 className="w-4 h-4 flex-shrink-0" />
                <span className="hidden sm:inline">Chart View</span>
                <span className="sm:hidden">Chart</span>
              </button>
              <button
                onClick={() => setActiveTab('table')}
                className={`tab min-h-[48px] flex items-center gap-2 transition-colors flex-1 sm:flex-none ${activeTab === 'table' ? 'tab-active' : ''}`}
              >
                <Table className="w-4 h-4 flex-shrink-0" />
                <span className="hidden sm:inline">Table View</span>
                <span className="sm:hidden">Table</span>
              </button>
            </div>

            {/* Tab Content */}
            {activeTab === 'chart' ? (
              <PlotView
                data={filteredData}
                selectedSignals={selectedSignals}
                title="Filtered Data"
                height={400}
              />
            ) : (
              <DataTableView data={filteredData} selectedSignals={selectedSignals} />
            )}

            {/* Original vs Filtered Comparison */}
            {hasData && filteredData !== data && (
              <PlotView
                data={data}
                selectedSignals={selectedSignals}
                title="Original Data"
                height={350}
              />
            )}
          </div>

          {/* Right Sidebar - Hidden on mobile and tablet, visible on desktop */}
          <aside className="col-span-3 hidden lg:block space-y-4">
            {/* Right Panel Tabs */}
            <div className="flex overflow-x-auto border-b border-dark-700 text-xs">
              <button
                onClick={() => setRightPanelTab('stats')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${rightPanelTab === 'stats' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                Stats
              </button>
              <button
                onClick={() => setRightPanelTab('analytics')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${rightPanelTab === 'analytics' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <FlaskConical className="w-3 h-3 mr-1 flex-shrink-0" />
                Analytics
              </button>
              <button
                onClick={() => setRightPanelTab('trendline')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${rightPanelTab === 'trendline' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <LineChart className="w-3 h-3 mr-1 flex-shrink-0" />
                Trendline
              </button>
              <button
                onClick={() => setRightPanelTab('export')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${rightPanelTab === 'export' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                Export
              </button>
              <button
                onClick={() => setRightPanelTab('help')}
                className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${rightPanelTab === 'help' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
              >
                <HelpCircle className="w-3 h-3 mr-1 flex-shrink-0" />
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

            {rightPanelTab === 'analytics' && (
              <AnalyticsSuite
                data={filteredData}
                signals={signals}
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
      <footer className="bg-dark-800 border-t border-dark-700 px-4 md:px-6 py-4 mt-8">
        <div className="text-center text-xs md:text-sm text-dark-500">
          Data Processor v2.0 · <span className="hidden sm:inline">Built with React + TypeScript + Tauri</span>
        </div>
      </footer>

      {/* Toast Notifications */}
      <ToastContainer notifications={notifications} onClose={removeNotification} />
    </div>
  );
}

// Simple table view component
interface DataTableViewProps {
  data: Array<Record<string, number | string>>;
  selectedSignals: string[];
}

const DataTableView = memo(function DataTableView({ data, selectedSignals }: DataTableViewProps) {
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
});

export default App;
