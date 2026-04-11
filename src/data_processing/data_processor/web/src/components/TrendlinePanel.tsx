import { useEffect, useState } from "react";
import { TrendingUp, LineChart, Save, Folder } from "lucide-react";
import type {
  TrendlineType,
  TrendlineConfig,
  TrendlineResult,
  PlotConfig,
} from "../types";

interface TrendlinePanelProps {
  signals: string[];
  selectedSignals: string[];
  timeColumn: string | null;
  disabled: boolean;
  onCalculateTrendline: (config: TrendlineConfig) => TrendlineResult | null;
  onSavePlotConfig: (config: PlotConfig) => void;
  savedConfigs: string[];
  onLoadPlotConfig: (name: string) => PlotConfig | null;
}

export function TrendlinePanel({
  signals,
  selectedSignals,
  timeColumn,
  disabled,
  onCalculateTrendline,
  onSavePlotConfig,
  savedConfigs,
  onLoadPlotConfig,
}: TrendlinePanelProps) {
  const [trendlineType, setTrendlineType] = useState<TrendlineType>("linear");
  const [degree, setDegree] = useState(2);
  const [xColumn, setXColumn] = useState(timeColumn || "");
  const [yColumn, setYColumn] = useState(selectedSignals[0] || "");
  const [xMin, setXMin] = useState("");
  const [xMax, setXMax] = useState("");
  const [result, setResult] = useState<TrendlineResult | null>(null);
  const [configName, setConfigName] = useState("");
  const [selectedConfig, setSelectedConfig] = useState("");

  // Update xColumn when timeColumn changes
  useEffect(() => {
    if (timeColumn) setXColumn(timeColumn);
  }, [timeColumn]);

  // Update yColumn when selectedSignals changes
  useEffect(() => {
    if (selectedSignals.length > 0) setYColumn(selectedSignals[0]);
  }, [selectedSignals]);

  const handleCalculateTrendline = () => {
    if (!xColumn || !yColumn) return;

    const config: TrendlineConfig = {
      type: trendlineType,
      xColumn,
      yColumn,
      degree: trendlineType === "polynomial" ? degree : undefined,
      xMin: xMin ? parseFloat(xMin) : undefined,
      xMax: xMax ? parseFloat(xMax) : undefined,
    };

    const trendlineResult = onCalculateTrendline(config);
    setResult(trendlineResult);
  };

  const handleSavePlotConfig = () => {
    if (!configName) return;

    const config: PlotConfig = {
      name: configName,
      xAxis: xColumn,
      ySignals: selectedSignals,
      trendline: result
        ? {
            type: trendlineType,
            xColumn,
            yColumn,
            degree,
            xMin: xMin ? parseFloat(xMin) : undefined,
            xMax: xMax ? parseFloat(xMax) : undefined,
          }
        : undefined,
    };

    onSavePlotConfig(config);
    setConfigName("");
  };

  const handleLoadConfig = () => {
    if (!selectedConfig) return;
    const config = onLoadPlotConfig(selectedConfig);
    if (config) {
      setXColumn(config.xAxis);
      if (config.trendline) {
        setTrendlineType(config.trendline.type);
        setXMin(config.trendline.xMin?.toString() || "");
        setXMax(config.trendline.xMax?.toString() || "");
        if (config.trendline.degree) setDegree(config.trendline.degree);
      }
    }
  };

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <LineChart className="w-4 h-4" />
        Trendline Analysis
      </div>
      <div className="card-body space-y-4">
        {/* Trendline Settings */}
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">X-Axis</label>
              <select
                className="select w-full"
                value={xColumn}
                onChange={(e) => setXColumn(e.target.value)}
                disabled={disabled}
              >
                <option value="">Select column...</option>
                {signals.map((signal) => (
                  <option key={signal} value={signal}>
                    {signal}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">Y-Axis</label>
              <select
                className="select w-full"
                value={yColumn}
                onChange={(e) => setYColumn(e.target.value)}
                disabled={disabled}
              >
                <option value="">Select signal...</option>
                {selectedSignals.map((signal) => (
                  <option key={signal} value={signal}>
                    {signal}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-1">
            <label className="block text-xs text-dark-400">
              Trendline Type
            </label>
            <select
              className="select w-full"
              value={trendlineType}
              onChange={(e) =>
                setTrendlineType(e.target.value as TrendlineType)
              }
              disabled={disabled}
            >
              <option value="linear">Linear</option>
              <option value="polynomial">Polynomial</option>
              <option value="exponential">Exponential</option>
              <option value="power">Power</option>
            </select>
          </div>

          {trendlineType === "polynomial" && (
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">
                Polynomial Degree
              </label>
              <input
                type="number"
                className="input w-full"
                value={degree}
                onChange={(e) => setDegree(parseInt(e.target.value) || 2)}
                min={2}
                max={10}
                disabled={disabled}
              />
            </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">
                X Min (optional)
              </label>
              <input
                type="number"
                className="input w-full"
                value={xMin}
                onChange={(e) => setXMin(e.target.value)}
                placeholder="Auto"
                disabled={disabled}
              />
            </div>
            <div className="space-y-1">
              <label className="block text-xs text-dark-400">
                X Max (optional)
              </label>
              <input
                type="number"
                className="input w-full"
                value={xMax}
                onChange={(e) => setXMax(e.target.value)}
                placeholder="Auto"
                disabled={disabled}
              />
            </div>
          </div>

          <button
            className="btn btn-primary w-full flex items-center justify-center gap-2"
            onClick={handleCalculateTrendline}
            disabled={disabled || !xColumn || !yColumn}
          >
            <TrendingUp className="w-4 h-4" />
            Calculate Trendline
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="bg-dark-700/50 rounded-lg p-3 space-y-2">
            <h4 className="text-xs font-medium text-dark-400">
              Trendline Result
            </h4>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span className="text-dark-500">Type:</span>
                <span className="text-dark-200 capitalize">{result.type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-dark-500">Equation:</span>
                <span className="text-dark-200 font-mono text-xs">
                  {result.equation}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-dark-500">R²:</span>
                <span className="text-dark-200">
                  {result.rSquared.toFixed(6)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Save/Load Plot Config */}
        <div className="pt-4 border-t border-dark-700 space-y-3">
          <h4 className="text-xs font-medium text-dark-400">
            Plot Configuration
          </h4>

          <div className="flex gap-2">
            <input
              type="text"
              className="input flex-1"
              value={configName}
              onChange={(e) => setConfigName(e.target.value)}
              placeholder="Config name..."
              disabled={disabled}
            />
            <button
              className="btn btn-secondary flex items-center gap-1"
              onClick={handleSavePlotConfig}
              disabled={disabled || !configName}
            >
              <Save className="w-3 h-3" />
              Save
            </button>
          </div>

          {savedConfigs.length > 0 && (
            <div className="flex gap-2">
              <select
                className="select flex-1"
                value={selectedConfig}
                onChange={(e) => setSelectedConfig(e.target.value)}
                disabled={disabled}
              >
                <option value="">Load saved config...</option>
                {savedConfigs.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              <button
                className="btn btn-secondary flex items-center gap-1"
                onClick={handleLoadConfig}
                disabled={disabled || !selectedConfig}
              >
                <Folder className="w-3 h-3" />
                Load
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrendlinePanel;
