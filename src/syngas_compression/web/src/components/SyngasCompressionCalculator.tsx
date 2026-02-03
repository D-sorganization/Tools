import { useState, useCallback, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

// Types
interface GasComposition {
  H2: number;
  CO: number;
  CO2: number;
  CH4: number;
  N2: number;
  H2O: number;
  Ar: number;
}

interface CompressionStage {
  inletPressure: number;
  outletPressure: number;
  inletTemperature: number;
  efficiency: number;
  compressionType: 'isentropic' | 'polytropic' | 'isothermal';
}

interface StageResult {
  stage: number;
  inletPressure: number;
  outletPressure: number;
  pressureRatio: number;
  inletTemperature: number;
  outletTemperature: number;
  temperatureRise: number;
  work: number;
  power: number;
  waterDropout: number;
}

interface CalculationResult {
  stages: StageResult[];
  totalPower: number;
  totalWork: number;
  totalWaterDropout: number;
  warnings: string[];
  recommendations: string[];
}

// Constants
const R_GAS = 8.314462618; // J/(mol·K)

const GAMMA_VALUES: Record<string, number> = {
  H2: 1.41,
  CO: 1.40,
  CO2: 1.30,
  CH4: 1.32,
  N2: 1.40,
  H2O: 1.33,
  Ar: 1.67,
};

const MOLECULAR_WEIGHTS: Record<string, number> = {
  H2: 2.016,
  CO: 28.01,
  CO2: 44.01,
  CH4: 16.04,
  N2: 28.01,
  H2O: 18.015,
  Ar: 39.948,
};

// Preset compositions
const PRESETS: Record<string, GasComposition> = {
  'Biomass Syngas': { H2: 20, CO: 20, CO2: 15, CH4: 5, N2: 35, H2O: 5, Ar: 0 },
  'Coal Syngas': { H2: 30, CO: 40, CO2: 15, CH4: 5, N2: 5, H2O: 5, Ar: 0 },
  'Natural Gas Reforming': { H2: 50, CO: 15, CO2: 10, CH4: 5, N2: 15, H2O: 5, Ar: 0 },
  'Custom': { H2: 25, CO: 25, CO2: 15, CH4: 5, N2: 25, H2O: 5, Ar: 0 },
};

export function SyngasCompressionCalculator() {
  const [activeTab, setActiveTab] = useState<'input' | 'results' | 'analysis' | 'charts'>('input');

  // Gas composition state
  const [composition, setComposition] = useState<GasComposition>(PRESETS['Custom']);
  const [selectedPreset, setSelectedPreset] = useState<string>('Custom');

  // Process parameters
  const [flowRate, setFlowRate] = useState(100); // kmol/h
  const [intercooling, setIntercooling] = useState(true);
  const [intercoolerTemp, setIntercoolerTemp] = useState(40); // °C

  // Compression stages
  const [stages, setStages] = useState<CompressionStage[]>([
    { inletPressure: 1.0, outletPressure: 3.0, inletTemperature: 40, efficiency: 0.75, compressionType: 'isentropic' },
    { inletPressure: 3.0, outletPressure: 10.0, inletTemperature: 40, efficiency: 0.75, compressionType: 'isentropic' },
    { inletPressure: 10.0, outletPressure: 30.0, inletTemperature: 40, efficiency: 0.75, compressionType: 'isentropic' },
  ]);

  // Results
  const [results, setResults] = useState<CalculationResult | null>(null);

  // Calculate mixture properties
  const mixtureProperties = useMemo(() => {
    const total = Object.values(composition).reduce((a, b) => a + b, 0);
    const normalizedComp: Record<string, number> = {};

    for (const [key, value] of Object.entries(composition)) {
      normalizedComp[key] = value / total;
    }

    // Calculate molecular weight
    let mw = 0;
    for (const [species, fraction] of Object.entries(normalizedComp)) {
      mw += fraction * MOLECULAR_WEIGHTS[species];
    }

    // Calculate heat capacity ratio (gamma)
    let gamma = 0;
    for (const [species, fraction] of Object.entries(normalizedComp)) {
      gamma += fraction * GAMMA_VALUES[species];
    }

    return { molecularWeight: mw, gamma, composition: normalizedComp };
  }, [composition]);

  // Handle preset selection
  const handlePresetChange = useCallback((preset: string) => {
    setSelectedPreset(preset);
    setComposition(PRESETS[preset]);
  }, []);

  // Handle composition change
  const handleCompositionChange = useCallback((species: keyof GasComposition, value: number) => {
    setComposition(prev => ({ ...prev, [species]: value }));
    setSelectedPreset('Custom');
  }, []);

  // Handle stage change
  const handleStageChange = useCallback((index: number, field: keyof CompressionStage, value: number | string) => {
    setStages(prev => {
      const newStages = [...prev];
      newStages[index] = { ...newStages[index], [field]: value };

      // Auto-update next stage inlet pressure
      if (field === 'outletPressure' && index < prev.length - 1) {
        newStages[index + 1] = { ...newStages[index + 1], inletPressure: value as number };
      }

      return newStages;
    });
  }, []);

  // Add/remove stages
  const addStage = useCallback(() => {
    if (stages.length < 6) {
      const lastStage = stages[stages.length - 1];
      setStages(prev => [...prev, {
        inletPressure: lastStage.outletPressure,
        outletPressure: lastStage.outletPressure * 3,
        inletTemperature: intercooling ? intercoolerTemp : 40,
        efficiency: 0.75,
        compressionType: 'isentropic',
      }]);
    }
  }, [stages, intercooling, intercoolerTemp]);

  const removeStage = useCallback((index: number) => {
    if (stages.length > 1) {
      setStages(prev => prev.filter((_, i) => i !== index));
    }
  }, [stages.length]);

  // Calculate compression
  const calculateCompression = useCallback(() => {
    const { gamma, molecularWeight } = mixtureProperties;
    const stageResults: StageResult[] = [];
    let totalPower = 0;
    let totalWork = 0;
    let totalWaterDropout = 0;
    const warnings: string[] = [];
    const recommendations: string[] = [];

    for (let i = 0; i < stages.length; i++) {
      const stage = stages[i];
      const pressureRatio = stage.outletPressure / stage.inletPressure;
      const inletTempK = stage.inletTemperature + 273.15;

      // Calculate outlet temperature (isentropic)
      const isentropicTempRatio = Math.pow(pressureRatio, (gamma - 1) / gamma);
      const isentropicOutletTemp = inletTempK * isentropicTempRatio;

      // Actual outlet temperature based on efficiency
      const actualOutletTemp = inletTempK + (isentropicOutletTemp - inletTempK) / stage.efficiency;
      const outletTempC = actualOutletTemp - 273.15;
      const temperatureRise = outletTempC - stage.inletTemperature;

      // Calculate work (J/mol)
      const work = (R_GAS * inletTempK * gamma / (gamma - 1)) *
                   (Math.pow(pressureRatio, (gamma - 1) / gamma) - 1) / stage.efficiency;

      // Calculate power (HP)
      const molarFlowRate = flowRate * 1000 / 3600; // kmol/h to mol/s
      const powerWatts = work * molarFlowRate;
      const powerHP = powerWatts / 745.7;

      // Estimate water dropout (simplified)
      const waterDropout = outletTempC < 100 ?
        composition.H2O * 0.1 * (100 - outletTempC) / 100 : 0;

      stageResults.push({
        stage: i + 1,
        inletPressure: stage.inletPressure,
        outletPressure: stage.outletPressure,
        pressureRatio,
        inletTemperature: stage.inletTemperature,
        outletTemperature: outletTempC,
        temperatureRise,
        work,
        power: powerHP,
        waterDropout,
      });

      totalPower += powerHP;
      totalWork += work;
      totalWaterDropout += waterDropout;

      // Generate warnings
      if (outletTempC > 200) {
        warnings.push(`Stage ${i + 1}: High outlet temperature (${outletTempC.toFixed(1)}°C)`);
      }
      if (pressureRatio > 4) {
        warnings.push(`Stage ${i + 1}: High pressure ratio (${pressureRatio.toFixed(2)})`);
      }
      if (stage.efficiency < 0.7) {
        warnings.push(`Stage ${i + 1}: Low efficiency (${(stage.efficiency * 100).toFixed(0)}%)`);
      }
    }

    // Generate recommendations
    if (warnings.some(w => w.includes('High outlet temperature'))) {
      recommendations.push('Consider adding intercooling between stages to reduce outlet temperatures.');
    }
    if (warnings.some(w => w.includes('High pressure ratio'))) {
      recommendations.push('Consider splitting high pressure ratio stages into multiple stages.');
    }
    if (totalPower > 1000) {
      recommendations.push('High total power requirement. Consider process optimization.');
    }
    if (totalWaterDropout > 0.5) {
      recommendations.push('Significant water dropout expected. Install knockout drums between stages.');
    }

    setResults({
      stages: stageResults,
      totalPower,
      totalWork,
      totalWaterDropout,
      warnings,
      recommendations,
    });

    setActiveTab('results');
  }, [stages, mixtureProperties, flowRate, composition.H2O]);

  // Chart data
  const temperatureChartData = useMemo(() => {
    if (!results) return [];
    return results.stages.map(s => ({
      name: `Stage ${s.stage}`,
      inlet: s.inletTemperature,
      outlet: s.outletTemperature,
      rise: s.temperatureRise,
    }));
  }, [results]);

  const pressureChartData = useMemo(() => {
    if (!results) return [];
    return results.stages.map(s => ({
      name: `Stage ${s.stage}`,
      inlet: s.inletPressure,
      outlet: s.outletPressure,
    }));
  }, [results]);

  const powerChartData = useMemo(() => {
    if (!results) return [];
    return results.stages.map(s => ({
      name: `Stage ${s.stage}`,
      power: s.power,
    }));
  }, [results]);

  return (
    <div className="space-y-6">
      {/* Tabs */}
      <div className="flex space-x-4 border-b border-slate-700">
        {(['input', 'results', 'analysis', 'charts'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-medium capitalize transition-colors ${
              activeTab === tab
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Input Tab */}
      {activeTab === 'input' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Gas Composition */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Gas Composition</h3>

            <div className="mb-4">
              <label className="block text-sm text-slate-400 mb-1">Preset</label>
              <select
                value={selectedPreset}
                onChange={(e) => handlePresetChange(e.target.value)}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
              >
                {Object.keys(PRESETS).map(preset => (
                  <option key={preset} value={preset}>{preset}</option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {(Object.keys(composition) as (keyof GasComposition)[]).map(species => (
                <div key={species}>
                  <label className="block text-sm text-slate-400 mb-1">{species} (%)</label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={composition[species]}
                    onChange={(e) => handleCompositionChange(species, parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              ))}
            </div>

            <div className="mt-4 p-3 bg-slate-700/50 rounded">
              <div className="text-sm text-slate-400">
                Total: <span className={`font-semibold ${Math.abs(Object.values(composition).reduce((a, b) => a + b, 0) - 100) < 0.1 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {Object.values(composition).reduce((a, b) => a + b, 0).toFixed(1)}%
                </span>
              </div>
              <div className="text-sm text-slate-400">
                MW: <span className="text-white">{mixtureProperties.molecularWeight.toFixed(2)} g/mol</span>
              </div>
              <div className="text-sm text-slate-400">
                γ: <span className="text-white">{mixtureProperties.gamma.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* Process Parameters */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Process Parameters</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Flow Rate (kmol/h)</label>
                <input
                  type="number"
                  min="1"
                  step="10"
                  value={flowRate}
                  onChange={(e) => setFlowRate(parseFloat(e.target.value) || 1)}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id="intercooling"
                  checked={intercooling}
                  onChange={(e) => setIntercooling(e.target.checked)}
                  className="w-4 h-4 rounded bg-slate-700 text-blue-500 focus:ring-blue-500"
                />
                <label htmlFor="intercooling" className="text-sm text-slate-300">
                  Enable intercooling between stages
                </label>
              </div>

              {intercooling && (
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Intercooler Temperature (°C)</label>
                  <input
                    type="number"
                    min="20"
                    max="60"
                    value={intercoolerTemp}
                    onChange={(e) => setIntercoolerTemp(parseFloat(e.target.value) || 40)}
                    className="w-full bg-slate-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Compression Stages */}
          <div className="bg-slate-800 rounded-lg p-6 lg:col-span-2">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">Compression Stages</h3>
              <button
                onClick={addStage}
                disabled={stages.length >= 6}
                className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                + Add Stage
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-700">
                    <th className="text-left py-2 px-3">Stage</th>
                    <th className="text-left py-2 px-3">Inlet P (bar)</th>
                    <th className="text-left py-2 px-3">Outlet P (bar)</th>
                    <th className="text-left py-2 px-3">Inlet T (°C)</th>
                    <th className="text-left py-2 px-3">Efficiency</th>
                    <th className="text-left py-2 px-3">Type</th>
                    <th className="text-left py-2 px-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {stages.map((stage, index) => (
                    <tr key={index} className="border-b border-slate-700/50">
                      <td className="py-2 px-3 text-white font-medium">{index + 1}</td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min="0.1"
                          step="0.1"
                          value={stage.inletPressure}
                          onChange={(e) => handleStageChange(index, 'inletPressure', parseFloat(e.target.value) || 1)}
                          className="w-20 bg-slate-700 text-white rounded px-2 py-1"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min="0.1"
                          step="0.5"
                          value={stage.outletPressure}
                          onChange={(e) => handleStageChange(index, 'outletPressure', parseFloat(e.target.value) || 1)}
                          className="w-20 bg-slate-700 text-white rounded px-2 py-1"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min="0"
                          max="200"
                          value={stage.inletTemperature}
                          onChange={(e) => handleStageChange(index, 'inletTemperature', parseFloat(e.target.value) || 40)}
                          className="w-20 bg-slate-700 text-white rounded px-2 py-1"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min="0.5"
                          max="1"
                          step="0.01"
                          value={stage.efficiency}
                          onChange={(e) => handleStageChange(index, 'efficiency', parseFloat(e.target.value) || 0.75)}
                          className="w-20 bg-slate-700 text-white rounded px-2 py-1"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <select
                          value={stage.compressionType}
                          onChange={(e) => handleStageChange(index, 'compressionType', e.target.value)}
                          className="bg-slate-700 text-white rounded px-2 py-1"
                        >
                          <option value="isentropic">Isentropic</option>
                          <option value="polytropic">Polytropic</option>
                          <option value="isothermal">Isothermal</option>
                        </select>
                      </td>
                      <td className="py-2 px-3">
                        {stages.length > 1 && (
                          <button
                            onClick={() => removeStage(index)}
                            className="text-red-400 hover:text-red-300"
                          >
                            ×
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Calculate Button */}
          <div className="lg:col-span-2">
            <button
              onClick={calculateCompression}
              className="w-full py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
            >
              Calculate Compression
            </button>
          </div>
        </div>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && results && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Total Power</div>
              <div className="text-2xl font-bold text-blue-400">{results.totalPower.toFixed(1)} HP</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Total Work</div>
              <div className="text-2xl font-bold text-green-400">{(results.totalWork / 1000).toFixed(2)} kJ/mol</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Stages</div>
              <div className="text-2xl font-bold text-purple-400">{results.stages.length}</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <div className="text-slate-400 text-sm">Water Dropout</div>
              <div className="text-2xl font-bold text-yellow-400">{results.totalWaterDropout.toFixed(2)}%</div>
            </div>
          </div>

          {/* Stage Results Table */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Stage Results</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-700">
                    <th className="text-left py-2 px-3">Stage</th>
                    <th className="text-right py-2 px-3">P<sub>in</sub> (bar)</th>
                    <th className="text-right py-2 px-3">P<sub>out</sub> (bar)</th>
                    <th className="text-right py-2 px-3">Ratio</th>
                    <th className="text-right py-2 px-3">T<sub>in</sub> (°C)</th>
                    <th className="text-right py-2 px-3">T<sub>out</sub> (°C)</th>
                    <th className="text-right py-2 px-3">ΔT (°C)</th>
                    <th className="text-right py-2 px-3">Power (HP)</th>
                  </tr>
                </thead>
                <tbody>
                  {results.stages.map((stage) => (
                    <tr key={stage.stage} className="border-b border-slate-700/50 text-white">
                      <td className="py-2 px-3 font-medium">{stage.stage}</td>
                      <td className="py-2 px-3 text-right">{stage.inletPressure.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right">{stage.outletPressure.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right">{stage.pressureRatio.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right">{stage.inletTemperature.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right">
                        <span className={stage.outletTemperature > 200 ? 'text-red-400' : ''}>
                          {stage.outletTemperature.toFixed(1)}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right">{stage.temperatureRise.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right">{stage.power.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="text-white font-semibold border-t border-slate-600">
                    <td colSpan={7} className="py-2 px-3 text-right">Total Power:</td>
                    <td className="py-2 px-3 text-right text-blue-400">{results.totalPower.toFixed(1)} HP</td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Tab */}
      {activeTab === 'analysis' && results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Warnings */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <span className="w-2 h-2 rounded-full bg-yellow-400 mr-2"></span>
              Warnings ({results.warnings.length})
            </h3>
            {results.warnings.length > 0 ? (
              <ul className="space-y-2">
                {results.warnings.map((warning, index) => (
                  <li key={index} className="text-yellow-400 text-sm flex items-start">
                    <span className="mr-2">⚠</span>
                    {warning}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-400 text-sm">No warnings. All parameters within acceptable ranges.</p>
            )}
          </div>

          {/* Recommendations */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <span className="w-2 h-2 rounded-full bg-blue-400 mr-2"></span>
              Recommendations ({results.recommendations.length})
            </h3>
            {results.recommendations.length > 0 ? (
              <ul className="space-y-2">
                {results.recommendations.map((rec, index) => (
                  <li key={index} className="text-blue-400 text-sm flex items-start">
                    <span className="mr-2">💡</span>
                    {rec}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-400 text-sm">No recommendations. Process design looks optimal.</p>
            )}
          </div>

          {/* Process Summary */}
          <div className="bg-slate-800 rounded-lg p-6 md:col-span-2">
            <h3 className="text-lg font-semibold text-white mb-4">Process Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-slate-400">Feed Composition</div>
                <div className="text-white mt-1">
                  {Object.entries(composition)
                    .filter(([_, v]) => v > 0)
                    .map(([k, v]) => `${k}: ${v.toFixed(1)}%`)
                    .join(', ')}
                </div>
              </div>
              <div>
                <div className="text-slate-400">Flow Rate</div>
                <div className="text-white mt-1">{flowRate} kmol/h</div>
              </div>
              <div>
                <div className="text-slate-400">Inlet Pressure</div>
                <div className="text-white mt-1">{stages[0].inletPressure} bar</div>
              </div>
              <div>
                <div className="text-slate-400">Final Pressure</div>
                <div className="text-white mt-1">{stages[stages.length - 1].outletPressure} bar</div>
              </div>
              <div>
                <div className="text-slate-400">Overall Ratio</div>
                <div className="text-white mt-1">
                  {(stages[stages.length - 1].outletPressure / stages[0].inletPressure).toFixed(1)}:1
                </div>
              </div>
              <div>
                <div className="text-slate-400">Intercooling</div>
                <div className="text-white mt-1">{intercooling ? `Yes (${intercoolerTemp}°C)` : 'No'}</div>
              </div>
              <div>
                <div className="text-slate-400">Molecular Weight</div>
                <div className="text-white mt-1">{mixtureProperties.molecularWeight.toFixed(2)} g/mol</div>
              </div>
              <div>
                <div className="text-slate-400">Heat Capacity Ratio</div>
                <div className="text-white mt-1">{mixtureProperties.gamma.toFixed(3)}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts Tab */}
      {activeTab === 'charts' && results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Temperature Chart */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Temperature Profile</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={temperatureChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="inlet" name="Inlet T (°C)" stroke="#3b82f6" strokeWidth={2} />
                  <Line type="monotone" dataKey="outlet" name="Outlet T (°C)" stroke="#ef4444" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Pressure Chart */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Pressure Profile</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={pressureChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="inlet" name="Inlet P (bar)" stroke="#22c55e" strokeWidth={2} />
                  <Line type="monotone" dataKey="outlet" name="Outlet P (bar)" stroke="#a855f7" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Power Chart */}
          <div className="bg-slate-800 rounded-lg p-6 md:col-span-2">
            <h3 className="text-lg font-semibold text-white mb-4">Power by Stage</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={powerChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Bar dataKey="power" name="Power (HP)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* No results message */}
      {(activeTab === 'results' || activeTab === 'analysis' || activeTab === 'charts') && !results && (
        <div className="bg-slate-800 rounded-lg p-12 text-center">
          <p className="text-slate-400">No results yet. Configure your inputs and click "Calculate Compression".</p>
        </div>
      )}
    </div>
  );
}
