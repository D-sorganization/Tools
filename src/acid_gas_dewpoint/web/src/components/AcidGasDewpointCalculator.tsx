import { useState, useCallback } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

// API base URL -- defaults to shared calc backend.
// Override via VITE_CALC_API_URL environment variable.
// See issue #609.
const CALC_API_BASE = import.meta.env.VITE_CALC_API_URL ?? 'http://localhost:8010'

// Antoine equation constants for acid gases (client-side fallback)
const ANTOINE_CONSTANTS: Record<string, { A: number; B: number; C: number }> = {
  H2O: { A: 8.07131, B: 1730.63, C: 233.426 },
  HF: { A: 7.158, B: 1111.0, C: 235.0 },
  HCl: { A: 7.96, B: 1118.0, C: 240.0 },
  H2S: { A: 6.987, B: 884.0, C: 240.0 },
}

// Preset compositions
const PRESETS: Record<string, Record<string, number>> = {
  'Typical Syngas': { H2O: 15.0, HF: 0.01, HCl: 0.02, H2S: 0.1 },
  'High Acid Content': { H2O: 20.0, HF: 0.1, HCl: 0.2, H2S: 0.5 },
  'Coal Gasification': { H2O: 12.0, HF: 0.05, HCl: 0.1, H2S: 0.3 },
  'Biomass Gasification': { H2O: 18.0, HF: 0.02, HCl: 0.05, H2S: 0.2 },
}

interface DewpointResult {
  component: string
  dewpoint: number
  partialPressure: number
  vaporPressure: number
}

interface CalculationResults {
  dewpoints: DewpointResult[]
  overallDewpoint: number
  limitingComponent: string
  margin: number
  risk: string
  warnings: string[]
  engine: string
}

// ---------------------------------------------------------------------------
// Client-side fallback (used when shared calc backend is unreachable)
// See issue #609 for context on why the Python backend is preferred.
// ---------------------------------------------------------------------------

function calculateVaporPressure(tempC: number, component: string): number {
  const constants = ANTOINE_CONSTANTS[component]
  if (!constants) return 0
  const logP = constants.A - constants.B / (constants.C + tempC)
  return Math.pow(10, logP) * 133.322 // Convert mmHg to Pa
}

function calculateDewpoint(partialPressurePa: number, component: string): number {
  if (partialPressurePa <= 0) return NaN
  const constants = ANTOINE_CONSTANTS[component]
  if (!constants) return NaN
  const pMmHg = partialPressurePa / 133.322
  if (pMmHg <= 0) return NaN
  try {
    return constants.B / (constants.A - Math.log10(pMmHg)) - constants.C
  } catch {
    return NaN
  }
}

function getRiskLevel(margin: number): string {
  if (isNaN(margin)) return 'Unknown'
  if (margin < 0) return 'HIGH - Condensation occurring'
  if (margin < 10) return 'MEDIUM - Within 10°C of dewpoint'
  if (margin < 30) return 'LOW - Safe margin'
  return 'VERY LOW - Large safety margin'
}

function calculateFallback(
  temperature: number,
  pressure: number,
  composition: Record<string, number>,
): CalculationResults {
  const pressurePa = pressure * 1e5
  const warnings: string[] = []

  if (temperature < -100 || temperature > 400) {
    warnings.push('Temperature outside recommended range (-100 to 400 deg C)')
  }
  if (pressure < 0.1 || pressure > 300) {
    warnings.push('Pressure outside recommended range (0.1 to 300 bar)')
  }

  const components = ['H2O', 'HF', 'HCl', 'H2S']
  const dewpointResults: DewpointResult[] = []

  for (const comp of components) {
    const moleFraction = (composition[comp] || 0) / 100
    const partialPressure = moleFraction * pressurePa
    const vaporPressure = calculateVaporPressure(temperature, comp)
    const dewpoint = calculateDewpoint(partialPressure, comp)

    dewpointResults.push({ component: comp, dewpoint, partialPressure, vaporPressure })
  }

  const validDewpoints = dewpointResults.filter((r) => !isNaN(r.dewpoint))
  const overallDewpoint = validDewpoints.length > 0
    ? Math.max(...validDewpoints.map((r) => r.dewpoint))
    : NaN

  const limitingComponent = validDewpoints.length > 0
    ? validDewpoints.reduce((a, b) => (a.dewpoint > b.dewpoint ? a : b)).component
    : 'Unknown'

  const margin = temperature - overallDewpoint
  const risk = getRiskLevel(margin)

  return { dewpoints: dewpointResults, overallDewpoint, limitingComponent, margin, risk, warnings, engine: 'client-fallback' }
}

/**
 * Call the shared Python backend for acid gas dewpoint calculation.
 * Falls back to client-side Antoine equations when the API is unreachable.
 * See issue #609.
 */
async function fetchFromBackend(
  temperature: number,
  pressure: number,
  composition: Record<string, number>,
): Promise<CalculationResults> {
  const response = await fetch(`${CALC_API_BASE}/api/calc/acid-gas-dewpoint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      temperature_c: temperature,
      pressure_bar: pressure,
      h2o_fraction: (composition.H2O || 0) / 100,
      hf_fraction: (composition.HF || 0) / 100,
      hcl_fraction: (composition.HCl || 0) / 100,
      h2s_fraction: (composition.H2S || 0) / 100,
      method: 'antoine',
    }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? `API error ${response.status}`)
  }

  const data = await response.json()

  // Map backend response to frontend CalculationResults shape
  const componentKeys = ['H2O', 'HF', 'HCl', 'H2S'] as const
  const dewpoints: DewpointResult[] = componentKeys.map((comp) => ({
    component: comp,
    dewpoint: data.components[comp]?.dewpoint_c ?? NaN,
    partialPressure: data.components[comp]?.partial_pressure_pa ?? 0,
    vaporPressure: data.components[comp]?.vapor_pressure_pa ?? 0,
  }))

  return {
    dewpoints,
    overallDewpoint: data.overall_dewpoint_c ?? NaN,
    limitingComponent: data.limiting_component,
    margin: data.dewpoint_margin_c ?? NaN,
    risk: data.condensation_risk,
    warnings: data.warnings ?? [],
    engine: 'python-backend',
  }
}

export function AcidGasDewpointCalculator() {
  const [temperature, setTemperature] = useState(150)
  const [pressure, setPressure] = useState(30)
  const [preset, setPreset] = useState('Typical Syngas')
  const [composition, setComposition] = useState(PRESETS['Typical Syngas'])
  const [results, setResults] = useState<CalculationResults | null>(null)
  const [loading, setLoading] = useState(false)

  const handlePresetChange = useCallback((newPreset: string) => {
    setPreset(newPreset)
    if (PRESETS[newPreset]) {
      setComposition(PRESETS[newPreset])
    }
  }, [])

  const handleCompositionChange = useCallback((component: string, value: number) => {
    setComposition((prev) => ({ ...prev, [component]: value }))
    setPreset('Custom')
  }, [])

  const calculate = useCallback(async () => {
    setLoading(true)
    try {
      // Try the validated Python backend first; fall back to client-side
      // Antoine calculations if the API is unreachable.  See issue #609.
      let res: CalculationResults
      try {
        res = await fetchFromBackend(temperature, pressure, composition)
      } catch {
        res = calculateFallback(temperature, pressure, composition)
      }
      setResults(res)
    } finally {
      setLoading(false)
    }
  }, [temperature, pressure, composition])

  const chartData = results?.dewpoints.map((d) => ({
    name: d.component,
    dewpoint: isNaN(d.dewpoint) ? 0 : d.dewpoint,
    hasData: !isNaN(d.dewpoint),
  })) || []

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Input Panel */}
      <div className="lg:col-span-1 space-y-6">
        {/* Operating Conditions */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Operating Conditions</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-slate-300 mb-1">Temperature (°C)</label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Pressure (bar)</label>
              <input
                type="number"
                value={pressure}
                onChange={(e) => setPressure(Number(e.target.value))}
                step="0.1"
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Composition Preset */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Composition Preset</h2>
          <select
            value={preset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
          >
            {Object.keys(PRESETS).map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
            <option value="Custom">Custom</option>
          </select>
        </div>

        {/* Gas Composition */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Gas Composition (mol%)</h2>
          <div className="space-y-3">
            {[
              { key: 'H2O', label: 'Water Vapor (H₂O)', max: 100 },
              { key: 'HF', label: 'Hydrogen Fluoride (HF)', max: 10 },
              { key: 'HCl', label: 'Hydrogen Chloride (HCl)', max: 10 },
              { key: 'H2S', label: 'Hydrogen Sulfide (H₂S)', max: 10 },
            ].map(({ key, label, max }) => (
              <div key={key}>
                <label className="block text-sm text-slate-300 mb-1">{label}</label>
                <input
                  type="number"
                  value={composition[key] || 0}
                  onChange={(e) => handleCompositionChange(key, Number(e.target.value))}
                  step="0.01"
                  min="0"
                  max={max}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Calculate Button */}
        <button
          onClick={calculate}
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Calculating...' : 'Calculate Dewpoints'}
        </button>
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-2 space-y-6">
        {results && (
          <>
            {/* Summary Card */}
            <div className={`rounded-lg p-6 ${
              results.risk.includes('HIGH') ? 'bg-red-900/50 border border-red-700' :
              results.risk.includes('MEDIUM') ? 'bg-yellow-900/50 border border-yellow-700' :
              'bg-green-900/50 border border-green-700'
            }`}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-slate-400 text-sm">Overall Dewpoint</p>
                  <p className="text-2xl font-bold text-white">
                    {isNaN(results.overallDewpoint) ? 'N/A' : `${results.overallDewpoint.toFixed(1)}°C`}
                  </p>
                </div>
                <div>
                  <p className="text-slate-400 text-sm">Limiting Component</p>
                  <p className="text-2xl font-bold text-white">{results.limitingComponent}</p>
                </div>
                <div>
                  <p className="text-slate-400 text-sm">Temperature Margin</p>
                  <p className="text-2xl font-bold text-white">
                    {isNaN(results.margin) ? 'N/A' : `${results.margin.toFixed(1)}°C`}
                  </p>
                </div>
                <div>
                  <p className="text-slate-400 text-sm">Risk Level</p>
                  <p className={`text-lg font-semibold ${
                    results.risk.includes('HIGH') ? 'text-red-400' :
                    results.risk.includes('MEDIUM') ? 'text-yellow-400' :
                    'text-green-400'
                  }`}>
                    {results.risk.split(' - ')[0]}
                  </p>
                </div>
              </div>
            </div>

            {/* Dewpoint Chart */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Dewpoint Comparison</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" label={{ value: '°C', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                      labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <ReferenceLine
                      y={temperature}
                      stroke="#fbbf24"
                      strokeDasharray="5 5"
                      label={{ value: `Operating T: ${temperature}°C`, fill: '#fbbf24', position: 'right' }}
                    />
                    <Bar dataKey="dewpoint" fill="#3b82f6" name="Dewpoint (°C)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Detailed Results Table */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Detailed Results</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="py-2 px-3 text-slate-300">Component</th>
                      <th className="py-2 px-3 text-slate-300">Dewpoint (°C)</th>
                      <th className="py-2 px-3 text-slate-300">Partial Pressure (Pa)</th>
                      <th className="py-2 px-3 text-slate-300">Vapor Pressure (Pa)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.dewpoints.map((d) => (
                      <tr key={d.component} className="border-b border-slate-700/50">
                        <td className="py-2 px-3 text-white font-medium">{d.component}</td>
                        <td className="py-2 px-3 text-white">
                          {isNaN(d.dewpoint) ? 'N/A' : d.dewpoint.toFixed(2)}
                        </td>
                        <td className="py-2 px-3 text-white">{d.partialPressure.toFixed(2)}</td>
                        <td className="py-2 px-3 text-white">{d.vaporPressure.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Engine Info -- See issue #609 */}
            <div className="bg-slate-800 rounded-lg p-3 text-right">
              <span className="text-xs text-slate-500">
                Engine: {results.engine}
              </span>
            </div>

            {/* Warnings */}
            {results.warnings.length > 0 && (
              <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-yellow-400 mb-2">Warnings</h2>
                <ul className="list-disc list-inside text-yellow-200 space-y-1">
                  {results.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {!results && (
          <div className="bg-slate-800 rounded-lg p-8 text-center">
            <p className="text-slate-400">
              Enter operating conditions and gas composition, then click "Calculate Dewpoints" to see results.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
