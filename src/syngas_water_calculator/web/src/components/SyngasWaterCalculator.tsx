/**
 * Syngas Water Calculator - React Web Component
 *
 * Calculates water content and dew point in syngas systems.
 * Matches PyQt6 functionality with vapor pressure methods,
 * composition presets, and condensation risk assessment.
 *
 * See issue #608.
 */

import { useState, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

// Composition presets matching PyQt6 version
const COMPOSITION_PRESETS: Record<string, Record<string, number>> = {
  typical_syngas: { H2: 25, CO: 40, CO2: 15, H2O: 10, N2: 8, CH4: 2 },
  biomass_syngas: { H2: 15, CO: 20, CO2: 15, H2O: 20, N2: 25, CH4: 5 },
  coal_syngas: { H2: 30, CO: 45, CO2: 10, H2O: 5, N2: 8, CH4: 2 },
  natural_gas_reforming: { H2: 50, CO: 10, CO2: 5, H2O: 30, N2: 2, CH4: 3 },
}

const PRESET_LABELS: Record<string, string> = {
  typical_syngas: 'Typical Syngas',
  biomass_syngas: 'Biomass Gasification',
  coal_syngas: 'Coal Gasification',
  natural_gas_reforming: 'Natural Gas Reforming',
}

type VaporPressureMethod = 'auto' | 'antoine' | 'buck' | 'iapws' | 'magnus'

const METHOD_LABELS: Record<VaporPressureMethod, string> = {
  auto: 'Auto (Recommended)',
  antoine: 'Antoine',
  buck: 'Buck',
  iapws: 'IAPWS-IF97',
  magnus: 'Magnus',
}

// Antoine equation coefficients for water (NIST)
const ANTOINE_A = 8.07131
const ANTOINE_B = 1730.63
const ANTOINE_C = 233.426

// Calculate vapor pressure using different methods
function calculateVaporPressure(tempC: number, method: VaporPressureMethod): number {
  if (method === 'auto') {
    // Use Buck for -40..50 C, Antoine for 50..374 C
    method = tempC < 50 ? 'buck' : 'antoine'
  }

  switch (method) {
    case 'antoine': {
      // Antoine equation: log10(P_mmHg) = A - B/(C+T)
      const logP = ANTOINE_A - ANTOINE_B / (ANTOINE_C + tempC)
      return Math.pow(10, logP) * 133.322 / 1e5 // mmHg to bar
    }
    case 'buck': {
      // Buck (1981) equation
      const P_hPa = 6.1121 * Math.exp((18.678 - tempC / 234.5) * tempC / (257.14 + tempC))
      return P_hPa / 1000 // hPa to bar
    }
    case 'iapws': {
      // Simplified IAPWS-IF97 approximation
      const T_K = tempC + 273.15
      const Tc = 647.096 // K
      const Pc = 220.64 // bar
      const tau = 1 - T_K / Tc
      const a = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502]
      const b = [1, 1.5, 3, 3.5, 4, 7.5]
      let sum = 0
      for (let i = 0; i < 6; i++) {
        sum += a[i] * Math.pow(tau, b[i])
      }
      return Pc * Math.exp((Tc / T_K) * sum)
    }
    case 'magnus': {
      // Magnus formula
      const P_hPa = 6.1078 * Math.exp((17.27 * tempC) / (tempC + 237.3))
      return P_hPa / 1000 // hPa to bar
    }
    default:
      return 0
  }
}

interface CalculationResults {
  moleFractionWater: number
  waterContentMgNm3: number
  waterContentPpmv: number
  waterContentGM3: number
  waterContentLbMmscf: number
  vaporPressureBar: number
  dewPointC: number
}

interface RiskAssessment {
  temperatureMarginC: number
  riskLevel: string
  recommendedMinTempC: number
}

function calculateWaterContent(
  tempC: number,
  pressureBar: number,
  compositionKey: string,
  method: VaporPressureMethod
): CalculationResults {
  const composition = COMPOSITION_PRESETS[compositionKey]
  const waterMolPct = composition.H2O || 0

  // Vapor pressure of pure water at operating temperature
  const vaporPressure = calculateVaporPressure(tempC, method)

  // Mole fraction of water in syngas (from composition)
  const moleFraction = waterMolPct / 100

  // Saturation mole fraction at operating conditions
  const saturationMoleFraction = vaporPressure / pressureBar

  // Effective water mole fraction (limited by saturation)
  const effectiveMoleFraction = Math.min(moleFraction, saturationMoleFraction)

  // Convert to various units
  const ppmv = effectiveMoleFraction * 1e6

  // mg/Nm3: water at STP (0C, 1 atm)
  // 1 mol ideal gas at STP = 22.414 L = 0.022414 m3
  // MW water = 18.015 g/mol
  const mgPerNm3 = effectiveMoleFraction * (18.015 / 0.022414) * 1000

  // g/m3 at actual conditions
  const gPerM3 = effectiveMoleFraction * (18.015 * pressureBar * 1e5) / (8.314 * (tempC + 273.15))

  // lb/MMscf (standard cubic feet)
  const lbPerMmscf = ppmv * 18.015 / (385.5 * 453.592) * 1e6

  // Dew point: temperature at which vapor pressure equals partial pressure of water
  const partialPressure = effectiveMoleFraction * pressureBar
  // Inverse Antoine to find dew point
  const dewPointC = partialPressure > 0
    ? ANTOINE_B / (ANTOINE_A - Math.log10(partialPressure * 1e5 / 133.322)) - ANTOINE_C
    : -273.15

  return {
    moleFractionWater: effectiveMoleFraction,
    waterContentMgNm3: mgPerNm3,
    waterContentPpmv: ppmv,
    waterContentGM3: gPerM3,
    waterContentLbMmscf: lbPerMmscf,
    vaporPressureBar: vaporPressure,
    dewPointC,
  }
}

function assessCondensationRisk(tempC: number, dewPointC: number): RiskAssessment {
  const margin = tempC - dewPointC
  let riskLevel: string
  const safetyMargin = 15 // degrees C

  if (margin < 0) {
    riskLevel = 'Critical - Condensation Occurring'
  } else if (margin < 5) {
    riskLevel = 'High'
  } else if (margin < 15) {
    riskLevel = 'Medium'
  } else {
    riskLevel = 'Low'
  }

  return {
    temperatureMarginC: margin,
    riskLevel,
    recommendedMinTempC: dewPointC + safetyMargin,
  }
}

// Generate dew point curve data for chart
function generateDewPointCurve(
  pressureBar: number,
  compositionKey: string,
  method: VaporPressureMethod
): Array<{ pressure: number; dewPoint: number }> {
  const data = []
  for (let p = 1; p <= Math.max(pressureBar * 1.5, 50); p += 1) {
    const result = calculateWaterContent(200, p, compositionKey, method)
    data.push({ pressure: p, dewPoint: result.dewPointC })
  }
  return data
}

export function SyngasWaterCalculator() {
  const [temperature, setTemperature] = useState(40)
  const [pressure, setPressure] = useState(30)
  const [compositionKey, setCompositionKey] = useState('typical_syngas')
  const [method, setMethod] = useState<VaporPressureMethod>('auto')
  const [results, setResults] = useState<CalculationResults | null>(null)
  const [risk, setRisk] = useState<RiskAssessment | null>(null)

  const calculate = useCallback(() => {
    const r = calculateWaterContent(temperature, pressure, compositionKey, method)
    setResults(r)
    setRisk(assessCondensationRisk(temperature, r.dewPointC))
  }, [temperature, pressure, compositionKey, method])

  const chartData = results
    ? generateDewPointCurve(pressure, compositionKey, method)
    : []

  const getRiskColor = (level: string) => {
    if (level.includes('Critical')) return 'text-red-400'
    if (level === 'High') return 'text-orange-400'
    if (level === 'Medium') return 'text-yellow-400'
    return 'text-green-400'
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-2xl font-bold text-blue-400 mb-6">
        Syngas Water Calculator
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="space-y-4">
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">Input Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-slate-300 mb-1">Temperature (C)</label>
                <input
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  min={-50}
                  max={400}
                  step={0.1}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Pressure (bar)</label>
                <input
                  type="number"
                  value={pressure}
                  onChange={(e) => setPressure(Number(e.target.value))}
                  min={0.1}
                  max={500}
                  step={0.1}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Gas Composition</label>
                <select
                  value={compositionKey}
                  onChange={(e) => setCompositionKey(e.target.value)}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                >
                  {Object.entries(PRESET_LABELS).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Calculation Method</label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value as VaporPressureMethod)}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                >
                  {Object.entries(METHOD_LABELS).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Composition display */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-slate-300 mb-2">
              Composition ({PRESET_LABELS[compositionKey]})
            </h3>
            <div className="grid grid-cols-2 gap-1">
              {Object.entries(COMPOSITION_PRESETS[compositionKey]).map(([gas, pct]) => (
                <div key={gas} className="flex justify-between text-xs">
                  <span className="text-slate-400">{gas}</span>
                  <span className="text-white">{pct}%</span>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={calculate}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
          >
            Calculate Water Content
          </button>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {results && risk ? (
            <>
              {/* Water Content Results */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Water Content Results</h2>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    ['Mole Fraction', results.moleFractionWater.toFixed(6)],
                    ['Water Content', `${results.waterContentMgNm3.toFixed(2)} mg/Nm3`],
                    ['Water Content', `${results.waterContentPpmv.toFixed(1)} ppmv`],
                    ['Water Content', `${results.waterContentGM3.toFixed(4)} g/m3`],
                    ['Water Content', `${results.waterContentLbMmscf.toFixed(2)} lb/MMscf`],
                    ['Vapor Pressure', `${results.vaporPressureBar.toFixed(4)} bar`],
                    ['Dew Point', `${results.dewPointC.toFixed(1)} C`],
                  ].map(([label, value], idx) => (
                    <div key={idx} className="flex justify-between items-center p-2 bg-slate-700/50 rounded">
                      <span className="text-sm text-slate-400">{label}</span>
                      <span className="text-sm font-medium text-green-400">{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Risk Assessment */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Condensation Risk Assessment</h2>
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-3 bg-slate-700/50 rounded text-center">
                    <p className="text-xs text-slate-400">Temperature Margin</p>
                    <p className={`text-xl font-bold ${getRiskColor(risk.riskLevel)}`}>
                      {risk.temperatureMarginC.toFixed(1)} C
                    </p>
                  </div>
                  <div className="p-3 bg-slate-700/50 rounded text-center">
                    <p className="text-xs text-slate-400">Risk Level</p>
                    <p className={`text-xl font-bold ${getRiskColor(risk.riskLevel)}`}>
                      {risk.riskLevel}
                    </p>
                  </div>
                  <div className="p-3 bg-slate-700/50 rounded text-center">
                    <p className="text-xs text-slate-400">Recommended Min Temp</p>
                    <p className="text-xl font-bold text-yellow-400">
                      {risk.recommendedMinTempC.toFixed(1)} C
                    </p>
                  </div>
                </div>
              </div>

              {/* Dew Point Chart */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Dew Point vs Pressure</h2>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                      <XAxis
                        dataKey="pressure"
                        stroke="#94a3b8"
                        label={{ value: 'Pressure (bar)', position: 'bottom', fill: '#94a3b8', offset: -5 }}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        label={{ value: 'Dew Point (C)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        labelStyle={{ color: '#f1f5f9' }}
                        formatter={(value: number) => [`${value.toFixed(1)} C`, 'Dew Point']}
                        labelFormatter={(label) => `${label} bar`}
                      />
                      <ReferenceLine
                        x={pressure}
                        stroke="#f59e0b"
                        strokeDasharray="3 3"
                        label={{ value: 'Operating', fill: '#f59e0b', fontSize: 10 }}
                      />
                      <ReferenceLine
                        y={temperature}
                        stroke="#ef4444"
                        strokeDasharray="3 3"
                        label={{ value: 'Operating T', fill: '#ef4444', fontSize: 10 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="dewPoint"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-slate-800 rounded-lg p-8 text-center">
              <p className="text-slate-400">
                Enter operating conditions and click "Calculate Water Content" to see results.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
