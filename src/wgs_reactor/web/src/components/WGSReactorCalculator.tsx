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
} from 'recharts'

// Physical constants
const R_GAS = 8.314 // J/(mol·K)

interface CompositionResult {
  CO: number
  H2: number
  CO2: number
  H2O: number
}

interface EquilibriumResult {
  conversion: number
  composition: CompositionResult
  h2CoRatio: number
  equilibriumConstant: number
  heatReleased: number
}

interface SizingResult {
  reactorVolume: number
  catalystVolume: number
  diameter: number
  length: number
  heatDuty: number
  ghsv: number
}

interface CalculationResults {
  equilibrium: EquilibriumResult
  sizing: SizingResult
  inlet: CompositionResult
}

function calculateEquilibriumConstant(temperatureK: number): number {
  // CO + H2O <-> CO2 + H2
  // dH = -41.2 kJ/mol, dS = -42.1 J/(mol·K)
  const deltaH = -41200 // J/mol
  const deltaS = -42.1 // J/(mol·K)

  const lnK = -deltaH / (R_GAS * temperatureK) + deltaS / R_GAS
  return Math.exp(lnK)
}

function calculateEquilibrium(
  inlet: CompositionResult,
  temperatureK: number,
  steamRatio: number
): EquilibriumResult {
  // Initial moles
  const nCO0 = inlet.CO
  const nH2O0 = inlet.H2O + inlet.CO * steamRatio
  const nCO20 = inlet.CO2
  const nH20 = inlet.H2

  const nTotal0 = nCO0 + nH2O0 + nCO20 + nH20

  if (nTotal0 === 0) {
    return {
      conversion: 0,
      composition: { CO: 0, H2O: 0, CO2: 0, H2: 0 },
      h2CoRatio: 0,
      equilibriumConstant: calculateEquilibriumConstant(temperatureK),
      heatReleased: 0,
    }
  }

  const Keq = calculateEquilibriumConstant(temperatureK)

  // Solve quadratic for extent of reaction
  const a = Keq - 1
  const b = Keq * (nCO0 + nH2O0) + nCO20 + nH20
  const c = Keq * nCO0 * nH2O0 - nCO20 * nH20

  let xEq: number
  if (Math.abs(a) < 1e-10) {
    xEq = Math.abs(b) > 1e-10 ? -c / b : 0
  } else {
    const discriminant = b * b - 4 * a * c
    if (discriminant < 0) {
      xEq = 0
    } else {
      const x1 = (-b + Math.sqrt(discriminant)) / (2 * a)
      const x2 = (-b - Math.sqrt(discriminant)) / (2 * a)

      const maxExtent = Math.min(nCO0, nH2O0)
      if (x1 >= 0 && x1 <= maxExtent) {
        xEq = x1
      } else if (x2 >= 0 && x2 <= maxExtent) {
        xEq = x2
      } else {
        xEq = Math.max(0, Math.min(x1, maxExtent))
      }
    }
  }

  // Equilibrium composition
  const nCOEq = nCO0 - xEq
  const nH2OEq = nH2O0 - xEq
  const nCO2Eq = nCO20 + xEq
  const nH2Eq = nH20 + xEq
  const nTotalEq = nCOEq + nH2OEq + nCO2Eq + nH2Eq

  const composition: CompositionResult = {
    CO: nTotalEq > 0 ? (nCOEq / nTotalEq) * 100 : 0,
    H2O: nTotalEq > 0 ? (nH2OEq / nTotalEq) * 100 : 0,
    CO2: nTotalEq > 0 ? (nCO2Eq / nTotalEq) * 100 : 0,
    H2: nTotalEq > 0 ? (nH2Eq / nTotalEq) * 100 : 0,
  }

  const conversion = nCO0 > 0 ? (xEq / nCO0) * 100 : 0
  const h2CoRatio = composition.CO > 0 ? composition.H2 / composition.CO : Infinity
  const heatReleased = xEq * 41.2

  return {
    conversion,
    composition,
    h2CoRatio,
    equilibriumConstant: Keq,
    heatReleased,
  }
}

function sizeReactor(feedRate: number, conversion: number): SizingResult {
  const ghsv = 3000 // h^-1
  const reactorVolume = feedRate / ghsv
  const catalystVolume = reactorVolume * 0.8

  const ldRatio = 3.0
  const diameter = Math.pow((4 * reactorVolume) / (Math.PI * ldRatio), 1 / 3)
  const length = diameter * ldRatio

  const heatDuty = (feedRate * conversion) / 100 * 41.2 / 3.6

  return {
    reactorVolume,
    catalystVolume,
    diameter,
    length,
    heatDuty,
    ghsv,
  }
}

export function WGSReactorCalculator() {
  // Reactor configuration
  const [temperature, setTemperature] = useState(400)
  const [pressure, setPressure] = useState(25)
  const [steamRatio, setSteamRatio] = useState(2.0)
  const [feedRate, setFeedRate] = useState(100)

  // Feed composition
  const [inlet, setInlet] = useState<CompositionResult>({
    CO: 25,
    H2: 20,
    CO2: 10,
    H2O: 5,
  })

  // Shift type
  const [shiftType, setShiftType] = useState('HTS')

  const [results, setResults] = useState<CalculationResults | null>(null)

  const handleInletChange = useCallback((species: keyof CompositionResult, value: number) => {
    setInlet((prev) => ({ ...prev, [species]: value }))
  }, [])

  const handleShiftTypeChange = useCallback((type: string) => {
    setShiftType(type)
    if (type === 'HTS') {
      setTemperature(400)
    } else if (type === 'LTS') {
      setTemperature(220)
    } else {
      setTemperature(400)
    }
  }, [])

  const calculate = useCallback(() => {
    const tempK = temperature + 273.15

    const equilibrium = calculateEquilibrium(inlet, tempK, steamRatio)
    const sizing = sizeReactor(feedRate, equilibrium.conversion)

    setResults({
      equilibrium,
      sizing,
      inlet,
    })
  }, [temperature, steamRatio, feedRate, inlet])

  const chartData = results
    ? [
        { name: 'CO', inlet: results.inlet.CO, outlet: results.equilibrium.composition.CO },
        { name: 'H2', inlet: results.inlet.H2, outlet: results.equilibrium.composition.H2 },
        { name: 'CO2', inlet: results.inlet.CO2, outlet: results.equilibrium.composition.CO2 },
        { name: 'H2O', inlet: results.inlet.H2O, outlet: results.equilibrium.composition.H2O },
      ]
    : []

  const getCatalystInfo = () => {
    switch (shiftType) {
      case 'HTS':
        return 'Catalyst: Fe-Cr (350-450°C)'
      case 'LTS':
        return 'Catalyst: Cu-Zn-Al (180-250°C)'
      default:
        return 'Stage 1: Fe-Cr, Stage 2: Cu-Zn-Al'
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Input Panel */}
      <div className="lg:col-span-1 space-y-6">
        {/* Reactor Configuration */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Reactor Configuration</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-slate-300 mb-1">Shift Type</label>
              <select
                value={shiftType}
                onChange={(e) => handleShiftTypeChange(e.target.value)}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              >
                <option value="HTS">High Temperature Shift (HTS)</option>
                <option value="LTS">Low Temperature Shift (LTS)</option>
                <option value="TWO">Two-Stage (HTS + LTS)</option>
              </select>
              <p className="text-xs text-slate-400 mt-1">{getCatalystInfo()}</p>
            </div>
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
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Steam/CO Ratio</label>
              <input
                type="number"
                value={steamRatio}
                onChange={(e) => setSteamRatio(Number(e.target.value))}
                step="0.1"
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Feed Rate (kmol/h)</label>
              <input
                type="number"
                value={feedRate}
                onChange={(e) => setFeedRate(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Feed Composition */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Feed Composition (mol%)</h2>
          <div className="space-y-3">
            {(['CO', 'H2', 'CO2', 'H2O'] as const).map((species) => (
              <div key={species}>
                <label className="block text-sm text-slate-300 mb-1">{species}</label>
                <input
                  type="number"
                  value={inlet[species]}
                  onChange={(e) => handleInletChange(species, Number(e.target.value))}
                  step="0.1"
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Calculate Button */}
        <button
          onClick={calculate}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
        >
          Calculate WGS Performance
        </button>
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-2 space-y-6">
        {results && (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`rounded-lg p-4 ${
                results.equilibrium.conversion > 80 ? 'bg-green-900/50' :
                results.equilibrium.conversion > 50 ? 'bg-yellow-900/50' : 'bg-red-900/50'
              }`}>
                <p className="text-slate-400 text-sm">CO Conversion</p>
                <p className="text-2xl font-bold text-white">
                  {results.equilibrium.conversion.toFixed(1)}%
                </p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">H2/CO Ratio</p>
                <p className="text-2xl font-bold text-white">
                  {results.equilibrium.h2CoRatio === Infinity
                    ? '∞'
                    : results.equilibrium.h2CoRatio.toFixed(2)}
                </p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Heat Duty</p>
                <p className="text-2xl font-bold text-white">
                  {results.sizing.heatDuty.toFixed(1)} kW
                </p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Equilibrium K</p>
                <p className="text-2xl font-bold text-white">
                  {results.equilibrium.equilibriumConstant.toFixed(2)}
                </p>
              </div>
            </div>

            {/* Composition Chart */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Composition Comparison</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" label={{ value: 'mol%', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                      labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <Bar dataKey="inlet" fill="#f97316" name="Inlet (mol%)" />
                    <Bar dataKey="outlet" fill="#3b82f6" name="Outlet (mol%)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Reactor Sizing Table */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Reactor Sizing</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="py-2 px-3 text-slate-300">Parameter</th>
                      <th className="py-2 px-3 text-slate-300">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ['Reactor Volume', `${results.sizing.reactorVolume.toFixed(2)} m³`],
                      ['Catalyst Volume', `${results.sizing.catalystVolume.toFixed(2)} m³`],
                      ['Diameter', `${results.sizing.diameter.toFixed(2)} m`],
                      ['Length', `${results.sizing.length.toFixed(2)} m`],
                      ['GHSV', `${results.sizing.ghsv} h⁻¹`],
                      ['Heat Released', `${results.equilibrium.heatReleased.toFixed(1)} kJ/mol CO`],
                    ].map(([param, value]) => (
                      <tr key={param} className="border-b border-slate-700/50">
                        <td className="py-2 px-3 text-white">{param}</td>
                        <td className="py-2 px-3 text-white">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {!results && (
          <div className="bg-slate-800 rounded-lg p-8 text-center">
            <p className="text-slate-400">
              Enter operating conditions and click "Calculate WGS Performance" to see results.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
