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
const R_GAS = 8314.0 // J/(kmol·K)
const WATER_VISCOSITY = 0.001 // Pa·s
const WATER_CP = 4186.0 // J/(kg·K)
const LATENT_HEAT_WATER = 2.26e6 // J/kg
const GRAVITY = 9.81 // m/s²

// Molecular weights
const MW_GASES: Record<string, number> = {
  HCl: 36.458,
  SO2: 64.06,
  H2S: 34.08,
  HF: 20.01,
}

const MW_NAOH = 40.0

// Stoichiometric ratios for NaOH neutralization
const NAOH_STOICH: Record<string, number> = {
  HCl: 1.0,
  SO2: 2.0,
  H2S: 2.0,
  HF: 1.0,
}

// Packing database
interface PackingProperties {
  name: string
  material: string
  nominalSize: number
  specificSurfaceArea: number
  voidFraction: number
  packingFactor: number
  cFlood: number
  ch: number
  n: number
}

const PACKING_DATABASE: Record<string, PackingProperties> = {
  'Ceramic Raschig Rings': {
    name: 'Ceramic Raschig Rings',
    material: 'Ceramic',
    nominalSize: 50,
    specificSurfaceArea: 95,
    voidFraction: 0.74,
    packingFactor: 155,
    cFlood: 0.082,
    ch: 0.85,
    n: 0.30,
  },
  'Metal Pall Rings': {
    name: 'Metal Pall Rings',
    material: 'Stainless Steel',
    nominalSize: 50,
    specificSurfaceArea: 112,
    voidFraction: 0.95,
    packingFactor: 66,
    cFlood: 0.11,
    ch: 0.65,
    n: 0.28,
  },
  'Plastic Cascade Rings': {
    name: 'Plastic Cascade Rings',
    material: 'Polypropylene',
    nominalSize: 50,
    specificSurfaceArea: 105,
    voidFraction: 0.92,
    packingFactor: 72,
    cFlood: 0.10,
    ch: 0.75,
    n: 0.29,
  },
  'Structured Packing': {
    name: 'Structured Packing (250Y)',
    material: 'Stainless Steel',
    nominalSize: 250,
    specificSurfaceArea: 250,
    voidFraction: 0.98,
    packingFactor: 33,
    cFlood: 0.15,
    ch: 0.45,
    n: 0.22,
  },
}

interface AcidGasInput {
  inletPpmv: number
  removalPct: number
}

interface AcidGasResult {
  name: string
  inletPpmv: number
  outletPpmv: number
  removedKgHr: number
  ntu: number
}

interface CalculationResults {
  diameter: number
  packedHeight: number
  pressureDrop: number
  naohPureKgHr: number
  naohSolutionLHr: number
  gasDensity: number
  floodingVelocity: number
  designVelocity: number
  crossSection: number
  htu: number
  maxNtu: number
  totalHeatKw: number
  sensibleHeatKw: number
  latentHeatKw: number
  coolingWaterLMin: number
  saltProducedKgHr: number
  acidGasDetails: AcidGasResult[]
}

function calculateGasDensity(tempK: number, pressurePa: number, mw: number): number {
  return (pressurePa * mw) / (R_GAS * tempK)
}

function calculateFloodingVelocity(
  liquidMassFlux: number,
  gasDensity: number,
  liquidDensity: number,
  packing: PackingProperties
): number {
  const flowParam = liquidMassFlux * Math.sqrt(gasDensity / liquidDensity)
  const yFlood = packing.cFlood * Math.exp(-1.5 * Math.pow(flowParam, 0.5))

  const gFloodSquared =
    (yFlood * gasDensity * liquidDensity * GRAVITY) /
    (packing.packingFactor * Math.pow(WATER_VISCOSITY / WATER_VISCOSITY, 0.1))

  const gFlood = Math.sqrt(Math.max(0, gFloodSquared))
  return gasDensity > 0 ? gFlood / gasDensity : 0
}

function calculatePressureDrop(
  gasVelocity: number,
  gasDensity: number,
  liquidMassFlux: number,
  liquidDensity: number,
  packing: PackingProperties,
  packedHeight: number
): number {
  const gGas = gasVelocity * gasDensity
  const flowParam = (liquidMassFlux / Math.max(gGas, 0.001)) * Math.sqrt(gasDensity / liquidDensity)

  const y =
    (Math.pow(gGas, 2) * packing.packingFactor * Math.pow(WATER_VISCOSITY / WATER_VISCOSITY, 0.1)) /
    (gasDensity * liquidDensity * GRAVITY)

  const alpha = 85.0
  const beta = 1.1
  const gamma = 3.5

  let dpPerM = alpha * Math.pow(y, beta) * (1.0 + gamma * flowParam)
  dpPerM = Math.min(dpPerM, 2000.0)

  return dpPerM * packedHeight
}

function calculateNtu(inletFrac: number, outletFrac: number): number {
  if (outletFrac <= 0 || inletFrac <= 0 || inletFrac <= outletFrac) return 0
  return Math.log(inletFrac / outletFrac)
}

function calculateHtu(
  gasMassFlux: number,
  liquidMassFlux: number,
  packing: PackingProperties,
  kla: number
): number {
  const klaPerS = kla / 3600.0
  if (klaPerS <= 0) return 1.0

  const lOverG = liquidMassFlux / Math.max(gasMassFlux, 0.001)
  let htu = packing.ch / (klaPerS * packing.specificSurfaceArea * Math.pow(lOverG, packing.n))

  return Math.max(0.1, Math.min(3.0, htu))
}

export function ScrubberCalculator() {
  // Gas conditions
  const [gasFlow, setGasFlow] = useState(10000)
  const [inletTemp, setInletTemp] = useState(200)
  const [pressure, setPressure] = useState(1.5)
  const [mw, setMw] = useState(22)
  const [outletTemp, setOutletTemp] = useState(38)

  // Packing
  const [packingType, setPackingType] = useState('Metal Pall Rings')
  const [percentFlood, setPercentFlood] = useState(70)
  const [safetyFactor, setSafetyFactor] = useState(1.2)

  // Acid gases
  const [acidGases, setAcidGases] = useState<Record<string, AcidGasInput>>({
    HCl: { inletPpmv: 500, removalPct: 99.0 },
    SO2: { inletPpmv: 200, removalPct: 95.0 },
    H2S: { inletPpmv: 1000, removalPct: 90.0 },
    HF: { inletPpmv: 100, removalPct: 99.0 },
  })

  // Liquid conditions
  const [lgRatio, setLgRatio] = useState(3.0)
  const [causticConc, setCausticConc] = useState(20)
  const [cwInletTemp, setCwInletTemp] = useState(25)
  const [kla, setKla] = useState(200)

  const [results, setResults] = useState<CalculationResults | null>(null)

  const handleAcidGasChange = useCallback((gas: string, field: 'inletPpmv' | 'removalPct', value: number) => {
    setAcidGases((prev) => ({
      ...prev,
      [gas]: { ...prev[gas], [field]: value },
    }))
  }, [])

  const calculate = useCallback(() => {
    const packing = PACKING_DATABASE[packingType]
    if (!packing) return

    // Calculate gas properties
    const tempK = inletTemp + 273.15
    const pressurePa = pressure * 1e5
    const gasDensity = calculateGasDensity(tempK, pressurePa, mw)

    // Liquid properties
    const liquidFlowKgHr = gasFlow * lgRatio
    const liquidDensity = 1000.0 + 10.8 * causticConc

    // Initial estimate of cross-section area
    let estimatedArea = 2.0
    let liquidMassFlux = (liquidFlowKgHr / 3600.0) / estimatedArea

    // Calculate flooding velocity
    const floodingVelocity = calculateFloodingVelocity(
      liquidMassFlux,
      gasDensity,
      liquidDensity,
      packing
    )

    // Calculate design velocity and column diameter
    const designVelocity = floodingVelocity * (percentFlood / 100.0)
    const gasFlowM3s = gasFlow / (gasDensity * 3600.0)
    const crossSection = designVelocity > 0 ? gasFlowM3s / designVelocity : 0
    const diameter = Math.sqrt((4.0 * crossSection) / Math.PI)

    // Recalculate fluxes with actual area
    liquidMassFlux = crossSection > 0 ? (liquidFlowKgHr / 3600.0) / crossSection : 0
    const gasMassFlux = crossSection > 0 ? (gasFlow / 3600.0) / crossSection : 0

    // Calculate NTU and HTU for each acid gas
    const acidGasDetails: AcidGasResult[] = []
    const acidGasRemoved: Record<string, number> = {}
    let maxNtu = 0

    const gasMolarFlow = gasFlow / mw // kmol/hr

    for (const [gasName, input] of Object.entries(acidGases)) {
      if (input.inletPpmv > 0 && input.removalPct > 0) {
        const inletFrac = input.inletPpmv / 1e6
        const outletPpmv = input.inletPpmv * (1 - input.removalPct / 100.0)
        const outletFrac = outletPpmv / 1e6

        const ntu = calculateNtu(inletFrac, outletFrac)
        maxNtu = Math.max(maxNtu, ntu)

        const mwGas = MW_GASES[gasName] || 30.0
        const removedKmolHr = gasMolarFlow * (inletFrac - outletFrac)
        const removedKgHr = removedKmolHr * mwGas

        acidGasDetails.push({
          name: gasName,
          inletPpmv: input.inletPpmv,
          outletPpmv,
          removedKgHr,
          ntu,
        })
        acidGasRemoved[gasName] = removedKgHr
      }
    }

    // Calculate HTU
    const htu = calculateHtu(gasMassFlux, liquidMassFlux, packing, kla)

    // Calculate packed height
    const packedHeight = maxNtu * htu * safetyFactor

    // Calculate pressure drop
    const pressureDrop = calculatePressureDrop(
      designVelocity,
      gasDensity,
      liquidMassFlux,
      liquidDensity,
      packing,
      packedHeight
    )

    // Calculate caustic requirement
    let naohTotalKgHr = 0
    let saltTotalKgHr = 0
    const saltMw: Record<string, number> = { HCl: 58.44, SO2: 142.04, H2S: 78.04, HF: 41.99 }

    for (const [gasName, massRemoved] of Object.entries(acidGasRemoved)) {
      const stoich = NAOH_STOICH[gasName] || 1.0
      const mwGas = MW_GASES[gasName] || 30.0
      const molesGasHr = (massRemoved * 1000.0) / mwGas
      const naohMolHr = molesGasHr * stoich * 1.15 // 15% excess
      naohTotalKgHr += (naohMolHr * MW_NAOH) / 1000.0

      const saltMolHr = molesGasHr * (stoich === 2.0 ? 0.5 : 1.0)
      saltTotalKgHr += (saltMolHr * (saltMw[gasName] || 100)) / 1000.0
    }

    const naohSolutionKgHr = causticConc > 0 ? naohTotalKgHr / (causticConc / 100.0) : 0
    const solutionDensity = 1000.0 + 10.8 * causticConc
    const naohSolutionLHr = (naohSolutionKgHr / solutionDensity) * 1000.0

    // Calculate heat transfer duty
    const gasFlowKgS = gasFlow / 3600.0
    const deltaT = inletTemp - outletTemp
    const gasCp = 1100.0 // J/(kg·K) typical syngas

    const sensibleHeatW = gasFlowKgS * gasCp * deltaT
    const waterCondensed = gasFlow * 0.15 * deltaT / 100.0
    const waterCondensedKgS = waterCondensed / 3600.0
    const latentHeatW = waterCondensedKgS * LATENT_HEAT_WATER
    const totalHeatW = sensibleHeatW + latentHeatW

    // Calculate cooling water requirement
    const waterOutletTemp = outletTemp - 5.0 // 5°C approach
    const deltaTWater = waterOutletTemp - cwInletTemp
    let coolingWaterLMin = 0
    if (deltaTWater > 0) {
      const waterFlowKgS = totalHeatW / (WATER_CP * deltaTWater)
      coolingWaterLMin = waterFlowKgS * 60.0
    }

    setResults({
      diameter,
      packedHeight,
      pressureDrop,
      naohPureKgHr: naohTotalKgHr,
      naohSolutionLHr,
      gasDensity,
      floodingVelocity,
      designVelocity,
      crossSection,
      htu,
      maxNtu,
      totalHeatKw: totalHeatW / 1000.0,
      sensibleHeatKw: sensibleHeatW / 1000.0,
      latentHeatKw: latentHeatW / 1000.0,
      coolingWaterLMin,
      saltProducedKgHr: saltTotalKgHr,
      acidGasDetails,
    })
  }, [
    gasFlow, inletTemp, pressure, mw, outletTemp,
    packingType, percentFlood, safetyFactor,
    acidGases, lgRatio, causticConc, cwInletTemp, kla,
  ])

  const chartData = results?.acidGasDetails.map((d) => ({
    name: d.name,
    ntu: d.ntu,
    removed: d.removedKgHr,
  })) || []

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Input Panel */}
      <div className="lg:col-span-1 space-y-6">
        {/* Gas Conditions */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Gas Conditions</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-slate-300 mb-1">Gas Flow Rate (kg/hr)</label>
              <input
                type="number"
                value={gasFlow}
                onChange={(e) => setGasFlow(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Inlet Temperature (°C)</label>
              <input
                type="number"
                value={inletTemp}
                onChange={(e) => setInletTemp(Number(e.target.value))}
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
            <div>
              <label className="block text-sm text-slate-300 mb-1">Avg. Molecular Weight (kg/kmol)</label>
              <input
                type="number"
                value={mw}
                onChange={(e) => setMw(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Target Outlet Temp (°C)</label>
              <input
                type="number"
                value={outletTemp}
                onChange={(e) => setOutletTemp(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Packing Selection */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Packing Selection</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-slate-300 mb-1">Packing Type</label>
              <select
                value={packingType}
                onChange={(e) => setPackingType(e.target.value)}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              >
                {Object.keys(PACKING_DATABASE).map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">% of Flooding</label>
              <input
                type="number"
                value={percentFlood}
                onChange={(e) => setPercentFlood(Number(e.target.value))}
                min={50}
                max={90}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Height Safety Factor</label>
              <input
                type="number"
                value={safetyFactor}
                onChange={(e) => setSafetyFactor(Number(e.target.value))}
                step="0.1"
                min={1.0}
                max={2.0}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Acid Gas Composition */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Acid Gas Composition</h2>
          <div className="space-y-3">
            {Object.entries(acidGases).map(([gas, values]) => (
              <div key={gas} className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">{gas} (ppmv)</label>
                  <input
                    type="number"
                    value={values.inletPpmv}
                    onChange={(e) => handleAcidGasChange(gas, 'inletPpmv', Number(e.target.value))}
                    className="w-full bg-slate-700 text-white rounded px-2 py-1 text-sm border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Removal (%)</label>
                  <input
                    type="number"
                    value={values.removalPct}
                    onChange={(e) => handleAcidGasChange(gas, 'removalPct', Number(e.target.value))}
                    step="0.1"
                    className="w-full bg-slate-700 text-white rounded px-2 py-1 text-sm border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Liquid Conditions */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-4">Liquid/Caustic Conditions</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-slate-300 mb-1">L/G Ratio (kg/kg)</label>
              <input
                type="number"
                value={lgRatio}
                onChange={(e) => setLgRatio(Number(e.target.value))}
                step="0.1"
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">NaOH Concentration (wt%)</label>
              <input
                type="number"
                value={causticConc}
                onChange={(e) => setCausticConc(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Cooling Water Inlet (°C)</label>
              <input
                type="number"
                value={cwInletTemp}
                onChange={(e) => setCwInletTemp(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">KLa (1/hr)</label>
              <input
                type="number"
                value={kla}
                onChange={(e) => setKla(Number(e.target.value))}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Calculate Button */}
        <button
          onClick={calculate}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
        >
          Calculate Design
        </button>
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-2 space-y-6">
        {results && (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Column Diameter</p>
                <p className="text-2xl font-bold text-white">{results.diameter.toFixed(2)} m</p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Packed Height</p>
                <p className="text-2xl font-bold text-white">{results.packedHeight.toFixed(2)} m</p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Pressure Drop</p>
                <p className="text-2xl font-bold text-white">{(results.pressureDrop / 1000).toFixed(2)} kPa</p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4">
                <p className="text-slate-400 text-sm">NaOH Requirement</p>
                <p className="text-2xl font-bold text-white">{results.naohPureKgHr.toFixed(1)} kg/hr</p>
              </div>
            </div>

            {/* NTU Chart */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Mass Transfer Analysis</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                      labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <Bar dataKey="ntu" fill="#3b82f6" name="NTU" />
                    <Bar dataKey="removed" fill="#22c55e" name="Removed (kg/hr)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Design Results Table */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Design Results</h2>
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
                      ['Gas Density', `${results.gasDensity.toFixed(3)} kg/m³`],
                      ['Flooding Velocity', `${results.floodingVelocity.toFixed(2)} m/s`],
                      ['Design Velocity', `${results.designVelocity.toFixed(2)} m/s`],
                      ['Column Cross-Section', `${results.crossSection.toFixed(2)} m²`],
                      ['Height of Transfer Unit', `${results.htu.toFixed(2)} m`],
                      ['Number of Transfer Units', `${results.maxNtu.toFixed(2)}`],
                      ['Total Heat Duty', `${results.totalHeatKw.toFixed(1)} kW`],
                      ['Sensible Heat', `${results.sensibleHeatKw.toFixed(1)} kW`],
                      ['Latent Heat', `${results.latentHeatKw.toFixed(1)} kW`],
                      ['Cooling Water Flow', `${results.coolingWaterLMin.toFixed(1)} L/min`],
                      ['NaOH Solution Flow', `${results.naohSolutionLHr.toFixed(1)} L/hr`],
                      ['Salt Produced', `${results.saltProducedKgHr.toFixed(2)} kg/hr`],
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

            {/* Acid Gas Details Table */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-4">Acid Gas Removal Details</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="py-2 px-3 text-slate-300">Component</th>
                      <th className="py-2 px-3 text-slate-300">Inlet (ppmv)</th>
                      <th className="py-2 px-3 text-slate-300">Outlet (ppmv)</th>
                      <th className="py-2 px-3 text-slate-300">Removed (kg/hr)</th>
                      <th className="py-2 px-3 text-slate-300">NTU</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.acidGasDetails.map((d) => (
                      <tr key={d.name} className="border-b border-slate-700/50">
                        <td className="py-2 px-3 text-white font-medium">{d.name}</td>
                        <td className="py-2 px-3 text-white">{d.inletPpmv.toFixed(0)}</td>
                        <td className="py-2 px-3 text-white">{d.outletPpmv.toFixed(1)}</td>
                        <td className="py-2 px-3 text-white">{d.removedKgHr.toFixed(3)}</td>
                        <td className="py-2 px-3 text-white">{d.ntu.toFixed(2)}</td>
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
              Enter operating conditions and click "Calculate Design" to see results.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
