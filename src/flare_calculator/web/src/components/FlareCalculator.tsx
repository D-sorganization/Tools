import { useState, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

// Gas properties database
const GAS_PROPERTIES: Record<string, { mw: number; hv: number; cp: number }> = {
  H2: { mw: 2.016, hv: 119930, cp: 14.3 },
  CO: { mw: 28.01, hv: 10100, cp: 1.04 },
  CH4: { mw: 16.04, hv: 50010, cp: 2.22 },
  CO2: { mw: 44.01, hv: 0, cp: 0.84 },
  N2: { mw: 28.01, hv: 0, cp: 1.04 },
  H2O: { mw: 18.02, hv: 0, cp: 1.87 },
  H2S: { mw: 34.08, hv: 16500, cp: 1.05 },
}

const R_UNIVERSAL = 8.314
const HOUR_TO_SECOND = 3600.0

interface FlareDesign {
  height: number
  diameter: number
  exitVelocity: number
  heatRelease: number
  radiationIntensity: number
}

interface SafetyZones {
  lethal: number
  damage: number
  safe: number
  comfort: number
}

export function FlareCalculator() {
  const [flowRate, setFlowRate] = useState(1000)
  const [temperature, setTemperature] = useState(473)
  const [pressure, setPressure] = useState(1.5)
  const [composition, setComposition] = useState({
    H2: 35,
    CO: 30,
    CH4: 5,
    CO2: 15,
    N2: 5,
    H2O: 10,
    H2S: 0,
  })
  const [design, setDesign] = useState<FlareDesign | null>(null)
  const [zones, setZones] = useState<SafetyZones | null>(null)
  const [mixProps, setMixProps] = useState({ mw: 0, hv: 0, density: 0 })

  const calculate = useCallback(() => {
    const totalComp = Object.values(composition).reduce((a, b) => a + b, 0)
    if (totalComp === 0) return

    const fractions = Object.fromEntries(
      Object.entries(composition).map(([k, v]) => [k, v / totalComp])
    )

    // Calculate mixture properties
    let mixMw = 0
    let mixHv = 0
    for (const [gas, props] of Object.entries(GAS_PROPERTIES)) {
      mixMw += (fractions[gas] || 0) * props.mw
      mixHv += (fractions[gas] || 0) * props.hv
    }

    // Heat release
    const heatRelease = (flowRate * mixHv) / HOUR_TO_SECOND

    // Gas density
    const pressurePa = pressure * 100000
    const mixMwKg = mixMw / 1000
    const gasDensity = mixMwKg > 0 && temperature > 0
      ? pressurePa / ((R_UNIVERSAL / mixMwKg) * temperature)
      : 1.0

    // Flare diameter
    const targetVelocity = 170
    const massFlowKgS = flowRate / HOUR_TO_SECOND
    const area = gasDensity > 0 ? massFlowKgS / (gasDensity * targetVelocity) : 0
    const diameter = Math.sqrt((4 * area) / Math.PI)

    // Flare height
    const targetRadiation = 1.6
    const emissivity = 0.3
    let height = heatRelease > 0
      ? Math.sqrt((emissivity * heatRelease) / (4 * Math.PI * targetRadiation))
      : 0
    height = Math.max(height, 10)

    setDesign({
      height,
      diameter,
      exitVelocity: targetVelocity,
      heatRelease,
      radiationIntensity: targetRadiation,
    })

    // Safety zones
    const radiationLevels = { lethal: 37.5, damage: 12.5, safe: 1.6, comfort: 0.5 }
    const newZones: SafetyZones = { lethal: 0, damage: 0, safe: 0, comfort: 0 }
    for (const [zone, level] of Object.entries(radiationLevels)) {
      newZones[zone as keyof SafetyZones] = heatRelease > 0
        ? Math.sqrt((emissivity * heatRelease) / (4 * Math.PI * level))
        : 0
    }
    setZones(newZones)

    setMixProps({ mw: mixMw, hv: mixHv, density: gasDensity })
  }, [flowRate, temperature, pressure, composition])

  const zoneColors = {
    lethal: '#f38ba8',
    damage: '#fab387',
    safe: '#a6e3a1',
    comfort: '#89b4fa',
  }

  const zoneData = zones
    ? [
        { name: 'Lethal', distance: zones.lethal, color: zoneColors.lethal },
        { name: 'Damage', distance: zones.damage, color: zoneColors.damage },
        { name: 'Safe', distance: zones.safe, color: zoneColors.safe },
        { name: 'Comfort', distance: zones.comfort, color: zoneColors.comfort },
      ]
    : []

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold text-[#89b4fa] mb-6">Flare Calculator</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="space-y-4">
          <div className="bg-[#313244] rounded-lg p-4">
            <h2 className="text-lg font-semibold text-[#89b4fa] mb-3">Operating Conditions</h2>

            <div className="space-y-3">
              <div>
                <label className="block text-sm text-[#a6adc8] mb-1">Flow Rate (kg/hr)</label>
                <input
                  type="number"
                  value={flowRate}
                  onChange={(e) => setFlowRate(Number(e.target.value))}
                  className="w-full bg-[#1e1e2e] border border-[#45475a] rounded px-3 py-2"
                />
              </div>

              <div>
                <label className="block text-sm text-[#a6adc8] mb-1">Temperature (K)</label>
                <input
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full bg-[#1e1e2e] border border-[#45475a] rounded px-3 py-2"
                />
              </div>

              <div>
                <label className="block text-sm text-[#a6adc8] mb-1">Pressure (bar)</label>
                <input
                  type="number"
                  value={pressure}
                  onChange={(e) => setPressure(Number(e.target.value))}
                  className="w-full bg-[#1e1e2e] border border-[#45475a] rounded px-3 py-2"
                  step="0.1"
                />
              </div>
            </div>
          </div>

          <div className="bg-[#313244] rounded-lg p-4">
            <h2 className="text-lg font-semibold text-[#89b4fa] mb-3">Gas Composition (mol%)</h2>
            <div className="space-y-2">
              {Object.entries(composition).map(([gas, value]) => (
                <div key={gas} className="flex items-center gap-2">
                  <label className="w-12 text-sm">{gas}</label>
                  <input
                    type="number"
                    value={value}
                    onChange={(e) =>
                      setComposition((prev) => ({ ...prev, [gas]: Number(e.target.value) }))
                    }
                    className="flex-1 bg-[#1e1e2e] border border-[#45475a] rounded px-3 py-1 text-sm"
                    min="0"
                    max="100"
                  />
                  <span className="text-sm text-[#a6adc8]">%</span>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={calculate}
            className="w-full bg-[#89b4fa] text-[#1e1e2e] font-bold py-3 rounded-lg hover:bg-[#cba6f7] transition-colors"
          >
            Calculate Flare Design
          </button>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {design && (
            <>
              <div className="bg-[#313244] rounded-lg p-4">
                <h2 className="text-lg font-semibold text-[#89b4fa] mb-3">Flare Design</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Height</div>
                    <div className="text-xl font-bold">{design.height.toFixed(1)} m</div>
                  </div>
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Diameter</div>
                    <div className="text-xl font-bold">{design.diameter.toFixed(3)} m</div>
                  </div>
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Exit Velocity</div>
                    <div className="text-xl font-bold">{design.exitVelocity.toFixed(0)} m/s</div>
                  </div>
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Heat Release</div>
                    <div className="text-xl font-bold">{design.heatRelease.toFixed(0)} kW</div>
                  </div>
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Design Radiation</div>
                    <div className="text-xl font-bold">{design.radiationIntensity.toFixed(1)} kW/m²</div>
                  </div>
                </div>
              </div>

              <div className="bg-[#313244] rounded-lg p-4">
                <h2 className="text-lg font-semibold text-[#89b4fa] mb-3">Safety Zones</h2>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={zoneData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#45475a" />
                      <XAxis type="number" stroke="#cdd6f4" />
                      <YAxis type="category" dataKey="name" stroke="#cdd6f4" width={80} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e1e2e', border: '1px solid #45475a' }}
                        formatter={(value: number) => [`${value.toFixed(1)} m`, 'Distance']}
                      />
                      <Bar dataKey="distance">
                        {zoneData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-[#313244] rounded-lg p-4">
                <h2 className="text-lg font-semibold text-[#89b4fa] mb-3">Mixture Properties</h2>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Mol. Weight</div>
                    <div className="text-xl font-bold">{mixProps.mw.toFixed(2)} g/mol</div>
                  </div>
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Heating Value</div>
                    <div className="text-xl font-bold">{mixProps.hv.toFixed(0)} kJ/kg</div>
                  </div>
                  <div className="bg-[#1e1e2e] p-3 rounded">
                    <div className="text-sm text-[#a6adc8]">Gas Density</div>
                    <div className="text-xl font-bold">{mixProps.density.toFixed(3)} kg/m³</div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
