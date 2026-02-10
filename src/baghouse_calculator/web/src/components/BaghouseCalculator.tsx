import { useState } from 'react'

// API base URL -- defaults to shared calc backend.
// Override via VITE_CALC_API_URL environment variable.
// See issue #609.
const CALC_API_BASE = import.meta.env.VITE_CALC_API_URL ?? 'http://localhost:8010'

const colors = {
  base: '#1e1e2e',
  mantle: '#181825',
  surface0: '#313244',
  surface1: '#45475a',
  text: '#cdd6f4',
  subtext0: '#a6adc8',
  blue: '#89b4fa',
  green: '#a6e3a1',
  red: '#f38ba8',
  yellow: '#f9e2af',
  peach: '#fab387',
  mauve: '#cba6f7',
  teal: '#94e2d5',
  lavender: '#b4befe',
}

interface Results {
  carbonRemoved: number
  ashRemoved: number
  totalSolids: number
  drumFillHours: number
  drumFillDays: number
  flowAcfm: number
  airToCloth: number
  outletTempC: number
  engine: string
}

// ---------------------------------------------------------------------------
// Client-side fallback (used when shared calc backend is unreachable)
// See issue #609 for context on why the Python backend is preferred.
// ---------------------------------------------------------------------------

function calculateFallback(
  gasFlow: number,
  inletTemp: number,
  pressure: number,
  carbonIn: number,
  ashIn: number,
  carbonEff: number,
  ashEff: number,
  heatLoss: number,
  drumVolume: number,
  solidDensity: number,
  bagArea: number,
): Results {
  const carbonRemoved = carbonIn * (carbonEff / 100)
  const ashRemoved = ashIn * (ashEff / 100)
  const totalSolids = carbonRemoved + ashRemoved

  const drumMassCapacity = solidDensity * drumVolume
  const drumFillHours = totalSolids > 0 ? drumMassCapacity / totalSolids : 0
  const drumFillDays = drumFillHours / 24

  // Simplified gas flow calculation
  const tempK = inletTemp + 273.15
  const pressurePa = pressure * 1000
  const molarFlow = gasFlow / 0.028 // Assume avg MW ~28 g/mol
  const volActual = (molarFlow * 8.314 * tempK) / pressurePa
  const flowAcfm = volActual * 2118.88 * 60

  const airToCloth = bagArea > 0 ? flowAcfm / bagArea : 0

  // Temperature drop from heat loss
  const cpMass = 1100 // J/(kg*K) approx
  const tempDrop = gasFlow > 0 && cpMass > 0 ? (heatLoss * 1000) / (gasFlow * cpMass) : 0
  const outletTempC = inletTemp - tempDrop

  return {
    carbonRemoved, ashRemoved, totalSolids, drumFillHours, drumFillDays,
    flowAcfm, airToCloth, outletTempC, engine: 'client-fallback',
  }
}

/**
 * Call the shared Python backend for baghouse calculation.
 * Falls back to client-side when API is unreachable.  See issue #609.
 */
async function fetchFromBackend(
  gasFlow: number,
  inletTemp: number,
  pressure: number,
  carbonIn: number,
  ashIn: number,
  carbonEff: number,
  ashEff: number,
  heatLoss: number,
  drumVolume: number,
  solidDensity: number,
  bagArea: number,
): Promise<Results> {
  const response = await fetch(`${CALC_API_BASE}/api/calc/baghouse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gas_flow_kg_s: gasFlow,
      inlet_temp_k: inletTemp + 273.15,
      pressure_pa: pressure * 1000,
      composition: { N2: 0.7, CO2: 0.15, H2O: 0.1, CO: 0.05 },
      solid_carbon_in_kg_hr: carbonIn,
      ash_in_kg_hr: ashIn,
      carbon_removal_efficiency: carbonEff / 100,
      ash_removal_efficiency: ashEff / 100,
      heat_loss_w: heatLoss * 1000,
      drum_volume_m3: drumVolume,
      solid_density_kg_m3: solidDensity,
      bag_area_ft2: bagArea,
    }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? `API error ${response.status}`)
  }

  const data = await response.json()

  return {
    carbonRemoved: data.carbon_removed_rate_kg_hr,
    ashRemoved: data.ash_removed_rate_kg_hr,
    totalSolids: data.total_solids_removed_rate_kg_hr,
    drumFillHours: data.drum_fill_time_hours,
    drumFillDays: data.drum_fill_time_days,
    flowAcfm: data.flow_acfm,
    airToCloth: data.air_to_cloth_ratio,
    outletTempC: data.outlet_temperature_c,
    engine: 'python-backend',
  }
}

function BaghouseCalculator() {
  const [gasFlow, setGasFlow] = useState(10)
  const [inletTemp, setInletTemp] = useState(200)
  const [pressure, setPressure] = useState(101.325)
  const [carbonIn, setCarbonIn] = useState(50)
  const [ashIn, setAshIn] = useState(20)
  const [carbonEff, setCarbonEff] = useState(99)
  const [ashEff, setAshEff] = useState(99)
  const [heatLoss, setHeatLoss] = useState(5)
  const [drumVolume, setDrumVolume] = useState(0.5)
  const [solidDensity, setSolidDensity] = useState(500)
  const [bagArea, setBagArea] = useState(1000)
  const [results, setResults] = useState<Results | null>(null)
  const [loading, setLoading] = useState(false)

  const calculate = async () => {
    setLoading(true)
    try {
      // Try the validated Python backend first; fall back to client-side.
      // See issue #609.
      let res: Results
      try {
        res = await fetchFromBackend(
          gasFlow, inletTemp, pressure, carbonIn, ashIn, carbonEff, ashEff,
          heatLoss, drumVolume, solidDensity, bagArea,
        )
      } catch {
        res = calculateFallback(
          gasFlow, inletTemp, pressure, carbonIn, ashIn, carbonEff, ashEff,
          heatLoss, drumVolume, solidDensity, bagArea,
        )
      }
      setResults(res)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8" style={{ color: colors.blue }}>
        Baghouse Calculator
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <Section title="Gas Stream">
            <InputField label="Gas Flow (kg/s)" value={gasFlow} onChange={setGasFlow} max={1000} />
            <InputField label="Inlet Temp (deg C)" value={inletTemp} onChange={setInletTemp} max={1000} />
            <InputField label="Pressure (kPa)" value={pressure} onChange={setPressure} min={50} max={500} step={0.1} />
          </Section>

          <Section title="Solids Input">
            <InputField label="Carbon Rate (kg/hr)" value={carbonIn} onChange={setCarbonIn} max={1000} />
            <InputField label="Ash Rate (kg/hr)" value={ashIn} onChange={setAshIn} max={500} />
          </Section>

          <Section title="Removal Efficiency">
            <InputField label="Carbon Removal (%)" value={carbonEff} onChange={setCarbonEff} max={100} />
            <InputField label="Ash Removal (%)" value={ashEff} onChange={setAshEff} max={100} />
          </Section>

          <Section title="Equipment Parameters">
            <InputField label="Heat Loss (kW)" value={heatLoss} onChange={setHeatLoss} max={100} />
            <InputField label="Drum Volume (m3)" value={drumVolume} onChange={setDrumVolume} max={10} step={0.1} />
            <InputField label="Solid Density (kg/m3)" value={solidDensity} onChange={setSolidDensity} min={100} max={2000} />
            <InputField label="Bag Area (ft2)" value={bagArea} onChange={setBagArea} min={100} max={10000} />
          </Section>

          <button
            onClick={calculate}
            disabled={loading}
            className="w-full py-3 rounded-lg font-bold text-lg"
            style={{
              backgroundColor: loading ? colors.surface1 : colors.blue,
              color: colors.base,
              cursor: loading ? 'wait' : 'pointer',
            }}
          >
            {loading ? 'Calculating...' : 'Calculate Baghouse Performance'}
          </button>
        </div>

        <div className="space-y-6">
          <h2 className="text-2xl font-bold" style={{ color: colors.green }}>Results</h2>
          {results && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard label="Carbon Removed" value={`${results.carbonRemoved.toFixed(1)} kg/hr`} color={colors.green} />
                <MetricCard label="Ash Removed" value={`${results.ashRemoved.toFixed(1)} kg/hr`} color={colors.yellow} />
                <MetricCard label="Total Solids" value={`${results.totalSolids.toFixed(1)} kg/hr`} color={colors.blue} />
                <MetricCard label="Drum Fill Time" value={`${results.drumFillHours.toFixed(1)} hrs`} color={colors.peach} />
                <MetricCard label="Drum Fill Days" value={`${results.drumFillDays.toFixed(2)} days`} color={colors.mauve} />
                <MetricCard label="Gas Flow" value={`${results.flowAcfm.toFixed(0)} ACFM`} color={colors.teal} />
                <MetricCard label="Air-to-Cloth" value={`${results.airToCloth.toFixed(2)} ft/min`} color={colors.lavender} />
                <MetricCard label="Outlet Temp" value={`${results.outletTempC.toFixed(1)} deg C`} color={colors.red} />
              </div>
              {/* Engine indicator -- See issue #609 */}
              <div className="text-right">
                <span className="text-xs" style={{ color: colors.subtext0 }}>
                  Engine: {results.engine}
                </span>
              </div>
            </>
          )}
          {!results && (
            <div className="rounded-lg p-8 text-center" style={{ backgroundColor: colors.surface0 }}>
              <p style={{ color: colors.subtext0 }}>Enter parameters and click Calculate</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg p-4" style={{ backgroundColor: colors.mantle, border: `1px solid ${colors.surface1}` }}>
      <h2 className="text-lg font-semibold mb-4" style={{ color: colors.lavender }}>{title}</h2>
      <div className="space-y-3">{children}</div>
    </div>
  )
}

function InputField({ label, value, onChange, min = 0, max = 100, step = 1 }: {
  label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <label className="text-sm" style={{ color: colors.text }}>{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min} max={max} step={step}
        className="w-28 px-3 py-2 rounded text-right"
        style={{ backgroundColor: colors.surface0, color: colors.text, border: `1px solid ${colors.surface1}` }}
      />
    </div>
  )
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-lg p-4" style={{ backgroundColor: colors.surface0 }}>
      <p className="text-sm mb-1" style={{ color: colors.subtext0 }}>{label}</p>
      <p className="text-lg font-bold" style={{ color }}>{value}</p>
    </div>
  )
}

export default BaghouseCalculator
