import { useState } from 'react'

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
  sky: '#89dceb',
  sapphire: '#74c7ec',
}

type CalculationMode = 'tp' | 'sat_t' | 'sat_p'

interface SteamProperties {
  temperature: number
  pressure: number
  density: number
  specificVolume: number
  enthalpy: number
  entropy: number
  internalEnergy: number
  cp: number
  cv: number
  speedOfSound: number
  thermalConductivity: number
  dynamicViscosity: number
  kinematicViscosity: number
  quality: number
  phase: string
  compressibilityFactor: number
  prandtlNumber: number
  specificHeatRatio: number
  engine: string
}

// API base URL -- defaults to localhost FastAPI backend.
// Override via VITE_STEAM_API_URL environment variable.
const API_BASE = import.meta.env.VITE_STEAM_API_URL ?? 'http://localhost:8002'

/**
 * Fetch steam properties from the validated Python backend.
 *
 * The Python engine uses CoolProp / Cantera / simplified correlations
 * (in that priority order) and returns physically accurate results.
 * See issue #605 for context on why the old hardcoded constants were wrong.
 */
async function fetchSteamProperties(
  mode: CalculationMode,
  temperature: number,
  pressure: number,
): Promise<SteamProperties> {
  const response = await fetch(`${API_BASE}/api/steam/calculate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, temperature, pressure, engine: 'auto' }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? `API error ${response.status}`)
  }

  return response.json()
}

// ---------------------------------------------------------------------------
// Fallback: simplified client-side calculation (used when API is unreachable)
// These correlations match the Python simplified backend so results are
// consistent, but they are NOT as accurate as CoolProp / Cantera.
// ---------------------------------------------------------------------------

function calculateSteamPropertiesFallback(
  _mode: CalculationMode,
  temperature: number,
  pressure: number
): SteamProperties {
  const R = 461.5 // J/kg-K for water vapor

  let phase: string
  let quality: number
  let density: number
  let cp: number
  let cv: number

  // Match Python simplified engine thresholds exactly
  if (temperature < 373.15 && pressure > 50000) {
    phase = 'liquid'
    quality = 0.0
    density = 1000.0
    cp = 4186.0
    cv = 4186.0
  } else {
    phase = 'vapor'
    quality = 1.0
    density = pressure / (R * temperature)
    // Python uses VAPOR_SPECIFIC_HEAT_CP = 1.9 kJ/kg-K -> 1900 J/kg-K
    cp = 1900.0
    // Python uses VAPOR_SPECIFIC_HEAT_CV = 1.4 kJ/kg-K -> 1400 J/kg-K
    cv = 1400.0
  }

  const specificVolume = 1.0 / density
  const enthalpy = phase === 'liquid'
    ? 4186.0 * (temperature - 273.15)
    : (2500 + 1.9 * (temperature - 273.15)) * 1000
  const entropy = phase === 'liquid'
    ? 4186.0 * Math.log(temperature / 273.15)
    : 8000 + 2000 * Math.log(temperature / 373.15)
  const internalEnergy = enthalpy - pressure * specificVolume
  const speedOfSound = phase === 'liquid' ? 1500 : Math.sqrt(1.3 * R * temperature)
  const thermalConductivity = phase === 'liquid' ? 0.6 : 0.025
  const dynamicViscosity = phase === 'liquid' ? 2.8e-4 : 1.2e-5
  const kinematicViscosity = dynamicViscosity / density
  const compressibilityFactor = pressure * specificVolume / (R * temperature)
  const specificHeatRatio = cp / cv
  const prandtlNumber = cp * dynamicViscosity / thermalConductivity

  return {
    temperature,
    pressure,
    density,
    specificVolume,
    enthalpy,
    entropy,
    internalEnergy,
    cp,
    cv,
    speedOfSound,
    thermalConductivity,
    dynamicViscosity,
    kinematicViscosity,
    quality,
    phase,
    compressibilityFactor,
    prandtlNumber,
    specificHeatRatio,
    engine: 'simplified-fallback',
  }
}

// Antoine equation helpers (kept for saturation mode fallbacks)
function getSaturationPressure(tempK: number): number {
  const A = 8.07131
  const B = 1730.63
  const C = 39.724
  const logP = A - B / (tempK - C)
  const pMmHg = Math.pow(10, logP)
  return pMmHg * 133.322
}

function getSaturationTemperature(pressurePa: number): number {
  const A = 8.07131
  const B = 1730.63
  const C = 39.724
  const pMmHg = pressurePa / 133.322
  const logP = Math.log10(pMmHg)
  return B / (A - logP) + C
}

function SteamEngineCalculator() {
  const [mode, setMode] = useState<CalculationMode>('tp')
  const [temperature, setTemperature] = useState(373.15)
  const [pressure, setPressure] = useState(101325)
  const [tempUnit, setTempUnit] = useState<'K' | 'C'>('K')
  const [pressureUnit, setPressureUnit] = useState<'Pa' | 'kPa' | 'bar' | 'MPa'>('Pa')
  const [results, setResults] = useState<SteamProperties | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const getTemperatureK = (): number => {
    return tempUnit === 'C' ? temperature + 273.15 : temperature
  }

  const getPressurePa = (): number => {
    switch (pressureUnit) {
      case 'kPa': return pressure * 1000
      case 'bar': return pressure * 100000
      case 'MPa': return pressure * 1000000
      default: return pressure
    }
  }

  const calculate = async () => {
    setError(null)
    setLoading(true)

    try {
      let tempK = getTemperatureK()
      let pressurePa = getPressurePa()

      // Validate inputs
      if (mode !== 'sat_p' && (tempK < 273.16 || tempK > 647.15)) {
        setError('Temperature must be between 273.16 K and 647.15 K')
        setLoading(false)
        return
      }

      if (mode !== 'sat_t' && pressurePa <= 0) {
        setError('Pressure must be positive')
        setLoading(false)
        return
      }

      // Handle saturation modes for fallback path
      if (mode === 'sat_t') {
        pressurePa = getSaturationPressure(tempK)
      } else if (mode === 'sat_p') {
        tempK = getSaturationTemperature(pressurePa)
      }

      // Try the validated Python backend first; fall back to client-side
      // simplified calculation if the API is unreachable.  See issue #605.
      let props: SteamProperties
      try {
        props = await fetchSteamProperties(mode, tempK, pressurePa)
      } catch {
        // API unreachable -- use client-side fallback
        props = calculateSteamPropertiesFallback(mode, tempK, pressurePa)
      }
      setResults(props)
    } catch (e) {
      setError(`Calculation error: ${e}`)
    } finally {
      setLoading(false)
    }
  }

  const getPhaseColor = (phase: string): string => {
    switch (phase.toLowerCase()) {
      case 'liquid': return colors.blue
      case 'vapor': return colors.peach
      case 'two-phase': return colors.yellow
      case 'supercritical': return colors.red
      default: return colors.text
    }
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-2" style={{ color: colors.blue }}>
        Steam Engine Calculator
      </h1>
      <p className="text-sm mb-8" style={{ color: colors.subtext0 }}>
        Thermodynamic property calculator for water/steam
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <div className="space-y-6">
          {/* Calculation Mode */}
          <Section title="Calculation Mode">
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as CalculationMode)}
              className="w-full px-4 py-3 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa]"
              style={{ backgroundColor: colors.surface0, color: colors.text, border: `1px solid ${colors.surface1}` }}
            >
              <option value="tp">Temperature & Pressure</option>
              <option value="sat_t">Saturated (from Temperature)</option>
              <option value="sat_p">Saturated (from Pressure)</option>
            </select>
          </Section>

          {/* Input Parameters */}
          <Section title="Input Parameters">
            {/* Temperature */}
            <div className="flex items-center gap-4 mb-4">
              <label className="w-24 text-sm" style={{ color: colors.text }}>Temperature</label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={mode === 'sat_p'}
                className="flex-1 px-3 py-2 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa]"
                style={{
                  backgroundColor: mode === 'sat_p' ? colors.surface1 : colors.surface0,
                  color: colors.text,
                  border: `1px solid ${colors.surface1}`
                }}
              />
              <select
                value={tempUnit}
                onChange={(e) => setTempUnit(e.target.value as 'K' | 'C')}
                className="w-20 px-2 py-2 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa]"
                style={{ backgroundColor: colors.surface0, color: colors.text, border: `1px solid ${colors.surface1}` }}
              >
                <option value="K">K</option>
                <option value="C">C</option>
              </select>
            </div>

            {/* Pressure */}
            <div className="flex items-center gap-4">
              <label className="w-24 text-sm" style={{ color: colors.text }}>Pressure</label>
              <input
                type="number"
                value={pressure}
                onChange={(e) => setPressure(Number(e.target.value))}
                disabled={mode === 'sat_t'}
                className="flex-1 px-3 py-2 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa]"
                style={{
                  backgroundColor: mode === 'sat_t' ? colors.surface1 : colors.surface0,
                  color: colors.text,
                  border: `1px solid ${colors.surface1}`
                }}
              />
              <select
                value={pressureUnit}
                onChange={(e) => setPressureUnit(e.target.value as 'Pa' | 'kPa' | 'bar' | 'MPa')}
                className="w-20 px-2 py-2 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa]"
                style={{ backgroundColor: colors.surface0, color: colors.text, border: `1px solid ${colors.surface1}` }}
              >
                <option value="Pa">Pa</option>
                <option value="kPa">kPa</option>
                <option value="bar">bar</option>
                <option value="MPa">MPa</option>
              </select>
            </div>
          </Section>

          {/* Calculate Button */}
          <button
            onClick={calculate}
            disabled={loading}
            className="w-full py-4 rounded-lg font-bold text-lg transition-colors hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa] focus-visible:ring-offset-2 focus-visible:ring-offset-[#1e1e2e]"
            style={{
              backgroundColor: loading ? colors.surface1 : colors.blue,
              color: colors.base,
              cursor: loading ? 'wait' : 'pointer',
            }}
          >
            {loading ? 'Calculating...' : 'Calculate Properties'}
          </button>

          {error && (
            <div className="p-4 rounded-lg" style={{ backgroundColor: colors.surface0, borderLeft: `4px solid ${colors.red}` }}>
              <p style={{ color: colors.red }}>{error}</p>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold" style={{ color: colors.green }}>Results</h2>

          {results ? (
            <>
              {/* Phase State */}
              <Section title="Phase State">
                <div className="flex justify-between items-center">
                  <span
                    className="text-2xl font-bold"
                    style={{ color: getPhaseColor(results.phase) }}
                  >
                    {results.phase.toUpperCase()}
                  </span>
                  <span style={{ color: colors.subtext0 }}>
                    Quality: {results.quality.toFixed(4)}
                  </span>
                </div>
                {/* Show which engine produced the results */}
                <div className="mt-2">
                  <span className="text-xs" style={{ color: colors.subtext0 }}>
                    Engine: {results.engine}
                  </span>
                </div>
              </Section>

              {/* Thermodynamic Properties */}
              <Section title="Thermodynamic Properties">
                <div className="grid grid-cols-3 gap-3">
                  <MetricCard
                    label="Temperature"
                    value={`${results.temperature.toFixed(2)} K`}
                    subvalue={`${(results.temperature - 273.15).toFixed(2)} C`}
                    color={colors.blue}
                  />
                  <MetricCard
                    label="Pressure"
                    value={`${(results.pressure / 1000).toFixed(2)} kPa`}
                    subvalue={`${(results.pressure / 1e5).toFixed(4)} bar`}
                    color={colors.blue}
                  />
                  <MetricCard
                    label="Density"
                    value={`${results.density.toFixed(4)} kg/m3`}
                    color={colors.peach}
                  />
                  <MetricCard
                    label="Specific Volume"
                    value={`${results.specificVolume.toFixed(6)} m3/kg`}
                    color={colors.peach}
                  />
                  <MetricCard
                    label="Enthalpy"
                    value={`${(results.enthalpy / 1000).toFixed(2)} kJ/kg`}
                    color={colors.green}
                  />
                  <MetricCard
                    label="Entropy"
                    value={`${(results.entropy / 1000).toFixed(4)} kJ/kg-K`}
                    color={colors.green}
                  />
                  <MetricCard
                    label="Internal Energy"
                    value={`${(results.internalEnergy / 1000).toFixed(2)} kJ/kg`}
                    color={colors.teal}
                  />
                  <MetricCard
                    label="Cp"
                    value={`${results.cp.toFixed(2)} J/kg-K`}
                    color={colors.yellow}
                  />
                  <MetricCard
                    label="Cv"
                    value={`${results.cv.toFixed(2)} J/kg-K`}
                    color={colors.yellow}
                  />
                </div>
              </Section>

              {/* Transport Properties */}
              <Section title="Transport Properties">
                <div className="grid grid-cols-2 gap-3">
                  <MetricCard
                    label="Speed of Sound"
                    value={`${results.speedOfSound.toFixed(2)} m/s`}
                    color={colors.mauve}
                  />
                  <MetricCard
                    label="Thermal Conductivity"
                    value={`${results.thermalConductivity.toFixed(6)} W/m-K`}
                    color={colors.lavender}
                  />
                  <MetricCard
                    label="Dynamic Viscosity"
                    value={`${results.dynamicViscosity.toExponential(2)} Pa-s`}
                    color={colors.sky}
                  />
                  <MetricCard
                    label="Kinematic Viscosity"
                    value={`${results.kinematicViscosity.toExponential(2)} m2/s`}
                    color={colors.sapphire}
                  />
                </div>
              </Section>

              {/* Derived Properties */}
              <Section title="Derived Properties">
                <div className="grid grid-cols-3 gap-3">
                  <MetricCard
                    label="Compressibility (Z)"
                    value={results.compressibilityFactor.toFixed(4)}
                    color={colors.teal}
                  />
                  <MetricCard
                    label="Prandtl Number"
                    value={results.prandtlNumber.toFixed(4)}
                    color={colors.peach}
                  />
                  <MetricCard
                    label="Cp/Cv Ratio (k)"
                    value={results.specificHeatRatio.toFixed(4)}
                    color={colors.yellow}
                  />
                </div>
              </Section>
            </>
          ) : (
            <div className="rounded-lg p-8 text-center" style={{ backgroundColor: colors.surface0 }}>
              <p style={{ color: colors.subtext0 }}>
                Enter parameters and click Calculate to see steam properties
              </p>
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

function MetricCard({
  label,
  value,
  subvalue,
  color
}: {
  label: string
  value: string
  subvalue?: string
  color: string
}) {
  return (
    <div className="rounded-lg p-3" style={{ backgroundColor: colors.surface0 }}>
      <p className="text-xs mb-1" style={{ color: colors.subtext0 }}>{label}</p>
      <p className="text-sm font-bold" style={{ color }}>{value}</p>
      {subvalue && (
        <p className="text-xs mt-1" style={{ color: colors.subtext0 }}>{subvalue}</p>
      )}
    </div>
  )
}

export default SteamEngineCalculator
