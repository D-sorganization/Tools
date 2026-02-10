/**
 * Thermal Profile Predictor - React Web Component
 *
 * Predicts temperature profiles in heated vessels using ODE integration.
 * Supports constant, linear ramp, and step power profiles.
 * Matches PyQt6 functionality.
 *
 * See issue #608.
 */

import { useState, useCallback, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

type PowerProfile = 'constant' | 'linear_ramp' | 'step'

const POWER_PROFILES: Record<PowerProfile, string> = {
  constant: 'Constant',
  linear_ramp: 'Linear Ramp',
  step: 'Step Function',
}

// Simple RK4 ODE solver for temperature profile
function solveTemperatureODE(
  initialTemp: number,
  ambientTemp: number,
  thermalMass: number,
  heatLossCoeff: number,
  powerFunc: (t: number) => number,
  tStart: number,
  tEnd: number,
  numPoints: number
): Array<{ time: number; temperature: number; power: number; heatLoss: number }> {
  const dt = (tEnd - tStart) / (numPoints - 1)
  const results: Array<{ time: number; temperature: number; power: number; heatLoss: number }> = []

  let T = initialTemp

  // dT/dt = (Q_in - Q_loss) / C_th
  // Q_loss = h * (T - T_ambient)
  const deriv = (t: number, temp: number): number => {
    const qIn = powerFunc(t)
    const qLoss = heatLossCoeff * (temp - ambientTemp)
    return (qIn - qLoss) / thermalMass
  }

  for (let i = 0; i < numPoints; i++) {
    const t = tStart + i * dt
    const power = powerFunc(t)
    const heatLoss = heatLossCoeff * (T - ambientTemp)

    results.push({
      time: t,
      temperature: T,
      power,
      heatLoss,
    })

    // RK4 step
    if (i < numPoints - 1) {
      const k1 = deriv(t, T)
      const k2 = deriv(t + dt / 2, T + (dt * k1) / 2)
      const k3 = deriv(t + dt / 2, T + (dt * k2) / 2)
      const k4 = deriv(t + dt, T + dt * k3)
      T += (dt * (k1 + 2 * k2 + 2 * k3 + k4)) / 6
    }
  }

  return results
}

interface ProfileResults {
  data: Array<{ time: number; temperature: number; power: number; heatLoss: number }>
  finalTemp: number
  maxTemp: number
  minTemp: number
  tempChange: number
  steadyStateTemp: number | null
}

export function ThermalProfilePredictor() {
  // Thermal parameters
  const [initialTemp, setInitialTemp] = useState(25)
  const [ambientTemp, setAmbientTemp] = useState(25)
  const [thermalMass, setThermalMass] = useState(50000)
  const [heatLossCoeff, setHeatLossCoeff] = useState(50)

  // Time parameters
  const [tStart, setTStart] = useState(0)
  const [tEnd, setTEnd] = useState(3600)
  const [numPoints, setNumPoints] = useState(100)

  // Power parameters
  const [powerProfile, setPowerProfile] = useState<PowerProfile>('constant')
  const [power, setPower] = useState(5000)
  const [rampRate, setRampRate] = useState(1)
  const [stepTime, setStepTime] = useState(1800)

  const [results, setResults] = useState<ProfileResults | null>(null)

  const calculate = useCallback(() => {
    // Create power function based on profile type
    let powerFunc: (t: number) => number
    switch (powerProfile) {
      case 'constant':
        powerFunc = () => power
        break
      case 'linear_ramp':
        powerFunc = (t: number) => power + rampRate * t
        break
      case 'step':
        powerFunc = (t: number) => (t < stepTime ? power : 0)
        break
    }

    const data = solveTemperatureODE(
      initialTemp, ambientTemp, thermalMass, heatLossCoeff,
      powerFunc, tStart, tEnd, numPoints
    )

    const temps = data.map((d) => d.temperature)
    const finalTemp = temps[temps.length - 1]
    const maxTemp = Math.max(...temps)
    const minTemp = Math.min(...temps)

    // Estimate steady state: P/h + T_amb (for constant power)
    const steadyStateTemp = powerProfile === 'constant' && heatLossCoeff > 0
      ? power / heatLossCoeff + ambientTemp
      : null

    setResults({
      data,
      finalTemp,
      maxTemp,
      minTemp,
      tempChange: finalTemp - initialTemp,
      steadyStateTemp,
    })
  }, [
    initialTemp, ambientTemp, thermalMass, heatLossCoeff,
    tStart, tEnd, numPoints, powerProfile, power, rampRate, stepTime,
  ])

  // Time constant
  const timeConstant = useMemo(
    () => heatLossCoeff > 0 ? thermalMass / heatLossCoeff : Infinity,
    [thermalMass, heatLossCoeff]
  )

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-2xl font-bold text-blue-400 mb-6">
        Thermal Profile Predictor
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="space-y-4">
          {/* Thermal Parameters */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">Thermal Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-slate-300 mb-1">Initial Temperature (C)</label>
                <input
                  type="number"
                  value={initialTemp}
                  onChange={(e) => setInitialTemp(Number(e.target.value))}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Ambient Temperature (C)</label>
                <input
                  type="number"
                  value={ambientTemp}
                  onChange={(e) => setAmbientTemp(Number(e.target.value))}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Thermal Mass (J/K)</label>
                <input
                  type="number"
                  value={thermalMass}
                  onChange={(e) => setThermalMass(Number(e.target.value))}
                  min={1}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Heat Loss Coeff (W/K)</label>
                <input
                  type="number"
                  value={heatLossCoeff}
                  onChange={(e) => setHeatLossCoeff(Number(e.target.value))}
                  min={0}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Time Parameters */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">Time Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-slate-300 mb-1">Start Time (s)</label>
                <input
                  type="number"
                  value={tStart}
                  onChange={(e) => setTStart(Number(e.target.value))}
                  min={0}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">End Time (s)</label>
                <input
                  type="number"
                  value={tEnd}
                  onChange={(e) => setTEnd(Number(e.target.value))}
                  min={1}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Data Points</label>
                <input
                  type="number"
                  value={numPoints}
                  onChange={(e) => setNumPoints(Number(e.target.value))}
                  min={10}
                  max={10000}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Power Parameters */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">Power Input</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-slate-300 mb-1">Power Profile</label>
                <select
                  value={powerProfile}
                  onChange={(e) => setPowerProfile(e.target.value as PowerProfile)}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                >
                  {Object.entries(POWER_PROFILES).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Power (W)</label>
                <input
                  type="number"
                  value={power}
                  onChange={(e) => setPower(Number(e.target.value))}
                  min={0}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
              </div>
              {powerProfile === 'linear_ramp' && (
                <div>
                  <label className="block text-sm text-slate-300 mb-1">Ramp Rate (W/s)</label>
                  <input
                    type="number"
                    value={rampRate}
                    onChange={(e) => setRampRate(Number(e.target.value))}
                    step={0.1}
                    className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              )}
              {powerProfile === 'step' && (
                <div>
                  <label className="block text-sm text-slate-300 mb-1">Step Time (s)</label>
                  <input
                    type="number"
                    value={stepTime}
                    onChange={(e) => setStepTime(Number(e.target.value))}
                    min={0}
                    className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              )}
              <div className="text-xs text-slate-400">
                Time constant: {timeConstant === Infinity ? '--' : `${timeConstant.toFixed(0)} s`}
              </div>
            </div>
          </div>

          <button
            onClick={calculate}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
          >
            Predict Temperature Profile
          </button>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {results ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-800 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Final Temp</p>
                  <p className="text-2xl font-bold text-white">{results.finalTemp.toFixed(1)} C</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Max Temp</p>
                  <p className="text-2xl font-bold text-white">{results.maxTemp.toFixed(1)} C</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Min Temp</p>
                  <p className="text-2xl font-bold text-white">{results.minTemp.toFixed(1)} C</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Temp Change</p>
                  <p className="text-2xl font-bold text-white">
                    {results.tempChange > 0 ? '+' : ''}{results.tempChange.toFixed(1)} C
                  </p>
                </div>
              </div>

              {results.steadyStateTemp !== null && (
                <div className="bg-slate-800 rounded-lg p-3 text-center">
                  <span className="text-slate-400 text-sm">
                    Theoretical steady-state temperature:{' '}
                  </span>
                  <span className="text-yellow-400 font-bold">
                    {results.steadyStateTemp.toFixed(1)} C
                  </span>
                </div>
              )}

              {/* Temperature Chart */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Temperature Profile</h2>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={results.data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                      <XAxis
                        dataKey="time"
                        stroke="#94a3b8"
                        label={{ value: 'Time (s)', position: 'bottom', fill: '#94a3b8', offset: -5 }}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        label={{ value: 'Temperature (C)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        labelStyle={{ color: '#f1f5f9' }}
                        formatter={(value: number, name: string) => [
                          `${value.toFixed(1)}`,
                          name === 'temperature' ? 'Temp (C)' : name,
                        ]}
                        labelFormatter={(label) => `t = ${Number(label).toFixed(0)} s`}
                      />
                      <Legend />
                      {results.steadyStateTemp !== null && (
                        <ReferenceLine
                          y={results.steadyStateTemp}
                          stroke="#f59e0b"
                          strokeDasharray="3 3"
                          label={{ value: 'Steady State', fill: '#f59e0b', fontSize: 10 }}
                        />
                      )}
                      <Line
                        type="monotone"
                        dataKey="temperature"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="Temperature"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Power & Heat Loss Chart */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Power & Heat Loss</h2>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={results.data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                      <XAxis
                        dataKey="time"
                        stroke="#94a3b8"
                        label={{ value: 'Time (s)', position: 'bottom', fill: '#94a3b8', offset: -5 }}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        label={{ value: 'Power (W)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        labelStyle={{ color: '#f1f5f9' }}
                        formatter={(value: number) => [`${value.toFixed(0)} W`]}
                        labelFormatter={(label) => `t = ${Number(label).toFixed(0)} s`}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="power"
                        stroke="#22c55e"
                        strokeWidth={2}
                        dot={false}
                        name="Power Input"
                      />
                      <Line
                        type="monotone"
                        dataKey="heatLoss"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        name="Heat Loss"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-slate-800 rounded-lg p-8 text-center">
              <p className="text-slate-400">
                Enter parameters and click "Predict Temperature Profile" to see results.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
