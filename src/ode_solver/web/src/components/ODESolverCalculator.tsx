/**
 * ODE Solver Calculator - React Web Component
 *
 * Solves systems of ordinary differential equations with preset examples.
 * Supports custom ODE definitions with parameters and initial conditions.
 * Uses RK4 integration and visualizes solutions with recharts.
 * Matches PyQt6 functionality.
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
  Legend,
  ResponsiveContainer,
} from 'recharts'

// Line colors for up to 6 variables
const LINE_COLORS = ['#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899']

// Preset ODE examples matching PyQt6 version
interface ODEPreset {
  derivatives: Record<string, string>
  parameters: Record<string, number>
  initial: Record<string, number>
  description: string
  tEnd: number
}

const ODE_PRESETS: Record<string, ODEPreset> = {
  'Exponential Decay': {
    derivatives: { y: '-k*y' },
    parameters: { k: 0.1 },
    initial: { y: 100 },
    description: 'dy/dt = -k*y (exponential decay)',
    tEnd: 50,
  },
  'Heating/Cooling': {
    derivatives: { T: 'k*(T_env - T)' },
    parameters: { k: 0.3, T_env: 350 },
    initial: { T: 300 },
    description: 'dT/dt = k*(T_env - T) (Newton\'s law of cooling)',
    tEnd: 20,
  },
  'Harmonic Oscillator': {
    derivatives: { x: 'v', v: '-omega*omega*x' },
    parameters: { omega: 1.0 },
    initial: { x: 1, v: 0 },
    description: 'dx/dt=v, dv/dt=-omega^2*x (simple harmonic motion)',
    tEnd: 30,
  },
  'Damped Oscillator': {
    derivatives: { x: 'v', v: '-2*zeta*omega*v - omega*omega*x' },
    parameters: { omega: 1.0, zeta: 0.1 },
    initial: { x: 1, v: 0 },
    description: 'Damped harmonic oscillator with damping ratio zeta',
    tEnd: 50,
  },
  'Lotka-Volterra': {
    derivatives: { x: 'a*x - b*x*y', y: '-c*y + d*x*y' },
    parameters: { a: 1.0, b: 0.1, c: 1.5, d: 0.075 },
    initial: { x: 10, y: 5 },
    description: 'Predator-prey model (x=prey, y=predators)',
    tEnd: 30,
  },
}

// ⚡ Bolt Optimization: Pre-compile expression to a native JS function.
// Eliminates repetitive "new Function()" and string processing during numerical integration loop.
function compileExpression(
  expr: string,
  varNames: string[]
): (...args: number[]) => number {
  let processedExpr = expr

  // Add Math functions
  processedExpr = processedExpr
    .replace(/\bsin\b/g, 'Math.sin')
    .replace(/\bcos\b/g, 'Math.cos')
    .replace(/\bexp\b/g, 'Math.exp')
    .replace(/\bsqrt\b/g, 'Math.sqrt')
    .replace(/\babs\b/g, 'Math.abs')
    .replace(/\bPI\b/g, 'Math.PI')

  // Note: JavaScript natively supports the ** operator for exponentiation,
  // so we don't need to replace it like we did previously.

  try {
    // eslint-disable-next-line no-new-func
    const compiled = new Function(...varNames, `"use strict"; return (${processedExpr})`) as (
      ...args: number[]
    ) => unknown
    return (...args: number[]) => {
      try {
        const result = compiled(...args)
        return typeof result === 'number' && isFinite(result) ? result : 0
      } catch {
        return 0
      }
    }
  } catch {
    return () => 0
  }
}

// RK4 solver for system of ODEs
function solveODESystem(
  derivatives: Record<string, string>,
  parameters: Record<string, number>,
  initialValues: Record<string, number>,
  tStart: number,
  tEnd: number,
  numPoints: number
): Array<Record<string, number>> {
  const varNames = Object.keys(derivatives)
  const paramNames = Object.keys(parameters)
  const allVarNames = [...varNames, 't', ...paramNames]

  const compiledDerivsArr: Array<(...args: number[]) => number> = new Array(varNames.length)
  for (let i = 0; i < varNames.length; i++) {
    compiledDerivsArr[i] = compileExpression(derivatives[varNames[i]], allVarNames)
  }

  const dt = (tEnd - tStart) / (numPoints - 1)
  const results: Array<Record<string, number>> = []

  const paramValues = paramNames.map(p => parameters[p])
  const numVars = varNames.length

  // ⚡ Bolt Optimization: Replace object property access with array index access in the RK4 loop.
  // This eliminates object hashing overhead and drastically improves execution speed for hot loops.
  const stateArr = new Array<number>(numVars)
  for (let i = 0; i < numVars; i++) {
    stateArr[i] = initialValues[varNames[i]]
  }

  const k1 = new Array<number>(numVars)
  const k2 = new Array<number>(numVars)
  const k3 = new Array<number>(numVars)
  const k4 = new Array<number>(numVars)
  const state2 = new Array<number>(numVars)
  const state3 = new Array<number>(numVars)
  const state4 = new Array<number>(numVars)

  // Pre-allocate arguments array for the compiled derivative functions
  const args = new Array<number>(numVars + 1 + paramValues.length)

  // Set constant parameters at the end of the args array
  for (let i = 0; i < paramValues.length; i++) {
    args[numVars + 1 + i] = paramValues[i]
  }

  const computeDerivativesArr = (
    t: number,
    currentArr: number[],
    outDerivsArr: number[]
  ) => {
    // ⚡ Bolt: Re-use pre-allocated args array
    for (let i = 0; i < numVars; i++) {
      args[i] = currentArr[i]
    }
    args[numVars] = t

    for (let i = 0; i < numVars; i++) {
      outDerivsArr[i] = compiledDerivsArr[i](...args)
    }
  }

  for (let i = 0; i < numPoints; i++) {
    const t = tStart + i * dt
    const point: Record<string, number> = { time: t }
    for (let j = 0; j < numVars; j++) {
      point[varNames[j]] = stateArr[j]
    }
    results.push(point)

    if (i < numPoints - 1) {
      // RK4
      computeDerivativesArr(t, stateArr, k1)

      for (let j = 0; j < numVars; j++) {
        state2[j] = stateArr[j] + (dt / 2) * k1[j]
      }
      computeDerivativesArr(t + dt / 2, state2, k2)

      for (let j = 0; j < numVars; j++) {
        state3[j] = stateArr[j] + (dt / 2) * k2[j]
      }
      computeDerivativesArr(t + dt / 2, state3, k3)

      for (let j = 0; j < numVars; j++) {
        state4[j] = stateArr[j] + dt * k3[j]
      }
      computeDerivativesArr(t + dt, state4, k4)

      for (let j = 0; j < numVars; j++) {
        stateArr[j] += (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
      }
    }
  }

  return results
}

export function ODESolverCalculator() {
  const [preset, setPreset] = useState('Harmonic Oscillator')
  const [derivativesText, setDerivativesText] = useState('x: v\nv: -omega*omega*x')
  const [parametersText, setParametersText] = useState('omega: 1.0')
  const [initialText, setInitialText] = useState('x: 1\nv: 0')
  const [tStart, setTStart] = useState(0)
  const [tEnd, setTEnd] = useState(30)
  const [numPoints, setNumPoints] = useState(200)
  const [results, setResults] = useState<Array<Record<string, number>> | null>(null)
  const [error, setError] = useState<string | null>(null)

  const parseKeyValue = useCallback((text: string): Record<string, string> => {
    const result: Record<string, string> = {}
    for (const line of text.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.includes(':')) continue
      const [key, ...rest] = trimmed.split(':')
      result[key.trim()] = rest.join(':').trim()
    }
    return result
  }, [])

  const handlePresetChange = useCallback((presetName: string) => {
    setPreset(presetName)
    if (presetName === 'Custom') return

    const p = ODE_PRESETS[presetName]
    if (!p) return

    setDerivativesText(Object.entries(p.derivatives).map(([k, v]) => `${k}: ${v}`).join('\n'))
    setParametersText(Object.entries(p.parameters).map(([k, v]) => `${k}: ${v}`).join('\n'))
    setInitialText(Object.entries(p.initial).map(([k, v]) => `${k}: ${v}`).join('\n'))
    setTEnd(p.tEnd)
  }, [])

  const solve = useCallback(() => {
    setError(null)
    try {
      const derivatives = parseKeyValue(derivativesText)
      const paramStr = parseKeyValue(parametersText)
      const initStr = parseKeyValue(initialText)

      if (Object.keys(derivatives).length === 0) {
        setError('No derivatives defined')
        return
      }

      const parameters: Record<string, number> = {}
      for (const [k, v] of Object.entries(paramStr)) {
        parameters[k] = parseFloat(v)
        if (isNaN(parameters[k])) {
          setError(`Invalid parameter value for '${k}'`)
          return
        }
      }

      const initialValues: Record<string, number> = {}
      for (const varName of Object.keys(derivatives)) {
        if (!(varName in initStr)) {
          setError(`Missing initial condition for '${varName}'`)
          return
        }
        initialValues[varName] = parseFloat(initStr[varName])
        if (isNaN(initialValues[varName])) {
          setError(`Invalid initial value for '${varName}'`)
          return
        }
      }

      const data = solveODESystem(
        derivatives, parameters, initialValues, tStart, tEnd, numPoints
      )
      setResults(data)
    } catch (e) {
      setError(`Error: ${e instanceof Error ? e.message : String(e)}`)
    }
  }, [derivativesText, parametersText, initialText, tStart, tEnd, numPoints, parseKeyValue])

  const varNames = results && results.length > 0
    ? Object.keys(results[0]).filter((k) => k !== 'time')
    : []

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-2xl font-bold text-blue-400 mb-6">
        ODE Solver
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="space-y-4">
          {/* Preset Selector */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">Preset Examples</h2>
            <select
              value={preset}
              onChange={(e) => handlePresetChange(e.target.value)}
              className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
            >
              <option value="Custom">Custom</option>
              {Object.keys(ODE_PRESETS).map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            {preset !== 'Custom' && ODE_PRESETS[preset] && (
              <p className="text-xs text-slate-400 mt-2 italic">
                {ODE_PRESETS[preset].description}
              </p>
            )}
          </div>

          {/* ODE Definition */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">ODE System</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-slate-300 mb-1">
                  Derivatives (var: expression)
                </label>
                <textarea
                  value={derivativesText}
                  onChange={(e) => setDerivativesText(e.target.value)}
                  rows={3}
                  placeholder="y: -k*y"
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none font-mono text-sm"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">
                  Parameters (name: value)
                </label>
                <textarea
                  value={parametersText}
                  onChange={(e) => setParametersText(e.target.value)}
                  rows={3}
                  placeholder="k: 0.1"
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none font-mono text-sm"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">
                  Initial Conditions (var: value)
                </label>
                <textarea
                  value={initialText}
                  onChange={(e) => setInitialText(e.target.value)}
                  rows={3}
                  placeholder="y: 100"
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none font-mono text-sm"
                />
              </div>
            </div>
          </div>

          {/* Time Parameters */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">Time Parameters</h2>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm text-slate-300 mb-1">Start Time</label>
                  <input
                    type="number"
                    value={tStart}
                    onChange={(e) => setTStart(Number(e.target.value))}
                    className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-300 mb-1">End Time</label>
                  <input
                    type="number"
                    value={tEnd}
                    onChange={(e) => setTEnd(Number(e.target.value))}
                    className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm text-slate-300 mb-1">Output Points</label>
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

          <button
            onClick={solve}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
          >
            Solve ODE System
          </button>

          {error && (
            <div className="bg-red-900/30 border border-red-500 rounded-lg p-3 text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {results && results.length > 0 ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {varNames.map((varName, idx) => {
                  // ⚡ Bolt Optimization: Replace map and spread min/max with single-pass loop
                  // Prevents "Maximum call stack size exceeded" on large datasets
                  let min = Infinity;
                  let max = -Infinity;
                  for (let i = 0; i < results.length; i++) {
                    const val = results[i][varName];
                    if (val < min) min = val;
                    if (val > max) max = val;
                  }
                  const final = results[results.length - 1]?.[varName];
                  return (
                    <div key={varName} className="bg-slate-800 rounded-lg p-4">
                      <p className="text-slate-400 text-sm" style={{ color: LINE_COLORS[idx % LINE_COLORS.length] }}>
                        {varName}
                      </p>
                      <p className="text-xl font-bold text-white">{final.toFixed(4)}</p>
                      <p className="text-xs text-slate-500">
                        [{min.toFixed(2)} .. {max.toFixed(2)}]
                      </p>
                    </div>
                  )
                })}
              </div>

              {/* Solution Chart */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Solution</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={results}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                      <XAxis
                        dataKey="time"
                        stroke="#94a3b8"
                        label={{ value: 'Time', position: 'bottom', fill: '#94a3b8', offset: -5 }}
                      />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        labelStyle={{ color: '#f1f5f9' }}
                        formatter={(value: number, name: string) => [value.toFixed(4), name]}
                        labelFormatter={(label) => `t = ${Number(label).toFixed(3)}`}
                      />
                      <Legend />
                      {varNames.map((varName, idx) => (
                        <Line
                          key={varName}
                          type="monotone"
                          dataKey={varName}
                          stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          name={varName}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Phase Portrait (for 2-variable systems) */}
              {varNames.length === 2 && (
                <div className="bg-slate-800 rounded-lg p-4">
                  <h2 className="text-lg font-semibold text-white mb-4">
                    Phase Portrait ({varNames[0]} vs {varNames[1]})
                  </h2>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={results}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                        <XAxis
                          dataKey={varNames[0]}
                          type="number"
                          stroke="#94a3b8"
                          label={{ value: varNames[0], position: 'bottom', fill: '#94a3b8', offset: -5 }}
                        />
                        <YAxis
                          dataKey={varNames[1]}
                          type="number"
                          stroke="#94a3b8"
                          label={{ value: varNames[1], angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                          labelStyle={{ color: '#f1f5f9' }}
                        />
                        <Line
                          type="monotone"
                          dataKey={varNames[1]}
                          stroke="#8b5cf6"
                          strokeWidth={2}
                          dot={false}
                          name="Trajectory"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Data Table */}
              <div className="bg-slate-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">Sample Data Points</h2>
                <div className="overflow-x-auto max-h-64">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="py-2 px-3 text-slate-300">Time</th>
                        {varNames.map((v) => (
                          <th key={v} className="py-2 px-3 text-slate-300">{v}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(() => {
                        // ⚡ Bolt Optimization: Replace .filter().map() chains with a single-pass loop.
                        // This prevents creating intermediate arrays and limits iterations to O(N/step).
                        const step = Math.max(1, Math.floor(results.length / 15));
                        const rows = [];
                        for (let i = 0; i < results.length; i += step) {
                          const row = results[i];
                          rows.push(
                            <tr key={i} className="border-b border-slate-700/50">
                              <td className="py-1 px-3 text-white">{row.time.toFixed(3)}</td>
                              {varNames.map((v) => (
                                <td key={v} className="py-1 px-3 text-white">{row[v].toFixed(6)}</td>
                              ))}
                            </tr>
                          );
                        }
                        return rows;
                      })()}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-slate-800 rounded-lg p-8 text-center">
              <p className="text-slate-400">
                Select a preset or define a custom ODE system and click "Solve ODE System" to see results.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
