export interface ODEPreset {
  derivatives: Record<string, string>
  parameters: Record<string, number>
  initial: Record<string, number>
  description: string
  tEnd: number
}

export type ODEResultPoint = Record<string, number>

const MIN_OUTPUT_POINTS = 2

export const ODE_PRESETS: Record<string, ODEPreset> = {
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

function assertFiniteNumber(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite`)
  }
}

function assertSolverInputs(
  derivatives: Record<string, string>,
  parameters: Record<string, number>,
  initialValues: Record<string, number>,
  tStart: number,
  tEnd: number,
  numPoints: number
): void {
  if (Object.keys(derivatives).length === 0) {
    throw new Error('At least one derivative is required')
  }
  if (!Number.isInteger(numPoints) || numPoints < MIN_OUTPUT_POINTS) {
    throw new Error(`numPoints must be an integer >= ${MIN_OUTPUT_POINTS}`)
  }

  assertFiniteNumber(tStart, 'tStart')
  assertFiniteNumber(tEnd, 'tEnd')

  for (const [name, value] of Object.entries(parameters)) {
    assertFiniteNumber(value, `Parameter '${name}'`)
  }
  for (const varName of Object.keys(derivatives)) {
    assertFiniteNumber(initialValues[varName], `Initial value for '${varName}'`)
  }
}

function compileExpression(
  expr: string,
  varNames: string[]
): (...args: number[]) => number {
  let processedExpr = expr

  processedExpr = processedExpr
    .replace(/\bsin\b/g, 'Math.sin')
    .replace(/\bcos\b/g, 'Math.cos')
    .replace(/\bexp\b/g, 'Math.exp')
    .replace(/\bsqrt\b/g, 'Math.sqrt')
    .replace(/\babs\b/g, 'Math.abs')
    .replace(/\bPI\b/g, 'Math.PI')

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

export function solveODESystem(
  derivatives: Record<string, string>,
  parameters: Record<string, number>,
  initialValues: Record<string, number>,
  tStart: number,
  tEnd: number,
  numPoints: number
): ODEResultPoint[] {
  assertSolverInputs(derivatives, parameters, initialValues, tStart, tEnd, numPoints)

  const varNames = Object.keys(derivatives)
  const paramNames = Object.keys(parameters)
  const allVarNames = [...varNames, 't', ...paramNames]

  const compiledDerivsArr: Array<(...args: number[]) => number> = new Array(varNames.length)
  for (let i = 0; i < varNames.length; i++) {
    compiledDerivsArr[i] = compileExpression(derivatives[varNames[i]], allVarNames)
  }

  const dt = (tEnd - tStart) / (numPoints - 1)
  const results: ODEResultPoint[] = []
  const paramValues = paramNames.map(p => parameters[p])
  const numVars = varNames.length
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
  const args = new Array<number>(numVars + 1 + paramValues.length)

  for (let i = 0; i < paramValues.length; i++) {
    args[numVars + 1 + i] = paramValues[i]
  }

  const computeDerivativesArr = (
    t: number,
    currentArr: number[],
    outDerivsArr: number[]
  ) => {
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
    const point: ODEResultPoint = { time: t }
    for (let j = 0; j < numVars; j++) {
      point[varNames[j]] = stateArr[j]
    }
    results.push(point)

    if (i < numPoints - 1) {
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
