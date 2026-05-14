import { describe, expect, it } from 'vitest'
import { solveODESystem, type ODEResultPoint } from './odeSolver'

function lastPoint(points: ODEResultPoint[]): ODEResultPoint {
  const point = points[points.length - 1]
  if (!point) {
    throw new Error('solver returned no points')
  }
  return point
}

describe('solveODESystem', () => {
  it('matches the exponential decay analytical solution', () => {
    const results = solveODESystem(
      { y: '-k*y' },
      { k: 0.1 },
      { y: 100 },
      0,
      10,
      101
    )

    const expectedFinalY = 100 * Math.exp(-1)

    expect(results).toHaveLength(101)
    expect(lastPoint(results).y).toBeCloseTo(expectedFinalY, 4)
  })

  it('preserves coupled variable order in harmonic oscillator integration', () => {
    const results = solveODESystem(
      { x: 'v', v: '-omega*omega*x' },
      { omega: 1 },
      { x: 1, v: 0 },
      0,
      Math.PI / 2,
      501
    )

    const finalPoint = lastPoint(results)

    expect(finalPoint.x).toBeCloseTo(0, 4)
    expect(finalPoint.v).toBeCloseTo(-1, 4)
  })

  it('rejects missing initial conditions before entering the RK4 loop', () => {
    expect(() =>
      solveODESystem({ y: '-k*y' }, { k: 0.1 }, {}, 0, 1, 10)
    ).toThrow("Initial value for 'y' must be finite")
  })
})
