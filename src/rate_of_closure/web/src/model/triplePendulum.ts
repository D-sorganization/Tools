export type TripleState = [number, number, number, number, number, number];

interface TripleParameters {
  mass: [number, number, number];
  length: [number, number, number];
  center: [number, number, number];
  inertia: [number, number, number];
  damping: [number, number, number];
}

export function golfTripleParameters(): TripleParameters {
  const shaftMass = 0.15;
  const headMass = 0.2;
  const clubMass = shaftMass + headMass;
  const shaftCenter = 0.43;
  const clubCenter = (shaftCenter * shaftMass + headMass) / clubMass;
  const shaftInertia = (shaftMass / 12) * 1.0 ** 2;
  const parallel =
    shaftMass * (shaftCenter - clubCenter) ** 2 +
    headMass * (1.0 - clubCenter) ** 2;
  return {
    mass: [4.5, 3.0, clubMass],
    length: [0.4, 0.35, 1.0],
    center: [0.4 * 0.45, 0.35 * 0.45, clubCenter],
    inertia: [(4.5 / 12) * 0.4 ** 2, (3.0 / 12) * 0.35 ** 2, shaftInertia + parallel],
    damping: [0.4, 0.3, 0.25],
  };
}

function solve3(matrix: number[][], rhs: number[]): [number, number, number] {
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  for (let pivot = 0; pivot < 3; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[best][pivot])) best = row;
    }
    [augmented[pivot], augmented[best]] = [augmented[best], augmented[pivot]];
    const divisor = augmented[pivot][pivot];
    if (Math.abs(divisor) < 1e-12) throw new Error("singular triple-pendulum mass matrix");
    for (let column = pivot; column < 4; column += 1) augmented[pivot][column] /= divisor;
    for (let row = 0; row < 3; row += 1) {
      if (row === pivot) continue;
      const factor = augmented[row][pivot];
      for (let column = pivot; column < 4; column += 1) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }
  return [augmented[0][3], augmented[1][3], augmented[2][3]];
}

function derivatives(
  p: TripleParameters,
  state: TripleState,
  gravity: [number, number],
): TripleState {
  const phi = state.slice(0, 3);
  const rate = state.slice(3, 6);
  const coefficients = Array.from({ length: 3 }, (_, body) =>
    Array.from({ length: 3 }, (_, coordinate) =>
      coordinate < body ? p.length[coordinate] : coordinate === body ? p.center[body] : 0,
    ),
  );
  const beta = Array.from({ length: 3 }, (_, row) =>
    Array.from({ length: 3 }, (_, column) =>
      p.mass.reduce(
        (sum, mass, body) => sum + mass * coefficients[body][row] * coefficients[body][column],
        0,
      ),
    ),
  );
  const massMatrix = beta.map((row, i) =>
    row.map((value, j) => value * Math.cos(phi[i] - phi[j]) + (i === j ? p.inertia[i] : 0)),
  );
  const coriolis = beta.map((row, i) =>
    row.reduce((sum, value, j) => sum + value * Math.sin(phi[i] - phi[j]) * rate[j] ** 2, 0),
  );
  const weights = [0, 1, 2].map((coordinate) =>
    p.mass.reduce((sum, mass, body) => sum + mass * coefficients[body][coordinate], 0),
  );
  const gravityTerms = weights.map(
    (weight, i) => -weight * (gravity[0] * Math.cos(phi[i]) + gravity[1] * Math.sin(phi[i])),
  );
  const relativeRates = [rate[0], rate[1] - rate[0], rate[2] - rate[1]];
  const damped = relativeRates.map((value, i) => value * p.damping[i]);
  const damping = [damped[0] - damped[1], damped[1] - damped[2], damped[2]];
  const acceleration = solve3(
    massMatrix,
    [0, 1, 2].map((i) => -(coriolis[i] + gravityTerms[i] + damping[i])),
  );
  return [rate[0], rate[1], rate[2], ...acceleration];
}

function addScaled(state: TripleState, derivative: TripleState, scale: number): TripleState {
  return state.map((value, index) => value + derivative[index] * scale) as TripleState;
}

export function simulateTriplePendulum(
  gravity: [number, number],
  dt: number,
  steps: number,
): TripleState[] {
  const parameters = golfTripleParameters();
  const result: TripleState[] = [[-Math.PI / 2, -Math.PI / 2, -Math.PI / 2, 0, 0, 0]];
  for (let index = 0; index < steps; index += 1) {
    const state = result[index];
    const k1 = derivatives(parameters, state, gravity);
    const k2 = derivatives(parameters, addScaled(state, k1, dt / 2), gravity);
    const k3 = derivatives(parameters, addScaled(state, k2, dt / 2), gravity);
    const k4 = derivatives(parameters, addScaled(state, k3, dt), gravity);
    result.push(
      state.map(
        (value, i) => value + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]),
      ) as TripleState,
    );
  }
  return result;
}
