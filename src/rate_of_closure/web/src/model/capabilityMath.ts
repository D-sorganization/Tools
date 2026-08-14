/** Deterministic numerical helpers shared by capability parsing and optimization. */

const MATRIX_TOLERANCE = 1e-10;

/** Return a deterministic lower factor, rejecting non-positive-semidefinite matrices. */
export function covarianceFactor(matrix: readonly (readonly number[])[]): number[][] {
  const size = matrix.length;
  const factor = Array.from({ length: size }, () => Array<number>(size).fill(0));
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let residual = matrix[row][column];
      for (let index = 0; index < column; index += 1) {
        residual -= factor[row][index] * factor[column][index];
      }
      if (row === column) {
        if (residual < -MATRIX_TOLERANCE) throw new RangeError("capability matrix must be positive semidefinite");
        factor[row][column] = Math.sqrt(Math.max(0, residual));
      } else if (factor[column][column] > MATRIX_TOLERANCE) {
        factor[row][column] = residual / factor[column][column];
      } else if (Math.abs(residual) > MATRIX_TOLERANCE) {
        throw new RangeError("capability matrix must be positive semidefinite");
      }
    }
  }
  return factor;
}

const isPrime = (candidate: number): boolean => {
  for (let divisor = 2; divisor * divisor <= candidate; divisor += 1) {
    if (candidate % divisor === 0) return false;
  }
  return candidate >= 2;
};

/** Return the first `count` prime bases for a low-discrepancy sequence. */
export function primeBases(count: number): readonly number[] {
  const primes: number[] = [];
  for (let candidate = 2; primes.length < count; candidate += 1) {
    if (isPrime(candidate)) primes.push(candidate);
  }
  return primes;
}

/** Return one radical-inverse sample in [0, 1). */
export function radicalInverse(sourceIndex: number, base: number): number {
  let index = sourceIndex;
  let result = 0;
  let factor = 1 / base;
  while (index > 0) {
    result += factor * (index % base);
    index = Math.floor(index / base);
    factor /= base;
  }
  return result;
}
