/**
 * Advanced Analytics Suite for the Data Processor.
 *
 * Provides:
 *  - Correlation matrix with heatmap visualization
 *  - PCA (Principal Component Analysis) with scree plot
 *  - Regression analysis with residual diagnostics
 *
 * See issue #607.
 */
import { useState, useMemo, useCallback } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
  LineChart,
  Line,
  Legend,
} from 'recharts';
import type {
  DataRow,
  CorrelationMatrix,
  PCAResult,
  RegressionResult,
} from '../types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface AnalyticsSuiteProps {
  data: DataRow[];
  signals: string[];
  selectedSignals: string[];
}

type AnalyticsTab = 'correlation' | 'pca' | 'regression';

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

function pearsonCorrelation(x: number[] | Float64Array, y: number[] | Float64Array): number {
  const len = x.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0, count = 0;

  for (let i = 0; i < len; i++) {
    const vx = x[i];
    const vy = y[i];
    if (!Number.isNaN(vx) && !Number.isNaN(vy)) {
      sumX += vx;
      sumY += vy;
      sumXY += vx * vy;
      sumX2 += vx * vx;
      sumY2 += vy * vy;
      count++;
    }
  }

  if (count < 2) return NaN;

  const num = count * sumXY - sumX * sumY;
  const denX = count * sumX2 - sumX * sumX;
  const denY = count * sumY2 - sumY * sumY;

  // Due to floating point inaccuracy, denX or denY might be very slightly negative
  // if variance is essentially 0. Clamp to 0 before sqrt.
  const den = Math.sqrt(Math.max(0, denX) * Math.max(0, denY));

  return den === 0 ? 0 : num / den;
}

function computeCorrelation(data: DataRow[], signals: string[]): CorrelationMatrix {
  const n = signals.length;
  const rowCount = data.length;
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  // ⚡ Bolt Optimization: Allocate column-wise buffers using Float64Array
  // instead of allocating standard array rows (N x P) via chained map() calls.
  // This avoids massive garbage collection pauses and O(N) allocations.
  const columns = signals.map((sig) => {
    const col = new Float64Array(rowCount);
    for (let i = 0; i < rowCount; i++) {
      const val = data[i][sig];
      col[i] = typeof val === 'number' ? val : NaN;
    }
    return col;
  });

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      // Avoid massive map/filter arrays for every pair calculation
      const r = pearsonCorrelation(columns[i], columns[j]);
      matrix[i][j] = r;
      matrix[j][i] = r;
    }
  }

  return { signals, matrix };
}

/**
 * Simple PCA via covariance eigen-decomposition (power iteration for the
 * top-k eigenvalues).  Good enough for interactive analytics on moderate
 * datasets.
 */
function computePCA(data: DataRow[], signals: string[], numComponents?: number): PCAResult {
  const n = data.length;
  const p = signals.length;
  const nc = Math.min(numComponents ?? p, p);

  // ⚡ Bolt Optimization: Allocate column-wise buffers using Float64Array
  // instead of allocating standard array rows (N x P) via chained map() calls.
  // This drastically reduces garbage collection overhead and improves execution speed.
  const cols = signals.map((sig) => {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const val = data[i][sig];
      col[i] = typeof val === 'number' ? val : 0;
    }
    return col;
  });

  // ⚡ Bolt Optimization: Replace map/reduce chains with a single-pass loop for variance
  // Reduces O(N*P) array allocations and amortized callback execution overhead
  const means: number[] = new Array(p);
  const stds: number[] = new Array(p);

  for (let ci = 0; ci < p; ci++) {
    const c = cols[ci];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += c[i];
    }
    const mean = sum / n;
    means[ci] = mean;

    let sqSum = 0;
    for (let i = 0; i < n; i++) {
      const diff = c[i] - mean;
      sqSum += diff * diff;
    }
    const s = Math.sqrt(sqSum / n);
    stds[ci] = s === 0 ? 1 : s;
  }

  // Standardized column matrix (p x n) using Float64Array for performance
  // This avoids O(N) row allocations and speeds up covariance / score calculations
  const Z_cols: Float64Array[] = Array.from({ length: p }, (_, j) => {
    const colBuffer = new Float64Array(n);
    const mean = means[j];
    const std = stds[j];
    const c = cols[j];
    for (let i = 0; i < n; i++) {
      colBuffer[i] = (c[i] - mean) / std;
    }
    return colBuffer;
  });

  // Covariance matrix (p x p)
  const cov: number[][] = Array.from({ length: p }, () => Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    const zi = Z_cols[i];
    for (let j = i; j < p; j++) {
      const zj = Z_cols[j];
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += zi[k] * zj[k];
      }
      s /= n - 1 || 1;
      cov[i][j] = s;
      cov[j][i] = s;
    }
  }

  // Power-iteration for top eigenvalues (simple Jacobi-like)
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];
  const A = cov.map((row) => [...row]);

  // ⚡ Bolt Optimization: Pre-allocate Av to avoid thousands of array allocations inside the tight iteration loop.
  const Av = new Array<number>(p);

  for (let comp = 0; comp < nc; comp++) {
    let v = Array.from({ length: p }, () => Math.random());

    // Initial norm
    let sqSum = 0;
    for (let i = 0; i < p; i++) {
      sqSum += v[i] * v[i];
    }
    let norm = Math.sqrt(sqSum);
    if (norm > 0) {
      for (let i = 0; i < p; i++) {
        v[i] /= norm;
      }
    }

    // ⚡ Bolt: Replaced .map() and .reduce() with standard loops.
    // Performance impact: Eliminates thousands of array allocations per PCA execution and avoids callback overhead.
    for (let iter = 0; iter < 300; iter++) {
      let nextSqSum = 0;
      for (let i = 0; i < p; i++) {
        let s = 0;
        const row = A[i];
        for (let j = 0; j < p; j++) {
          s += row[j] * v[j];
        }
        Av[i] = s;
        nextSqSum += s * s;
      }

      norm = Math.sqrt(nextSqSum);
      if (norm === 0) break;

      for (let i = 0; i < p; i++) {
        v[i] = Av[i] / norm;
      }
    }

    eigenvalues.push(norm);
    eigenvectors.push([...v]);

    // Deflate
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        A[i][j] -= norm * v[i] * v[j];
      }
    }
  }

  const totalVar = eigenvalues.reduce((a, b) => a + b, 0) || 1;
  const explained = eigenvalues.map((e) => e / totalVar);
  const cumulative: number[] = [];
  explained.reduce((acc, e, i) => {
    cumulative[i] = acc + e;
    return cumulative[i];
  }, 0);

  // Scores (n x nc)
  const scores: number[][] = Array.from({ length: n }, () => new Array(nc));
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nc; c++) {
      let s = 0;
      const ev = eigenvectors[c];
      for (let j = 0; j < p; j++) {
        s += Z_cols[j][i] * ev[j];
      }
      scores[i][c] = s;
    }
  }

  // Loadings (p x nc)
  const loadings = eigenvectors.map((ev) => [...ev]);

  return {
    explainedVariance: explained,
    cumulativeVariance: cumulative,
    numComponents: nc,
    scores,
    loadings: loadings[0].map((_, ci) => eigenvectors.map((ev) => ev[ci])),
    signals,
  };
}

function computeRegression(
  data: DataRow[],
  xSignal: string,
  ySignal: string,
  degree: number,
): RegressionResult {
  // ⚡ Bolt Optimization: Pre-allocate Float64Array buffers instead of pushing to standard JS arrays.
  // Performance impact: Eliminates array reallocation and garbage collection overhead when filtering data.
  const maxN = data.length;
  const xsBuffer = new Float64Array(maxN);
  const ysBuffer = new Float64Array(maxN);
  let n = 0;

  for (let i = 0; i < maxN; i++) {
    const row = data[i];
    const x = row[xSignal];
    const y = row[ySignal];

    if (typeof x === 'number' && typeof y === 'number' && !Number.isNaN(x) && !Number.isNaN(y)) {
      xsBuffer[n] = x;
      ysBuffer[n] = y;
      n++;
    }
  }

  const xs = Array.from(xsBuffer.subarray(0, n));
  const ys = Array.from(ysBuffer.subarray(0, n));

  let coefficients: number[];
  let predictions: number[];

  if (degree === 1) {
    // ⚡ Bolt Optimization: Calculate linear regression in single pass
    // instead of chained reduce and map to prevent allocation overhead
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;

    for (let i = 0; i < n; i++) {
      const x = xs[i];
      const y = ys[i];
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
    }

    const denom = n * sumXX - sumX * sumX;
    const slope = denom === 0 ? 0 : (n * sumXY - sumX * sumY) / denom;
    const intercept = (sumY - slope * sumX) / n;
    coefficients = [intercept, slope];

    predictions = new Array(n);
    for (let i = 0; i < n; i++) {
      predictions[i] = intercept + slope * xs[i];
    }
  } else {
    // Polynomial regression via normal equations
    // ⚡ Bolt Optimization: Replace multiple .reduce()/.map() calls with a single-pass loop
    // Performance impact: Drastically reduces array allocations and callback overhead for quadratic regressions.
    const cols = degree + 1;
    const XtX: number[][] = Array.from({ length: cols }, () => new Array(cols).fill(0));
    const XtY: number[] = new Array(cols).fill(0);

    for (let k = 0; k < n; k++) {
      const x = xs[k];
      const y = ys[k];

      let x_i = 1;
      for (let i = 0; i < cols; i++) {
        XtY[i] += x_i * y;
        let x_ij = x_i * x_i;
        for (let j = i; j < cols; j++) {
          XtX[i][j] += x_ij;
          if (i !== j) {
            XtX[j][i] += x_ij;
          }
          x_ij *= x;
        }
        x_i *= x;
      }
    }

    // Solve via Gaussian elimination
    coefficients = solveLinearSystem(XtX, XtY);

    // ⚡ Bolt Optimization: Replace map/reduce chains with a single-pass loop for predictions
    // Performance impact: Speeds up prediction calculation by avoiding map/reduce allocations and optimizing polynomial evaluation.
    predictions = new Array(n);
    const p = coefficients.length;
    for (let i = 0; i < n; i++) {
      const x = xs[i];
      let s = coefficients[0];
      let x_pow = x;
      for (let d = 1; d < p; d++) {
        s += coefficients[d] * x_pow;
        x_pow *= x;
      }
      predictions[i] = s;
    }
  }

  // ⚡ Bolt Optimization: Use single-pass loops for residuals and sum of squares
  // to avoid .map() allocations and chained .reduce() callback overhead
  const residuals = new Array(n);
  let sumY = 0;
  let ssResidual = 0;

  for (let i = 0; i < n; i++) {
    const y = ys[i];
    const r = y - predictions[i];
    residuals[i] = r;
    sumY += y;
    ssResidual += r * r;
  }

  const meanY = sumY / n;
  let ssTotal = 0;
  for (let i = 0; i < n; i++) {
    ssTotal += (ys[i] - meanY) ** 2;
  }
  const rSquared = ssTotal === 0 ? 1 : 1 - ssResidual / ssTotal;
  const p = coefficients.length;
  const adjustedRSquared = n <= p ? rSquared : 1 - ((1 - rSquared) * (n - 1)) / (n - p);

  // Build equation string
  let equation: string;
  if (degree === 1) {
    equation = `y = ${coefficients[1].toFixed(4)}x + ${coefficients[0].toFixed(4)}`;
  } else {
    const terms = coefficients
      .map((c, d) => {
        if (d === 0) return c.toFixed(4);
        const sign = c >= 0 ? ' + ' : ' - ';
        const coef = Math.abs(c).toFixed(4);
        const xPart = d === 1 ? 'x' : `x^${d}`;
        return `${sign}${coef}${xPart}`;
      })
      .join('');
    equation = `y = ${terms}`;
  }

  return {
    type: degree === 1 ? 'linear' : 'polynomial',
    equation,
    rSquared,
    adjustedRSquared,
    coefficients,
    residuals,
    predictions,
    xSignal,
    ySignal,
  };
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  // Forward elimination
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
    }
    [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];

    if (Math.abs(aug[i][i]) < 1e-12) continue;

    for (let k = i + 1; k < n; k++) {
      const factor = aug[k][i] / aug[i][i];
      for (let j = i; j <= n; j++) {
        aug[k][j] -= factor * aug[i][j];
      }
    }
  }

  // Back substitution
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = aug[i][n];
    for (let j = i + 1; j < n; j++) s -= aug[i][j] * x[j];
    x[i] = Math.abs(aug[i][i]) < 1e-12 ? 0 : s / aug[i][i];
  }
  return x;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const COLORS_POSITIVE = ['#0d47a1', '#1565c0', '#1976d2', '#42a5f5', '#90caf9'];
const COLORS_NEGATIVE = ['#b71c1c', '#c62828', '#d32f2f', '#ef5350', '#ef9a9a'];

function correlationColor(r: number): string {
  if (isNaN(r)) return '#4a4a4a';
  const idx = Math.min(4, Math.floor(Math.abs(r) * 5));
  return r >= 0 ? COLORS_POSITIVE[4 - idx] : COLORS_NEGATIVE[4 - idx];
}

export function AnalyticsSuite({ data, signals, selectedSignals }: AnalyticsSuiteProps) {
  const [tab, setTab] = useState<AnalyticsTab>('correlation');
  const [regXSignal, setRegXSignal] = useState<string>(selectedSignals[0] ?? '');
  const [regYSignal, setRegYSignal] = useState<string>(selectedSignals[1] ?? selectedSignals[0] ?? '');
  const [regDegree, setRegDegree] = useState<number>(1);

  const activeSignals = selectedSignals.length > 0 ? selectedSignals : signals.slice(0, 5);

  // Correlation
  const correlation = useMemo<CorrelationMatrix | null>(() => {
    if (data.length === 0 || activeSignals.length < 2) return null;
    return computeCorrelation(data, activeSignals);
  }, [data, activeSignals]);

  // PCA
  const pca = useMemo<PCAResult | null>(() => {
    if (data.length === 0 || activeSignals.length < 2) return null;
    return computePCA(data, activeSignals);
  }, [data, activeSignals]);

  // Regression (on demand)
  const [regression, setRegression] = useState<RegressionResult | null>(null);

  const runRegression = useCallback(() => {
    if (!regXSignal || !regYSignal || data.length === 0) return;
    const result = computeRegression(data, regXSignal, regYSignal, regDegree);
    setRegression(result);
  }, [data, regXSignal, regYSignal, regDegree]);

  // Memoize PCA chart data
  const pcaScreeData = useMemo(() => {
    if (!pca) return [];
    return pca.explainedVariance.map((ev, i) => ({
      name: `PC${i + 1}`,
      variance: +(ev * 100).toFixed(1),
      cumulative: +(pca.cumulativeVariance[i] * 100).toFixed(1),
    }));
  }, [pca]);

  const pcaScatterData = useMemo(() => {
    if (!pca || pca.numComponents < 2) return [];
    return pca.scores.map((s) => ({ x: s[0], y: s[1] }));
  }, [pca]);

  // Memoize Regression chart data
  const regressionScatterData = useMemo(() => {
    if (!regression) return [];
    // ⚡ Bolt Optimization: Use single-pass for loop to build scatter data, avoiding chained .map().filter() and GC overhead
    const result = [];
    for (let i = 0; i < data.length; i++) {
      const row = data[i];
      const x = row[regression.xSignal];
      const y = row[regression.ySignal];
      if (typeof x === 'number' && typeof y === 'number') {
        result.push({ x, y });
      }
    }
    return result;
  }, [data, regression]);

  const regressionResidualsData = useMemo(() => {
    if (!regression) return [];
    return regression.residuals.map((r, i) => ({ index: i, residual: r }));
  }, [regression]);

  if (data.length === 0 || activeSignals.length < 2) {
    return (
      <div className="card">
        <div className="card-header">Analytics Suite</div>
        <div className="card-body text-dark-400 text-center py-8">
          Load data and select at least 2 signals to use analytics.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Tab Selector */}
      <div className="flex border-b border-dark-700 text-sm">
        {(['correlation', 'pca', 'regression'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 capitalize ${
              tab === t
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-dark-400 hover:text-dark-300'
            }`}
          >
            {t === 'pca' ? 'PCA' : t}
          </button>
        ))}
      </div>

      {/* --- Correlation Matrix --- */}
      {tab === 'correlation' && correlation && (
        <div className="card">
          <div className="card-header">Correlation Matrix</div>
          <div className="card-body overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr>
                  <th className="py-1 px-2 text-left text-dark-400"></th>
                  {correlation.signals.map((s) => (
                    <th key={s} className="py-1 px-2 text-dark-400 text-center truncate max-w-[6rem]">
                      {s}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlation.signals.map((rowSig, ri) => (
                  <tr key={rowSig}>
                    <td className="py-1 px-2 text-dark-400 font-medium truncate max-w-[6rem]">
                      {rowSig}
                    </td>
                    {correlation.matrix[ri].map((r, ci) => (
                      <td
                        key={ci}
                        className="py-1 px-2 text-center font-mono"
                        style={{
                          backgroundColor: correlationColor(r),
                          color: Math.abs(r) > 0.5 ? '#fff' : '#ccc',
                        }}
                      >
                        {isNaN(r) ? '-' : r.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* --- PCA --- */}
      {tab === 'pca' && pca && (
        <div className="space-y-4">
          {/* Scree Plot */}
          <div className="card">
            <div className="card-header">Scree Plot (Variance Explained)</div>
            <div className="card-body h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={pcaScreeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" label={{ value: '%', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Legend />
                  <Bar dataKey="variance" name="Individual %" fill="#3b82f6" />
                  <Bar dataKey="cumulative" name="Cumulative %" fill="#22c55e" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* PCA Biplot (PC1 vs PC2) */}
          {pca.numComponents >= 2 && (
            <div className="card">
              <div className="card-header">
                PCA Score Plot (PC1 vs PC2)
              </div>
              <div className="card-body h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="PC1"
                      stroke="#94a3b8"
                      label={{ value: `PC1 (${(pca.explainedVariance[0] * 100).toFixed(1)}%)`, position: 'bottom', fill: '#94a3b8' }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="PC2"
                      stroke="#94a3b8"
                      label={{ value: `PC2 (${(pca.explainedVariance[1] * 100).toFixed(1)}%)`, angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                    />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                    <Scatter data={pcaScatterData} fill="#3b82f6">
                      {pcaScatterData.map((_, i) => (
                        <Cell key={i} fill="#3b82f6" opacity={0.6} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Loadings table */}
          <div className="card">
            <div className="card-header">Component Loadings</div>
            <div className="card-body overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-dark-400 border-b border-dark-700">
                    <th className="text-left py-1 px-2">Signal</th>
                    {pca.explainedVariance.map((_, i) => (
                      <th key={i} className="text-right py-1 px-2">PC{i + 1}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pca.signals.map((sig, si) => (
                    <tr key={sig} className="border-b border-dark-800">
                      <td className="py-1 px-2 text-dark-300">{sig}</td>
                      {pca.loadings[si]?.map((l, ci) => (
                        <td key={ci} className="text-right py-1 px-2 font-mono text-dark-300">
                          {l.toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* --- Regression --- */}
      {tab === 'regression' && (
        <div className="space-y-4">
          {/* Controls */}
          <div className="card">
            <div className="card-header">Regression Setup</div>
            <div className="card-body space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-dark-400 mb-1">X Signal</label>
                  <select
                    value={regXSignal}
                    onChange={(e) => setRegXSignal(e.target.value)}
                    className="w-full bg-dark-700 text-dark-100 rounded px-2 py-1 text-sm border border-dark-600"
                  >
                    {signals.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-dark-400 mb-1">Y Signal</label>
                  <select
                    value={regYSignal}
                    onChange={(e) => setRegYSignal(e.target.value)}
                    className="w-full bg-dark-700 text-dark-100 rounded px-2 py-1 text-sm border border-dark-600"
                  >
                    {signals.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-dark-400 mb-1">Degree</label>
                  <select
                    value={regDegree}
                    onChange={(e) => setRegDegree(Number(e.target.value))}
                    className="w-full bg-dark-700 text-dark-100 rounded px-2 py-1 text-sm border border-dark-600"
                  >
                    {[1, 2, 3, 4, 5].map((d) => (
                      <option key={d} value={d}>{d === 1 ? 'Linear' : `Polynomial (${d})`}</option>
                    ))}
                  </select>
                </div>
              </div>
              <button
                onClick={runRegression}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2 rounded transition-colors"
              >
                Run Regression
              </button>
            </div>
          </div>

          {/* Results */}
          {regression && (
            <>
              {/* Summary */}
              <div className="card">
                <div className="card-header">Regression Results</div>
                <div className="card-body space-y-2 text-sm">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-dark-400">Equation:</span>{' '}
                      <span className="text-dark-100 font-mono text-xs">{regression.equation}</span>
                    </div>
                    <div>
                      <span className="text-dark-400">R-squared:</span>{' '}
                      <span className="text-green-400 font-mono">{regression.rSquared.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-dark-400">Adj. R-squared:</span>{' '}
                      <span className="text-green-400 font-mono">{regression.adjustedRSquared.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-dark-400">Type:</span>{' '}
                      <span className="text-dark-100 capitalize">{regression.type}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Scatter + Fit */}
              <div className="card">
                <div className="card-header">Fit Plot</div>
                <div className="card-body h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" dataKey="x" name={regression.xSignal} stroke="#94a3b8" />
                      <YAxis type="number" dataKey="y" name={regression.ySignal} stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                      <Scatter
                        name="Data"
                        data={regressionScatterData}
                        fill="#3b82f6"
                        opacity={0.5}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Residual Plot */}
              <div className="card">
                <div className="card-header">Residual Plot</div>
                <div className="card-body h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={regressionResidualsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="index" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                      <Line
                        type="monotone"
                        dataKey="residual"
                        stroke="#f59e0b"
                        dot={false}
                        strokeWidth={1}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default AnalyticsSuite;
