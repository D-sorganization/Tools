/**
 * Mardia bivariate normality diagnostic and convex hull fallback (issue #4253 item c).
 */

export interface NormalityDiagnosticTs {
  isNormal: boolean;
  pValue: number;
  skewnessStat: number;
  skewnessPValue: number;
  kurtosisStat: number;
  kurtosisPValue: number;
  n: number;
}

function normalCdf(x: number): number {
  const t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - prob : prob;
}

/** Mardia bivariate normality diagnostic for landing points. */
export function mardiaBivariateNormality(
  points: Array<[number, number]>,
  alpha = 0.05,
): NormalityDiagnosticTs {
  const n = points.length;
  if (n < 8) {
    return { isNormal: true, pValue: 1, skewnessStat: 0, skewnessPValue: 1, kurtosisStat: 8, kurtosisPValue: 1, n };
  }
  let mx = 0; let my = 0;
  for (const [x, y] of points) { mx += x; my += y; }
  mx /= n; my /= n;

  let sxx = 0; let syy = 0; let sxy = 0;
  for (const [x, y] of points) {
    const dx = x - mx; const dy = y - my;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  }
  sxx /= n; syy /= n; sxy /= n;
  const det = sxx * syy - sxy * sxy;
  if (det < 1e-12) {
    return { isNormal: false, pValue: 0, skewnessStat: Infinity, skewnessPValue: 0, kurtosisStat: Infinity, kurtosisPValue: 0, n };
  }
  const invXx = syy / det; const invYy = sxx / det; const invXy = -sxy / det;

  let b12 = 0;
  let b22 = 0;
  for (let i = 0; i < n; i += 1) {
    const dxi = points[i][0] - mx; const dyi = points[i][1] - my;
    const mii = dxi * (invXx * dxi + invXy * dyi) + dyi * (invXy * dxi + invYy * dyi);
    b22 += mii * mii;
    for (let j = 0; j < n; j += 1) {
      const dxj = points[j][0] - mx; const dyj = points[j][1] - my;
      const mij = dxi * (invXx * dxj + invXy * dyj) + dyi * (invXy * dxj + invYy * dyj);
      b12 += mij * mij * mij;
    }
  }
  b12 /= n * n;
  b22 /= n;
  const tSkew = (n / 6.0) * b12;
  const pSkew = Math.max(0, Math.min(1, (1 + tSkew / 2) * Math.exp(-tSkew / 2)));
  const zKurt = (b22 - 8.0) / Math.sqrt(64.0 / n);
  const pKurt = Math.max(0, Math.min(1, 2 * (1 - normalCdf(Math.abs(zKurt)))));
  const pValue = Math.min(pSkew, pKurt);
  return { isNormal: pSkew >= alpha && pKurt >= alpha, pValue, skewnessStat: b12, skewnessPValue: pSkew, kurtosisStat: b22, kurtosisPValue: pKurt, n };
}

/** 2D convex hull via monotone chain algorithm. */
export function convexHull2d(points: Array<[number, number]>): Array<[number, number]> {
  if (points.length < 3) return points.slice();
  const sorted = points.slice().sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
  const cross = (o: [number, number], a: [number, number], b: [number, number]): number =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}
