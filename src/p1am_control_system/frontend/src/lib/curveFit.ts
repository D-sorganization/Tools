/**
 * Curve-fitting for the trend plots.
 *
 * A small registry of fit *methods* (currently just linear least-squares) sits
 * behind a stable {@link FitMethod} / {@link FitResult} contract, so a plot can
 * fit any single series and render the curve + equation without knowing which
 * method is active. Adding a method (polynomial, exponential, logarithmic, …)
 * is a one-line registry addition — nothing in the plots changes (open/closed).
 *
 * Pure math, no React/DOM — unit-testable and reusable (DRY). Every public
 * function validates its inputs (DbC).
 */

export interface FitPoint {
  x: number;
  y: number;
}

export interface FitResult {
  methodId: string;
  label: string;
  /** Method-specific coefficients (linear: [slope, intercept]). */
  coeffs: number[];
  /** Coefficient of determination R² in [0, 1]. */
  r2: number;
  /** Evaluate the fitted curve at x. */
  predict: (x: number) => number;
  /** Human-readable equation, e.g. "y = 1.23·x + 4.5". */
  equation: string;
}

export interface FitMethod {
  id: string;
  label: string;
  /** Fewest points the method needs to be well-posed. */
  minPoints: number;
  /**
   * Fit the method to the points.
   * @throws RangeError if the points are too few or degenerate for the method.
   *   Prefer {@link fitSeries}, which guards and returns null instead.
   */
  fit: (points: FitPoint[]) => FitResult;
}

/** Sentinel id meaning "no fit selected". */
export const NO_FIT_ID = "none";

/** Adaptive number formatting: avoids 12.0000001 and 0.00000 noise. */
function fmt(n: number): string {
  if (!Number.isFinite(n)) return "?";
  const abs = Math.abs(n);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return n.toExponential(2);
  const rounded = Number(n.toFixed(4));
  return `${rounded}`;
}

/** R² of a model's predictions against the observed points. */
function rSquared(points: FitPoint[], predict: (x: number) => number): number {
  const meanY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  let ssRes = 0;
  let ssTot = 0;
  for (const p of points) {
    ssRes += (p.y - predict(p.x)) ** 2;
    ssTot += (p.y - meanY) ** 2;
  }
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return Math.max(0, Math.min(1, 1 - ssRes / ssTot));
}

export const linearFit: FitMethod = {
  id: "linear",
  label: "Linear",
  minPoints: 2,
  fit(points: FitPoint[]): FitResult {
    if (points.length < 2) {
      throw new RangeError("linearFit: need at least 2 points");
    }
    const n = points.length;
    const meanX = points.reduce((s, p) => s + p.x, 0) / n;
    const meanY = points.reduce((s, p) => s + p.y, 0) / n;
    let sxx = 0;
    let sxy = 0;
    for (const p of points) {
      const dx = p.x - meanX;
      sxx += dx * dx;
      sxy += dx * (p.y - meanY);
    }
    if (sxx === 0) {
      throw new RangeError("linearFit: x has zero variance (vertical line)");
    }
    const slope = sxy / sxx;
    const intercept = meanY - slope * meanX;
    const predict = (x: number): number => slope * x + intercept;
    const sign = intercept >= 0 ? "+" : "−";
    return {
      methodId: "linear",
      label: "Linear",
      coeffs: [slope, intercept],
      r2: rSquared(points, predict),
      predict,
      equation: `y = ${fmt(slope)}·x ${sign} ${fmt(Math.abs(intercept))}`,
    };
  },
};

/** All registered fit methods. Add new methods here (open/closed). */
export const FIT_METHODS: readonly FitMethod[] = [linearFit];

/** Dropdown options including the "None" sentinel. */
export function fitOptions(): { id: string; label: string }[] {
  return [{ id: NO_FIT_ID, label: "None" }, ...FIT_METHODS.map((m) => ({ id: m.id, label: m.label }))];
}

export function getFitMethod(id: string): FitMethod | null {
  return FIT_METHODS.find((m) => m.id === id) ?? null;
}

/**
 * Fit a series safely: looks up the method, drops non-finite points, and
 * returns null (rather than throwing) when the method is unknown, "none", or
 * the data is too few / degenerate. This is the entry point plots should use.
 */
export function fitSeries(points: FitPoint[], methodId: string): FitResult | null {
  if (methodId === NO_FIT_ID) return null;
  const method = getFitMethod(methodId);
  if (!method) return null;
  const clean = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (clean.length < method.minPoints) return null;
  try {
    return method.fit(clean);
  } catch {
    return null;
  }
}
