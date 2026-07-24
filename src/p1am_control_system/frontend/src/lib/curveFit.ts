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
  let sumY = 0;
  for (const p of points) {
    sumY += p.y;
  }
  const meanY = sumY / points.length;
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
    let sumX = 0;
    let sumY = 0;
    for (const p of points) {
      sumX += p.x;
      sumY += p.y;
    }
    const meanX = sumX / n;
    const meanY = sumY / n;
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

// --------------------------------------------------------------------------
// Heat-up / cool-down rate (deg C per minute and per hour).
//
// The operator fits a linear trend over a chosen recent window to read the
// heater's ramp rate. These helpers are pure so they can be unit-tested and
// reused by any trend (DRY); the component just supplies the points, window,
// and the x-unit scale.
// --------------------------------------------------------------------------

/** A temperature ramp rate expressed in the two units operators asked for. */
export interface HeatUpRate {
  /** Degrees C per minute (positive = heating, negative = cooling). */
  perMinute: number;
  /** Degrees C per hour (positive = heating, negative = cooling). */
  perHour: number;
}

const MS_PER_MINUTE = 60_000;
const MS_PER_HOUR = 3_600_000;

/**
 * Keep only the points whose x is within `windowMs` of the most recent x, so a
 * fit regresses just the operator-chosen recent window. A non-positive or
 * non-finite `windowMs` means "no window" and returns the points unchanged.
 *
 * @throws TypeError if `points` is not an array.
 */
export function pointsInLastWindow(points: FitPoint[], windowMs: number): FitPoint[] {
  if (!Array.isArray(points)) {
    throw new TypeError("pointsInLastWindow: points must be an array");
  }
  if (!Number.isFinite(windowMs) || windowMs <= 0 || points.length === 0) {
    return points;
  }
  let maxX = -Infinity;
  for (const p of points) {
    if (Number.isFinite(p.x) && p.x > maxX) maxX = p.x;
  }
  if (!Number.isFinite(maxX)) return points;
  const cutoff = maxX - windowMs;
  return points.filter((p) => Number.isFinite(p.x) && p.x >= cutoff);
}

/**
 * Convert a linear fit's slope (deg C per x-unit) into a heat-up rate.
 *
 * The rate is only defined for the linear method (a straight-line ramp), so any
 * other / absent fit returns null and the caller can show a placeholder.
 * `msPerXUnit` is how many milliseconds one x-unit spans — 1 when x is in
 * milliseconds (as the live trend uses `Date.now()`).
 *
 * @throws RangeError if `msPerXUnit` is not a positive finite number.
 */
export function heatUpRateFromFit(
  fit: FitResult | null,
  msPerXUnit = 1,
): HeatUpRate | null {
  if (typeof msPerXUnit !== "number" || !Number.isFinite(msPerXUnit) || msPerXUnit <= 0) {
    throw new RangeError("heatUpRateFromFit: msPerXUnit must be a positive finite number");
  }
  if (!fit || fit.methodId !== "linear") return null;
  const slopePerX = fit.coeffs[0];
  if (typeof slopePerX !== "number" || !Number.isFinite(slopePerX)) return null;
  const perMs = slopePerX / msPerXUnit;
  return { perMinute: perMs * MS_PER_MINUTE, perHour: perMs * MS_PER_HOUR };
}

/**
 * Format a ramp rate for a readout box, e.g. "+12.3 °C/min · +738 °C/hr".
 * Returns a dash placeholder when no rate is available.
 */
export function formatHeatUpRate(rate: HeatUpRate | null): string {
  if (!rate) return "—";
  const sign = (n: number): string => (n >= 0 ? "+" : "−");
  const perMin = `${sign(rate.perMinute)}${Math.abs(rate.perMinute).toFixed(1)} °C/min`;
  const perHour = `${sign(rate.perHour)}${Math.abs(rate.perHour).toFixed(0)} °C/hr`;
  return `${perMin} · ${perHour}`;
}
