/** Numerically safe scientific invariants for finite Morris report metrics. */

import type { MorrisAvailability, MorrisEffects } from "./morrisGlobalSensitivityContract";

const PRODUCER_CLAMP_EPSILON_MULTIPLIER = 64;
const IDENTITY_ROUNDING_EPSILON_MULTIPLIER = 32;
const MAX_SAFELY_SQUARED_METRIC = Math.sqrt(Number.MAX_VALUE);

interface FiniteMetrics {
  readonly mu: number;
  readonly muStar: number;
  readonly standardError: number;
  readonly sigma: number;
}

const finiteMetrics = (effects: MorrisEffects): FiniteMetrics | null => {
  const { mu, muStar, muStarStandardError, sigma } = effects;
  if (mu === null || muStar === null || muStarStandardError === null || sigma === null) return null;
  return { mu, muStar, standardError: muStarStandardError, sigma };
};

const requireSafeMagnitude = (metrics: FiniteMetrics): void => {
  if (Object.values(metrics).some((metric) => Math.abs(metric) > MAX_SAFELY_SQUARED_METRIC)) {
    throw new RangeError("Morris metrics must be safely squared finite values");
  }
};

const requireWireClamp = (metric: number, delta: number, name: string): void => {
  if (metric > 0 && metric <= delta) {
    throw new RangeError(`serialized ${name} must be zero or strictly above the producer clamp`);
  }
};

const ordinaryDifferenceTolerance = (first: number, second: number): number => (
  IDENTITY_ROUNDING_EPSILON_MULTIPLIER * Number.EPSILON
  * Math.max(Math.abs(first), Math.abs(second))
);

const normalizedIdentity = (metrics: FiniteMetrics, sampleCount: number, delta: number): void => {
  const scale = Math.max(
    Math.abs(metrics.mu), metrics.muStar, metrics.standardError, metrics.sigma, delta,
  );
  const mu = metrics.mu / scale;
  const muStar = metrics.muStar / scale;
  const standardError = metrics.standardError / scale;
  const sigma = metrics.sigma / scale;
  const normalizedDelta = delta / scale;
  const correction = sampleCount / (sampleCount - 1);
  const sigmaTerm = sigma ** 2;
  const standardErrorTerm = sampleCount * standardError ** 2;
  const muStarTerm = correction * muStar ** 2;
  const muTerm = correction * mu ** 2;
  const residual = sigmaTerm - standardErrorTerm - muStarTerm + muTerm;
  const termScale = Math.abs(sigmaTerm) + Math.abs(standardErrorTerm)
    + Math.abs(muStarTerm) + Math.abs(muTerm);
  const rounding = IDENTITY_ROUNDING_EPSILON_MULTIPLIER * Number.EPSILON * termScale;
  const sigmaClamp = metrics.sigma === 0 ? normalizedDelta ** 2 : 0;
  const standardErrorClamp = metrics.standardError === 0
    ? sampleCount * normalizedDelta ** 2 : 0;
  if (Math.abs(residual) > rounding + sigmaClamp + standardErrorClamp) {
    throw new RangeError("Morris metric identity is inconsistent with valid_pairs");
  }
};

/** Validate realizability and the producer's exact zero-clamp wire semantics. */
export function validateMorrisMetrics(
  effects: MorrisEffects, availability: MorrisAvailability, sampleCount: number,
): void {
  const metrics = finiteMetrics(effects);
  if (metrics === null) return;
  requireSafeMagnitude(metrics);
  if (metrics.muStar === 0) {
    if (metrics.mu !== 0 || metrics.sigma !== 0 || metrics.standardError !== 0
        || availability !== "constant-output") {
      throw new RangeError("zero mu_star requires zero metrics and constant-output availability");
    }
    return;
  }
  if (availability === "constant-output") throw new RangeError("constant-output Morris effects must be zero");
  const delta = PRODUCER_CLAMP_EPSILON_MULTIPLIER * Number.EPSILON * Math.max(1, metrics.muStar);
  requireWireClamp(metrics.sigma, delta, "sigma");
  requireWireClamp(metrics.standardError, delta, "standard error");
  const meanDifference = Math.abs(metrics.muStar - Math.abs(metrics.mu));
  const meanTolerance = ordinaryDifferenceTolerance(metrics.muStar, Math.abs(metrics.mu));
  if (meanDifference <= meanTolerance && metrics.standardError === 0
      && metrics.sigma > Math.sqrt(sampleCount) * delta + meanTolerance) {
    throw new RangeError("Morris metric clamp-scale degeneracy is inconsistent");
  }
  if (metrics.sigma === 0 && (metrics.standardError !== 0 || meanDifference > meanTolerance)) {
    throw new RangeError("zero-sigma Morris metric relationship failed");
  }
  normalizedIdentity(metrics, sampleCount, delta);
}
