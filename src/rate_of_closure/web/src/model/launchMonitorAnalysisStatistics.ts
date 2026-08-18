/** Statistical primitives for launch-monitor analysis. */

import {
  finiteLaunchMonitorScalar,
  type CorrelationEstimate,
  type CorrelationMethod,
  type LaunchMonitorAnalysisRequest,
  type LaunchMonitorRow,
  type RegressionEstimate,
} from "./launchMonitorAnalysisTypes";

interface RegressionWork {
  count: number;
  parameterCount: number;
  outcome: number[];
  design: number[][];
  beta: number[];
  residuals: number[];
  inverseInformation: number[][];
}

const mean = (values: number[]) =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

const variance = (values: number[], degrees = 1): number => {
  const center = mean(values);
  return values.reduce((sum, value) => sum + (value - center) ** 2, 0) /
    Math.max(1, values.length - degrees);
};

const erf = (value: number): number => {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
    0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-(x ** 2)));
};

const normalCdf = (value: number) => 0.5 * (1 + erf(value / Math.sqrt(2)));

const logGamma = (value: number): number => {
  const coefficients = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.6150291621406, 12.507343278686905, -0.13857109526572012,
    9.984369578019572e-6, 1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  let x = 0.9999999999998099;
  const shifted = value - 1;
  coefficients.forEach((coefficient, index) => {
    x += coefficient / (shifted + index + 1);
  });
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
};

const betaFraction = (a: number, b: number, x: number): number => {
  const maxIterations = 200;
  const epsilon = 3e-14;
  const tiny = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let result = d;
  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const twice = 2 * iteration;
    let aa = iteration * (b - iteration) * x / ((qam + twice) * (a + twice));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    result *= d * c;
    aa = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    result *= delta;
    if (Math.abs(delta - 1) < epsilon) break;
  }
  return result;
};

const regularizedBeta = (x: number, a: number, b: number): number => {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) +
    a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? front * betaFraction(a, b, x) / a
    : 1 - front * betaFraction(b, a, 1 - x) / b;
};

const studentTwoSidedP = (tStatistic: number, degrees: number): number => {
  if (!Number.isFinite(tStatistic) || degrees <= 0) return 0;
  const x = degrees / (degrees + tStatistic ** 2);
  return Math.min(1, Math.max(0, regularizedBeta(x, degrees / 2, 0.5)));
};

const normalQuantile = (probability: number): number => {
  let low = -8;
  let high = 8;
  for (let index = 0; index < 80; index += 1) {
    const middle = (low + high) / 2;
    if (normalCdf(middle) < probability) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
};

const studentQuantile = (probability: number, degrees: number): number => {
  let low = -20;
  let high = 20;
  for (let index = 0; index < 90; index += 1) {
    const middle = (low + high) / 2;
    const cdf = middle >= 0
      ? 1 - studentTwoSidedP(middle, degrees) / 2
      : studentTwoSidedP(middle, degrees) / 2;
    if (cdf < probability) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
};

const ranks = (values: number[]): number[] => {
  const order = values.map((value, index) => ({ value, index }))
    .sort((left, right) => left.value - right.value);
  const result = Array(values.length).fill(0) as number[];
  let start = 0;
  while (start < order.length) {
    let end = start + 1;
    while (end < order.length && order[end].value === order[start].value) end += 1;
    const rank = (start + end + 1) / 2;
    for (let index = start; index < end; index += 1) result[order[index].index] = rank;
    start = end;
  }
  return result;
};

const pearson = (left: number[], right: number[]): number => {
  const leftMean = mean(left);
  const rightMean = mean(right);
  let numerator = 0;
  let leftSum = 0;
  let rightSum = 0;
  left.forEach((value, index) => {
    const x = value - leftMean;
    const y = right[index] - rightMean;
    numerator += x * y;
    leftSum += x * x;
    rightSum += y * y;
  });
  return numerator / Math.sqrt(leftSum * rightSum);
};

const kendall = (left: number[], right: number[]): number => {
  let concordant = 0;
  let discordant = 0;
  let leftTies = 0;
  let rightTies = 0;
  for (let first = 0; first < left.length; first += 1) {
    for (let second = first + 1; second < left.length; second += 1) {
      const dx = Math.sign(left[first] - left[second]);
      const dy = Math.sign(right[first] - right[second]);
      if (dx === 0 && dy !== 0) leftTies += 1;
      else if (dy === 0 && dx !== 0) rightTies += 1;
      else if (dx * dy > 0) concordant += 1;
      else if (dx * dy < 0) discordant += 1;
    }
  }
  return (concordant - discordant) / Math.sqrt(
    (concordant + discordant + leftTies) * (concordant + discordant + rightTies),
  );
};

const correlation = (
  left: number[], right: number[], method: CorrelationMethod,
): { coefficient: number; pValue: number } => {
  const coefficient = method === "pearson" ? pearson(left, right)
    : method === "spearman" ? pearson(ranks(left), ranks(right)) : kendall(left, right);
  const count = left.length;
  const pValue = method === "kendall"
    ? 2 * (1 - normalCdf(Math.abs(coefficient) * Math.sqrt(
      9 * count * (count - 1) / (2 * (2 * count + 5)),
    )))
    : studentTwoSidedP(
      coefficient * Math.sqrt((count - 2) / Math.max(Number.EPSILON, 1 - coefficient ** 2)),
      count - 2,
    );
  return { coefficient, pValue };
};

const adjustPValues = (values: Array<number | null>): Array<number | null> => {
  const ordered = values.map((value, index) => ({ value, index }))
    .filter((item): item is { value: number; index: number } => item.value !== null)
    .sort((left, right) => left.value - right.value);
  const adjusted = Array(values.length).fill(null) as Array<number | null>;
  let previous = 1;
  for (let index = ordered.length - 1; index >= 0; index -= 1) {
    const corrected = Math.min(previous, ordered[index].value * ordered.length / (index + 1));
    adjusted[ordered[index].index] = Math.min(1, corrected);
    previous = corrected;
  }
  return adjusted;
};

export const calculateCorrelations = (
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest,
): CorrelationEstimate[] => {
  const selected = [request.outcome, ...request.predictors];
  const candidates = request.missingPolicy === "listwise"
    ? rows.filter((row) => selected.every((column) => finiteLaunchMonitorScalar(row[column]) !== null))
    : rows;
  const estimates = request.predictors.map((predictor) =>
    correlationForPredictor(candidates, request, predictor));
  const adjusted = adjustPValues(estimates.map((item) => item.pValue));
  return estimates.map((item, index) => ({ ...item, adjustedPValue: adjusted[index] }));
};

const correlationForPredictor = (
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest, predictor: string,
): CorrelationEstimate => {
  const pairs = rows.map((row) => [
    finiteLaunchMonitorScalar(row[request.outcome]), finiteLaunchMonitorScalar(row[predictor]),
  ]).filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null);
  if (pairs.length < request.minSamples) {
    return { predictor, coefficient: null, pValue: null, adjustedPValue: null,
      ciLower: null, ciUpper: null, sampleCount: pairs.length, method: request.correlationMethod };
  }
  const result = correlation(
    pairs.map((pair) => pair[0]), pairs.map((pair) => pair[1]), request.correlationMethod,
  );
  const [ciLower, ciUpper] = pearsonInterval(result.coefficient, pairs.length, request);
  return { predictor, ...result, adjustedPValue: null, ciLower, ciUpper,
    sampleCount: pairs.length, method: request.correlationMethod };
};

const pearsonInterval = (
  coefficient: number, count: number, request: LaunchMonitorAnalysisRequest,
): [number | null, number | null] => {
  if (request.correlationMethod !== "pearson" || count <= 3) return [null, null];
  const clipped = Math.max(-0.999999, Math.min(0.999999, coefficient));
  const transformed = Math.atanh(clipped);
  const margin = normalQuantile(0.5 + request.confidenceLevel / 2) / Math.sqrt(count - 3);
  return [Math.tanh(transformed - margin), Math.tanh(transformed + margin)];
};

const inverse = (matrix: number[][]): number[][] => {
  const size = matrix.length;
  const augmented = matrix.map((row, rowIndex) => [
    ...row, ...Array.from({ length: size }, (_, column) => rowIndex === column ? 1 : 0),
  ]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) {
      throw new RangeError("Regression design matrix is rank deficient");
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    augmented[column] = augmented[column].map((value) => value / divisor);
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      augmented[row] = augmented[row].map(
        (value, index) => value - factor * augmented[column][index],
      );
    }
  }
  return augmented.map((row) => row.slice(size));
};

const multiply = (left: number[][], right: number[][]): number[][] =>
  left.map((row) => right[0].map((_, column) =>
    row.reduce((sum, value, index) => sum + value * right[index][column], 0)));

const transpose = (matrix: number[][]): number[][] =>
  matrix[0].map((_, column) => matrix.map((row) => row[column]));

export const calculateRegression = (
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest,
): RegressionEstimate => {
  const complete = rows.map((row) => [request.outcome, ...request.predictors]
    .map((column) => finiteLaunchMonitorScalar(row[column])))
    .filter((values): values is number[] => values.every((value) => value !== null));
  const parameterCount = request.predictors.length + 1;
  if (complete.length < Math.max(request.minSamples, parameterCount + 2)) {
    throw new RangeError("Too few complete observations for regression");
  }
  const outcome = complete.map((values) => values[0]);
  const design = complete.map((values) => [1, ...values.slice(1)]);
  const inverseInformation = inverse(multiply(transpose(design), design));
  const beta = multiply(
    multiply(inverseInformation, transpose(design)), outcome.map((value) => [value]),
  ).map((row) => row[0]);
  const residuals = outcome.map((value, index) => value - design[index]
    .reduce((sum, item, betaIndex) => sum + item * beta[betaIndex], 0));
  return regressionResult({
    count: complete.length,
    parameterCount,
    outcome,
    design,
    beta,
    residuals,
    inverseInformation,
  }, request);
};

const regressionResult = (
  work: RegressionWork,
  request: LaunchMonitorAnalysisRequest,
): RegressionEstimate => {
  const residualSum = work.residuals.reduce((sum, value) => sum + value ** 2, 0);
  const totalSum = work.outcome.reduce(
    (sum, value) => sum + (value - mean(work.outcome)) ** 2, 0,
  );
  const rSquared = 1 - residualSum / totalSum;
  const degrees = work.count - work.parameterCount;
  const sigmaSquared = residualSum / degrees;
  const critical = studentQuantile(0.5 + request.confidenceLevel / 2, degrees);
  const names = ["intercept", ...request.predictors];
  const coefficients = Object.fromEntries(names.map((name, index) => {
    const standardError = Math.sqrt(
      Math.max(0, sigmaSquared * work.inverseInformation[index][index]),
    );
    const estimate = work.beta[index];
    const tStatistic = standardError === 0 ? Number.POSITIVE_INFINITY : estimate / standardError;
    return [name, { estimate, standardError, tStatistic,
      pValue: studentTwoSidedP(tStatistic, degrees),
      ciLower: estimate - critical * standardError,
      ciUpper: estimate + critical * standardError }];
  }));
  const leverage = work.design.map((row) => multiply([row], work.inverseInformation)[0]
    .reduce((sum, value, index) => sum + value * row[index], 0));
  const cooks = work.residuals.map((residual, index) =>
    (residual ** 2 / Math.max(Number.EPSILON, work.parameterCount * sigmaSquared)) *
    leverage[index] / Math.max(Number.EPSILON, (1 - leverage[index]) ** 2));
  return { sampleCount: work.count, rSquared,
    adjustedRSquared: 1 - (1 - rSquared) * (work.count - 1) / degrees, coefficients,
    residualDiagnostics: {
      rmse: Math.sqrt(residualSum / work.count), mae: mean(work.residuals.map(Math.abs)),
      residualMean: mean(work.residuals),
      residualStd: Math.sqrt(variance(work.residuals, work.parameterCount)),
      durbinWatson: residualSum === 0 ? null : work.residuals.slice(1)
        .reduce((sum, value, index) => sum + (value - work.residuals[index]) ** 2, 0) /
        residualSum,
      influentialCount: cooks.filter((value) => value > 4 / work.count).length,
    } };
};
