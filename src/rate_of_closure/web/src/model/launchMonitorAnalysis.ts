/** UI-neutral, provenance-preserving launch-monitor statistical analysis. */

export const LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION = "1.0.0" as const;

export type LaunchMonitorScalar = string | number | boolean | null;
export type LaunchMonitorRow = Record<string, LaunchMonitorScalar>;
export type AnalysisMode = "correlation" | "regression" | "comprehensive";
export type CorrelationMethod = "pearson" | "spearman" | "kendall";
export type MissingPolicy = "pairwise" | "listwise" | "fail";

export interface LaunchMonitorAnalysisRequest {
  outcome: string;
  predictors: string[];
  analysisMode: AnalysisMode;
  correlationMethod: CorrelationMethod;
  missingPolicy: MissingPolicy;
  groupBy?: string;
  confidenceLevel: number;
  minSamples: number;
  allowAggregate?: boolean;
}

export interface CorrelationEstimate {
  predictor: string;
  coefficient: number | null;
  pValue: number | null;
  adjustedPValue: number | null;
  ciLower: number | null;
  ciUpper: number | null;
  sampleCount: number;
  method: CorrelationMethod;
}

export interface CoefficientEstimate {
  estimate: number;
  standardError: number;
  tStatistic: number;
  pValue: number;
  ciLower: number;
  ciUpper: number;
}

export interface RegressionEstimate {
  sampleCount: number;
  rSquared: number;
  adjustedRSquared: number;
  coefficients: Record<string, CoefficientEstimate>;
  residualDiagnostics: {
    rmse: number;
    mae: number;
    residualMean: number;
    residualStd: number;
    durbinWatson: number | null;
    influentialCount: number;
  };
}

export interface GroupAnalysis {
  groupValue: string;
  rowCount: number;
  correlations: CorrelationEstimate[];
  regression: RegressionEstimate | null;
  warnings: string[];
}

export interface LaunchMonitorAnalysisResult {
  contractVersion: typeof LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION;
  request: LaunchMonitorAnalysisRequest;
  dataset: {
    rowCount: number;
    completeRowCount: number;
    selectedColumns: string[];
    monitorVendors: string[];
    sessionIds: string[];
    observationKinds: string[];
    fingerprintSha256: string;
  };
  correlations: CorrelationEstimate[];
  regression: RegressionEstimate | null;
  groups: GroupAnalysis[];
  warnings: string[];
}

const finite = (value: LaunchMonitorScalar | undefined): number | null => {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const converted = typeof value === "number" ? value : Number(value);
  return Number.isFinite(converted) ? converted : null;
};

export function numericLaunchMonitorColumns(rows: LaunchMonitorRow[]): string[] {
  const columns = new Set(rows.flatMap((row) => Object.keys(row)));
  return [...columns].filter((column) =>
    rows.reduce((count, row) => count + (finite(row[column]) === null ? 0 : 1), 0) >= 3,
  ).sort();
}

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

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
  if (value < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  let x = 0.9999999999998099;
  const shifted = value - 1;
  coefficients.forEach((coefficient, index) => { x += coefficient / (shifted + index + 1); });
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
    aa = -(a + iteration) * (qab + iteration) * x /
      ((a + twice) * (qap + twice));
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
  return (concordant - discordant) /
    Math.sqrt((concordant + discordant + leftTies) *
      (concordant + discordant + rightTies));
};

const correlation = (left: number[], right: number[], method: CorrelationMethod) => {
  const coefficient = method === "pearson" ? pearson(left, right)
    : method === "spearman" ? pearson(ranks(left), ranks(right)) : kendall(left, right);
  const count = left.length;
  const pValue = method === "kendall"
    ? 2 * (1 - normalCdf(Math.abs(coefficient) * Math.sqrt(9 * count * (count - 1) /
      (2 * (2 * count + 5)))))
    : studentTwoSidedP(coefficient * Math.sqrt((count - 2) /
      Math.max(Number.EPSILON, 1 - coefficient ** 2)), count - 2);
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

const inverse = (matrix: number[][]): number[][] => {
  const size = matrix.length;
  const augmented = matrix.map((row, rowIndex) => [
    ...row,
    ...Array.from({ length: size }, (_, column) => rowIndex === column ? 1 : 0),
  ]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) throw new RangeError("Regression design matrix is rank deficient");
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    augmented[column] = augmented[column].map((value) => value / divisor);
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      augmented[row] = augmented[row].map((value, index) => value - factor * augmented[column][index]);
    }
  }
  return augmented.map((row) => row.slice(size));
};

const multiply = (left: number[][], right: number[][]): number[][] =>
  left.map((row) => right[0].map((_, column) =>
    row.reduce((sum, value, index) => sum + value * right[index][column], 0)));

const transpose = (matrix: number[][]): number[][] =>
  matrix[0].map((_, column) => matrix.map((row) => row[column]));

const ols = (rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest): RegressionEstimate => {
  const complete = rows.map((row) => [request.outcome, ...request.predictors]
    .map((column) => finite(row[column])))
    .filter((values): values is number[] => values.every((value) => value !== null));
  const parameterCount = request.predictors.length + 1;
  if (complete.length < Math.max(request.minSamples, parameterCount + 2)) {
    throw new RangeError("Too few complete observations for regression");
  }
  const y = complete.map((values) => values[0]);
  const design = complete.map((values) => [1, ...values.slice(1)]);
  const xt = transpose(design);
  const xtxInverse = inverse(multiply(xt, design));
  const beta = multiply(multiply(xtxInverse, xt), y.map((value) => [value])).map((row) => row[0]);
  const fitted = design.map((row) => row.reduce((sum, value, index) => sum + value * beta[index], 0));
  const residuals = y.map((value, index) => value - fitted[index]);
  const residualSum = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const yMean = mean(y);
  const totalSum = y.reduce((sum, value) => sum + (value - yMean) ** 2, 0);
  const rSquared = 1 - residualSum / totalSum;
  const degrees = complete.length - parameterCount;
  const sigmaSquared = residualSum / degrees;
  const critical = studentQuantile(0.5 + request.confidenceLevel / 2, degrees);
  const names = ["intercept", ...request.predictors];
  const coefficients = Object.fromEntries(names.map((name, index) => {
    const standardError = Math.sqrt(Math.max(0, sigmaSquared * xtxInverse[index][index]));
    const tStatistic = standardError === 0 ? Number.POSITIVE_INFINITY : beta[index] / standardError;
    return [name, {
      estimate: beta[index], standardError, tStatistic,
      pValue: studentTwoSidedP(tStatistic, degrees),
      ciLower: beta[index] - critical * standardError,
      ciUpper: beta[index] + critical * standardError,
    }];
  }));
  const leverage = design.map((row) => {
    const projected = multiply([row], xtxInverse)[0];
    return projected.reduce((sum, value, index) => sum + value * row[index], 0);
  });
  const cooks = residuals.map((residual, index) =>
    (residual ** 2 / Math.max(Number.EPSILON, parameterCount * sigmaSquared)) *
    leverage[index] / Math.max(Number.EPSILON, (1 - leverage[index]) ** 2));
  return {
    sampleCount: complete.length,
    rSquared,
    adjustedRSquared: 1 - (1 - rSquared) * (complete.length - 1) / degrees,
    coefficients,
    residualDiagnostics: {
      rmse: Math.sqrt(residualSum / complete.length),
      mae: mean(residuals.map(Math.abs)),
      residualMean: mean(residuals),
      residualStd: Math.sqrt(variance(residuals, parameterCount)),
      durbinWatson: residualSum === 0 ? null : residuals.slice(1).reduce((sum, value, index) =>
        sum + (value - residuals[index]) ** 2, 0) / residualSum,
      influentialCount: cooks.filter((value) => value > 4 / complete.length).length,
    },
  };
};

// Deterministic SHA-256 keeps browser exports comparable without a server round-trip.
const sha256 = (input: string): string => {
  const rightRotate = (value: number, amount: number) => (value >>> amount) | (value << (32 - amount));
  const powers: number[] = [];
  for (let candidate = 2; powers.length < 64; candidate += 1) {
    if (powers.every((prime) => candidate % prime !== 0)) powers.push(candidate);
  }
  const initial = powers.slice(0, 8).map((prime) => (Math.sqrt(prime) % 1) * 2 ** 32 | 0);
  const constants = powers.map((prime) => (Math.cbrt(prime) % 1) * 2 ** 32 | 0);
  const bytes = new TextEncoder().encode(input);
  const bitLength = bytes.length * 8;
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  new DataView(padded.buffer).setUint32(paddedLength - 4, bitLength, false);
  const hash = [...initial];
  for (let offset = 0; offset < padded.length; offset += 64) {
    const words = Array(64).fill(0) as number[];
    const view = new DataView(padded.buffer, offset, 64);
    for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const first = rightRotate(words[index - 15], 7) ^ rightRotate(words[index - 15], 18) ^ (words[index - 15] >>> 3);
      const second = rightRotate(words[index - 2], 17) ^ rightRotate(words[index - 2], 19) ^ (words[index - 2] >>> 10);
      words[index] = (words[index - 16] + first + words[index - 7] + second) | 0;
    }
    let [a, b, c, d, e, f, g, h] = hash;
    for (let index = 0; index < 64; index += 1) {
      const sigmaOne = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
      const choice = (e & f) ^ (~e & g);
      const first = (h + sigmaOne + choice + constants[index] + words[index]) | 0;
      const sigmaZero = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const second = (sigmaZero + majority) | 0;
      [h, g, f, e, d, c, b, a] = [g, f, e, (d + first) | 0, c, b, a, (first + second) | 0];
    }
    [a, b, c, d, e, f, g, h].forEach((value, index) => { hash[index] = (hash[index] + value) | 0; });
  }
  return hash.map((value) => (value >>> 0).toString(16).padStart(8, "0")).join("");
};

export const sha256Text = (input: string): string => sha256(input);

const uniqueStrings = (rows: LaunchMonitorRow[], column: string): string[] =>
  [...new Set(rows.map((row) => row[column]).filter((value) => value !== null && value !== undefined)
    .map(String).filter((value) => value.trim()))].sort();

const canonicalFingerprint = (rows: LaunchMonitorRow[], selected: string[]): string => {
  const identity = ["shot_id", "session_id", "source_row", "monitor_vendor"]
    .filter((column) => rows.some((row) => column in row) && !selected.includes(column));
  const columns = [...identity, ...selected];
  return sha256(JSON.stringify(rows.map((row) => Object.fromEntries(columns.map((column) =>
    [column, row[column] ?? null])))));
};

const validate = (rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest): void => {
  if (!rows.length) throw new RangeError("At least one observation is required");
  if (!request.outcome || !request.predictors.length) throw new RangeError("Select an outcome and predictors");
  if (request.predictors.includes(request.outcome)) throw new RangeError("outcome cannot also be a predictor");
  if (new Set(request.predictors).size !== request.predictors.length) throw new RangeError("predictors must be unique");
  if (!(request.confidenceLevel > 0.5 && request.confidenceLevel < 1)) throw new RangeError("confidenceLevel must be between 0.5 and 1");
  if (request.minSamples < 3) throw new RangeError("minSamples must be at least 3");
  const selected = [request.outcome, ...request.predictors];
  const missing = [...selected, ...(request.groupBy ? [request.groupBy] : [])]
    .filter((column) => !rows.some((row) => column in row));
  if (missing.length) throw new RangeError(`Columns not present: ${[...new Set(missing)].join(", ")}`);
  const constants = selected.filter((column) => new Set(rows.map((row) => finite(row[column]))
    .filter((value) => value !== null)).size < 2);
  if (constants.length) throw new RangeError(`Constant variables cannot be analyzed: ${constants.join(", ")}`);
  if (request.missingPolicy === "fail" && rows.some((row) => selected.some((column) => finite(row[column]) === null))) {
    throw new RangeError("Selected variables contain missing or non-numeric values");
  }
};

export function analyzeLaunchMonitorData(
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest,
): LaunchMonitorAnalysisResult {
  validate(rows, request);
  const selected = [request.outcome, ...request.predictors];
  const vendors = uniqueStrings(rows, "monitor_vendor");
  if (selected.some((column) => column.startsWith("source::")) && vendors.length > 1) {
    throw new RangeError("source fields cannot be pooled across multiple monitors");
  }
  const observationKinds = uniqueStrings(rows, "observation_kind");
  if (!observationKinds.length) observationKinds.push("shot");
  const aggregate = observationKinds.some((kind) => kind.toLowerCase() !== "shot");
  if (aggregate && request.analysisMode !== "correlation") {
    throw new RangeError("Aggregate observations cannot enter regression");
  }
  if (aggregate && !request.allowAggregate) throw new RangeError("Aggregate observations require allowAggregate=true");
  const warnings: string[] = [];
  if (aggregate) warnings.push("Aggregate correlations are descriptive only and may exhibit ecological bias.");
  if (request.correlationMethod !== "pearson" && request.analysisMode !== "regression") {
    warnings.push("Analytical confidence intervals are only reported for Pearson correlation.");
  }
  const listwiseRows = request.missingPolicy === "listwise"
    ? rows.filter((row) => selected.every((column) => finite(row[column]) !== null)) : rows;
  const correlations: CorrelationEstimate[] = request.analysisMode === "regression" ? [] : request.predictors.map((predictor): CorrelationEstimate => {
    const pairs = listwiseRows.map((row) => [finite(row[request.outcome]), finite(row[predictor])])
      .filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null);
    if (pairs.length < request.minSamples) return {
      predictor, coefficient: null, pValue: null, adjustedPValue: null,
      ciLower: null, ciUpper: null, sampleCount: pairs.length, method: request.correlationMethod,
    };
    const result = correlation(pairs.map((pair) => pair[0]), pairs.map((pair) => pair[1]), request.correlationMethod);
    let ciLower: number | null = null;
    let ciUpper: number | null = null;
    if (request.correlationMethod === "pearson" && pairs.length > 3) {
      const clipped = Math.max(-0.999999, Math.min(0.999999, result.coefficient));
      const transformed = Math.atanh(clipped);
      const margin = normalQuantile(0.5 + request.confidenceLevel / 2) / Math.sqrt(pairs.length - 3);
      ciLower = Math.tanh(transformed - margin);
      ciUpper = Math.tanh(transformed + margin);
    }
    return { predictor, coefficient: result.coefficient, pValue: result.pValue,
      adjustedPValue: null, ciLower, ciUpper, sampleCount: pairs.length,
      method: request.correlationMethod };
  });
  const adjusted = adjustPValues(correlations.map((item) => item.pValue));
  correlations.forEach((item, index) => { item.adjustedPValue = adjusted[index]; });
  const regression = request.analysisMode === "correlation" ? null : ols(rows, request);
  const groups: GroupAnalysis[] = [];
  if (request.groupBy) {
    const values = uniqueStrings(rows, request.groupBy);
    values.forEach((value) => {
      const groupRows = rows.filter((row) => String(row[request.groupBy as string]) === value);
      try {
        const result = analyzeLaunchMonitorData(groupRows, { ...request, groupBy: undefined });
        groups.push({ groupValue: value, rowCount: groupRows.length,
          correlations: result.correlations, regression: result.regression, warnings: result.warnings });
      } catch (error) {
        groups.push({ groupValue: value, rowCount: groupRows.length, correlations: [], regression: null,
          warnings: [error instanceof Error ? error.message : String(error)] });
      }
    });
  }
  return {
    contractVersion: LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION,
    request: { ...request, predictors: [...request.predictors] },
    dataset: {
      rowCount: rows.length,
      completeRowCount: rows.filter((row) => selected.every((column) => finite(row[column]) !== null)).length,
      selectedColumns: selected,
      monitorVendors: vendors,
      sessionIds: uniqueStrings(rows, "session_id"),
      observationKinds,
      fingerprintSha256: canonicalFingerprint(rows, selected),
    },
    correlations, regression, groups, warnings,
  };
}

const coerceCell = (value: string): LaunchMonitorScalar => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const numeric = Number(trimmed);
  return Number.isFinite(numeric) ? numeric : trimmed;
};

const parseCsvRows = (text: string): string[][] => {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') { cell += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) {
      row.push(cell); cell = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell); cell = "";
      if (row.some((value) => value.length)) rows.push(row);
      row = [];
    } else cell += character;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  if (quoted) throw new RangeError("CSV contains an unterminated quoted field");
  return rows;
};

export function parseLaunchMonitorFile(fileName: string, text: string): LaunchMonitorRow[] {
  if (fileName.toLowerCase().endsWith(".json")) {
    const parsed: unknown = JSON.parse(text);
    if (!Array.isArray(parsed) || parsed.some((row) => !row || typeof row !== "object" || Array.isArray(row))) {
      throw new RangeError("JSON launch-monitor data must be an array of record objects");
    }
    return parsed as LaunchMonitorRow[];
  }
  const parsed = parseCsvRows(text);
  if (parsed.length < 2) throw new RangeError("CSV must contain a header and at least one row");
  const headers = parsed[0].map((header) => header.trim());
  if (headers.some((header) => !header) || new Set(headers).size !== headers.length) {
    throw new RangeError("CSV headers must be non-empty and unique");
  }
  return parsed.slice(1).map((values) => Object.fromEntries(headers.map((header, index) =>
    [header, coerceCell(values[index] ?? "")]))) as LaunchMonitorRow[];
}
