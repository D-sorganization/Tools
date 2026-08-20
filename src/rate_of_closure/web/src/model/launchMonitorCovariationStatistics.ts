export interface AssociationValues {
  pearsonR: number | null;
  spearmanR: number | null;
  slope: number | null;
  intercept: number | null;
  rSquared: number | null;
}

const mean = (values: number[]) =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

const ranks = (values: number[]): number[] => {
  const ordered = values.map((value, index) => ({ value, index }))
    .sort((left, right) => left.value - right.value);
  const result = Array(values.length).fill(0) as number[];
  let start = 0;
  while (start < ordered.length) {
    let end = start + 1;
    while (end < ordered.length && ordered[end].value === ordered[start].value) end += 1;
    const rank = (start + end + 1) / 2;
    for (let index = start; index < end; index += 1) result[ordered[index].index] = rank;
    start = end;
  }
  return result;
};

export const pearsonCorrelation = (left: number[], right: number[]): number | null => {
  if (left.length < 2 || left.length !== right.length) return null;
  const leftMean = mean(left);
  const rightMean = mean(right);
  let numerator = 0;
  let leftSquares = 0;
  let rightSquares = 0;
  left.forEach((value, index) => {
    const centeredLeft = value - leftMean;
    const centeredRight = right[index] - rightMean;
    numerator += centeredLeft * centeredRight;
    leftSquares += centeredLeft ** 2;
    rightSquares += centeredRight ** 2;
  });
  const denominator = Math.sqrt(leftSquares * rightSquares);
  return denominator > 0 ? numerator / denominator : null;
};

export const associationValues = (left: number[], right: number[]): AssociationValues => {
  const pearsonR = pearsonCorrelation(left, right);
  const spearmanR = pearsonCorrelation(ranks(left), ranks(right));
  const leftMean = mean(left);
  const rightMean = mean(right);
  const denominator = left.reduce((sum, value) => sum + (value - leftMean) ** 2, 0);
  const slope = denominator > 0
    ? left.reduce((sum, value, index) => sum + (value - leftMean) * (right[index] - rightMean), 0) / denominator
    : null;
  return {
    pearsonR, spearmanR, slope,
    intercept: slope === null ? null : rightMean - slope * leftMean,
    rSquared: pearsonR === null ? null : pearsonR ** 2,
  };
};

const erf = (value: number): number => {
  const sign = value < 0 ? -1 : 1;
  const absolute = Math.abs(value);
  const factor = 1 / (1 + 0.3275911 * absolute);
  const polynomial = (((((1.061405429 * factor - 1.453152027) * factor) + 1.421413741) * factor -
    0.284496736) * factor + 0.254829592) * factor;
  return sign * (1 - polynomial * Math.exp(-(absolute ** 2)));
};

const normalCdf = (value: number) => 0.5 * (1 + erf(value / Math.sqrt(2)));

export const normalQuantile = (probability: number): number => {
  let low = -8;
  let high = 8;
  for (let index = 0; index < 80; index += 1) {
    const middle = (low + high) / 2;
    if (normalCdf(middle) < probability) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
};

const boundedCorrelation = (value: number) => Math.max(-0.999999, Math.min(0.999999, value));

export const fisherInterval = (
  coefficient: number | null, count: number, confidence: number,
): [number | null, number | null] => {
  if (coefficient === null || count <= 3) return [null, null];
  const center = Math.atanh(boundedCorrelation(coefficient));
  const margin = normalQuantile(0.5 + confidence / 2) / Math.sqrt(count - 3);
  return [Math.tanh(center - margin), Math.tanh(center + margin)];
};

export const fisherZ = (coefficient: number) => Math.atanh(boundedCorrelation(coefficient));
