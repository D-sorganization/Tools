const values = new Array(1000000).fill(0).map(() => Math.random() * 100);

// Original
function zScoreFilterOriginal(values, threshold) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(
    values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length,
  );

  if (std === 0) return values;

  return values.map((v) => {
    const zScore = Math.abs((v - mean) / std);
    return zScore > threshold ? mean : v;
  });
}

// Optimized
function zScoreFilterOptimized(values, threshold) {
  const len = values.length;
  if (len === 0) return [];

  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += values[i];
  }
  const mean = sum / len;

  let varianceSum = 0;
  for (let i = 0; i < len; i++) {
    varianceSum += (values[i] - mean) ** 2;
  }
  const std = Math.sqrt(varianceSum / len);

  if (std === 0) return values;

  const result = new Array(len);
  for (let i = 0; i < len; i++) {
    const v = values[i];
    const zScore = Math.abs((v - mean) / std);
    result[i] = zScore > threshold ? mean : v;
  }
  return result;
}

console.time("Original");
for (let i = 0; i < 10; i++) zScoreFilterOriginal(values, 3.0);
console.timeEnd("Original");

console.time("Optimized");
for (let i = 0; i < 10; i++) zScoreFilterOptimized(values, 3.0);
console.timeEnd("Optimized");
