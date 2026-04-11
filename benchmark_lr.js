const x = new Array(1000000).fill(0).map(() => Math.random() * 100);
const y = new Array(1000000).fill(0).map(() => Math.random() * 100);

function linearRegressionOriginal(x, y) {
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  const meanY = sumY / n;
  const ssTotal = y.reduce((sum, yi) => sum + (yi - meanY) ** 2, 0);
  const ssResidual = y.reduce(
    (sum, yi, i) => sum + (yi - (slope * x[i] + intercept)) ** 2,
    0,
  );
  const rSquared = 1 - ssResidual / ssTotal;

  return { slope, intercept, rSquared };
}

function linearRegressionOptimized(x, y) {
  const n = x.length;
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumXX = 0;

  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = y[i];
    sumX += xi;
    sumY += yi;
    sumXY += xi * yi;
    sumXX += xi * xi;
  }

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  const meanY = sumY / n;
  let ssTotal = 0,
    ssResidual = 0;

  for (let i = 0; i < n; i++) {
    const yi = y[i];
    ssTotal += (yi - meanY) ** 2;
    ssResidual += (yi - (slope * x[i] + intercept)) ** 2;
  }
  const rSquared = 1 - ssResidual / ssTotal;

  return { slope, intercept, rSquared };
}

console.time("LR Original");
for (let i = 0; i < 10; i++) linearRegressionOriginal(x, y);
console.timeEnd("LR Original");

console.time("LR Optimized");
for (let i = 0; i < 10; i++) linearRegressionOptimized(x, y);
console.timeEnd("LR Optimized");
