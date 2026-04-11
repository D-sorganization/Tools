const n_rows = 100000;
const signals = ["A", "B", "C", "D", "E"];
const data = Array.from({ length: n_rows }, () => {
  return {
    A: Math.random(),
    B: Math.random(),
    C: Math.random(),
    D: Math.random(),
    E: Math.random(),
  };
});

function computeCorrelationOld(data, signals) {
  const n = signals.length;
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));

  const columns = signals.map((sig) =>
    data.map((row) => (typeof row[sig] === "number" ? row[sig] : NaN)),
  );

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const valid = columns[i]
        .map((v, k) => ({ x: v, y: columns[j][k] }))
        .filter(({ x, y }) => !isNaN(x) && !isNaN(y));

      const x = valid.map((d) => d.x);
      const y = valid.map((d) => d.y);

      const len = x.length;
      if (len < 2) continue;
      const meanX = x.reduce((a, b) => a + b, 0) / len;
      const meanY = y.reduce((a, b) => a + b, 0) / len;
      let num = 0;
      let denX = 0;
      let denY = 0;
      for (let k = 0; k < len; k++) {
        const dx = x[k] - meanX;
        const dy = y[k] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }
      const den = Math.sqrt(denX * denY);
      const r = den === 0 ? 0 : num / den;

      matrix[i][j] = r;
      matrix[j][i] = r;
    }
  }
  return matrix;
}

function computeCorrelationNew(data, signals) {
  const n = signals.length;
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));

  const columns = signals.map((sig) =>
    data.map((row) => (typeof row[sig] === "number" ? row[sig] : NaN)),
  );

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const colI = columns[i];
      const colJ = columns[j];
      const len = colI.length;

      let sumX = 0,
        sumY = 0,
        count = 0;
      for (let k = 0; k < len; k++) {
        const vx = colI[k];
        const vy = colJ[k];
        if (!Number.isNaN(vx) && !Number.isNaN(vy)) {
          sumX += vx;
          sumY += vy;
          count++;
        }
      }

      if (count < 2) {
        matrix[i][j] = NaN;
        matrix[j][i] = NaN;
        continue;
      }

      const meanX = sumX / count;
      const meanY = sumY / count;

      let num = 0,
        denX = 0,
        denY = 0;
      for (let k = 0; k < len; k++) {
        const vx = colI[k];
        const vy = colJ[k];
        if (!Number.isNaN(vx) && !Number.isNaN(vy)) {
          const dx = vx - meanX;
          const dy = vy - meanY;
          num += dx * dy;
          denX += dx * dx;
          denY += dy * dy;
        }
      }

      const den = Math.sqrt(denX * denY);
      const r = den === 0 ? 0 : num / den;

      matrix[i][j] = r;
      matrix[j][i] = r;
    }
  }

  return matrix;
}

console.time("old");
for (let i = 0; i < 10; i++) computeCorrelationOld(data, signals);
console.timeEnd("old");

console.time("new");
for (let i = 0; i < 10; i++) computeCorrelationNew(data, signals);
console.timeEnd("new");
