const n = 100000;
const xData = new Array(n);
const yData = new Array(n);
for(let i=0; i<n; i++) {
  xData[i] = i;
  yData[i] = Math.random() > 0.5 ? Math.exp(i/10000) : -1;
}

console.time("Original Exponential");
for (let j = 0; j < 100; j++) {
  const lnY = yData.filter((y) => y > 0).map((y) => Math.log(y));
  const xFiltered = xData.filter((_, i) => yData[i] > 0);
}
console.timeEnd("Original Exponential");

console.time("Optimized Exponential");
for (let j = 0; j < 100; j++) {
  const len = xData.length;
  const lnY = new Array(len);
  const xFiltered = new Array(len);
  let count = 0;
  for (let i = 0; i < len; i++) {
    const y = yData[i];
    if (y > 0) {
      lnY[count] = Math.log(y);
      xFiltered[count] = xData[i];
      count++;
    }
  }
  lnY.length = count;
  xFiltered.length = count;
}
console.timeEnd("Optimized Exponential");

console.time("Original Power");
for (let j = 0; j < 100; j++) {
  const validPower = xData.map((x, i) => ({ x, y: yData[i] }))
    .filter(({ x, y }) => x > 0 && y > 0);
  const lnX = validPower.map((d) => Math.log(d.x));
  const lnY = validPower.map((d) => Math.log(d.y));
}
console.timeEnd("Original Power");

console.time("Optimized Power");
for (let j = 0; j < 100; j++) {
  const len = xData.length;
  const lnX = new Array(len);
  const lnY = new Array(len);
  let count = 0;
  for (let i = 0; i < len; i++) {
    const x = xData[i];
    const y = yData[i];
    if (x > 0 && y > 0) {
      lnX[count] = Math.log(x);
      lnY[count] = Math.log(y);
      count++;
    }
  }
  lnX.length = count;
  lnY.length = count;
}
console.timeEnd("Optimized Power");
