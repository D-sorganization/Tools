const rows = Array.from({ length: 100000 }, (_, i) => ({ a: 1, b: 2, c: 3, d: 4, e: 5 }));

function copyObjectKeys(row) {
  const newRow = {};
  for (const key of Object.keys(row)) {
    newRow[key] = row[key];
  }
  return newRow;
}

function copyForIn(row) {
  const newRow = {};
  for (const key in row) {
    if (Object.prototype.hasOwnProperty.call(row, key)) {
      newRow[key] = row[key];
    }
  }
  return newRow;
}

console.time('Object.keys');
for (let i = 0; i < 100000; i++) {
  copyObjectKeys(rows[i]);
}
console.timeEnd('Object.keys');

console.time('for...in');
for (let i = 0; i < 100000; i++) {
  copyForIn(rows[i]);
}
console.timeEnd('for...in');
