/** Deterministic provenance fingerprints for browser analysis exports. */

import type { LaunchMonitorRow } from "./launchMonitorAnalysisTypes";

const sha256 = (input: string): string => {
  const rotate = (value: number, amount: number) =>
    (value >>> amount) | (value << (32 - amount));
  const primes: number[] = [];
  for (let candidate = 2; primes.length < 64; candidate += 1) {
    if (primes.every((prime) => candidate % prime !== 0)) primes.push(candidate);
  }
  const hash = primes.slice(0, 8).map((prime) => (Math.sqrt(prime) % 1) * 2 ** 32 | 0);
  const constants = primes.map((prime) => (Math.cbrt(prime) % 1) * 2 ** 32 | 0);
  const bytes = new TextEncoder().encode(input);
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  new DataView(padded.buffer).setUint32(paddedLength - 4, bytes.length * 8, false);
  for (let offset = 0; offset < padded.length; offset += 64) {
    const words = Array(64).fill(0) as number[];
    const view = new DataView(padded.buffer, offset, 64);
    for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const first = rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^
        (words[index - 15] >>> 3);
      const second = rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^
        (words[index - 2] >>> 10);
      words[index] = (words[index - 16] + first + words[index - 7] + second) | 0;
    }
    let [a, b, c, d, e, f, g, h] = hash;
    for (let index = 0; index < 64; index += 1) {
      const sigmaOne = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const choice = (e & f) ^ (~e & g);
      const first = (h + sigmaOne + choice + constants[index] + words[index]) | 0;
      const sigmaZero = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const second = (sigmaZero + majority) | 0;
      [h, g, f, e, d, c, b, a] = [g, f, e, (d + first) | 0, c, b, a, (first + second) | 0];
    }
    [a, b, c, d, e, f, g, h].forEach((value, index) => {
      hash[index] = (hash[index] + value) | 0;
    });
  }
  return hash.map((value) => (value >>> 0).toString(16).padStart(8, "0")).join("");
};

export const sha256Text = (input: string): string => sha256(input);

export const uniqueStrings = (rows: LaunchMonitorRow[], column: string): string[] =>
  [...new Set(rows.map((row) => row[column])
    .filter((value) => value !== null && value !== undefined)
    .map(String).filter((value) => value.trim()))].sort();

export const canonicalFingerprint = (rows: LaunchMonitorRow[], selected: string[]): string => {
  const identity = ["shot_id", "session_id", "source_row", "monitor_vendor"]
    .filter((column) => rows.some((row) => column in row) && !selected.includes(column));
  const columns = [...identity, ...selected];
  return sha256(JSON.stringify(rows.map((row) => Object.fromEntries(
    columns.map((column) => [column, row[column] ?? null]),
  ))));
};
