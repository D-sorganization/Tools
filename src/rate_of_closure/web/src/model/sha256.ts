/** Deterministic synchronous SHA-256 for bounded browser-side evidence. */

const rotate = (value: number, amount: number): number =>
  (value >>> amount) | (value << (32 - amount));

const firstPrimes = (count: number): readonly number[] => {
  const primes: number[] = [];
  for (let candidate = 2; primes.length < count; candidate += 1) {
    if (primes.every((prime) => candidate % prime !== 0)) primes.push(candidate);
  }
  return primes;
};

const schedule = (block: DataView): number[] => {
  const words = Array(64).fill(0) as number[];
  for (let index = 0; index < 16; index += 1) {
    words[index] = block.getUint32(index * 4, false);
  }
  for (let index = 16; index < 64; index += 1) {
    const first = rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^
      (words[index - 15] >>> 3);
    const second = rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^
      (words[index - 2] >>> 10);
    words[index] = (words[index - 16] + first + words[index - 7] + second) | 0;
  }
  return words;
};

const compress = (
  state: number[],
  constants: readonly number[],
  words: readonly number[],
): void => {
  let [a, b, c, d, e, f, g, h] = state;
  for (let index = 0; index < 64; index += 1) {
    const sigmaOne = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
    const choice = (e & f) ^ (~e & g);
    const first = (h + sigmaOne + choice + constants[index] + words[index]) | 0;
    const sigmaZero = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
    const majority = (a & b) ^ (a & c) ^ (b & c);
    const second = (sigmaZero + majority) | 0;
    [h, g, f, e, d, c, b, a] = [g, f, e, (d + first) | 0, c, b, a,
      (first + second) | 0];
  }
  [a, b, c, d, e, f, g, h].forEach((value, index) => {
    state[index] = (state[index] + value) | 0;
  });
};

const paddedBytes = (input: string): Uint8Array => {
  const bytes = new TextEncoder().encode(input);
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const bitLength = BigInt(bytes.length) * 8n;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLength - 8, Number(bitLength >> 32n), false);
  view.setUint32(paddedLength - 4, Number(bitLength & 0xffff_ffffn), false);
  return padded;
};

/** Return the lowercase SHA-256 digest of one UTF-8 string. */
export const sha256Text = (input: string): string => {
  const primes = firstPrimes(64);
  const state = primes.slice(0, 8)
    .map((prime) => (Math.sqrt(prime) % 1) * 2 ** 32 | 0);
  const constants = primes
    .map((prime) => (Math.cbrt(prime) % 1) * 2 ** 32 | 0);
  const bytes = paddedBytes(input);
  for (let offset = 0; offset < bytes.length; offset += 64) {
    compress(state, constants, schedule(new DataView(bytes.buffer, offset, 64)));
  }
  return state.map((value) =>
    (value >>> 0).toString(16).padStart(8, "0")).join("");
};
