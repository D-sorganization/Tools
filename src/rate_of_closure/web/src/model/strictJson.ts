/** Parse strict JSON while rejecting duplicate fields at every object depth. */
export function parseUniqueJson(text: string): unknown {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`invalid profile JSON: ${String(error)}`);
  }
  assertUniqueKeys(text);
  return value;
}

function assertUniqueKeys(text: string): void {
  let index = 0;
  const whitespace = () => { while (/\s/.test(text[index] ?? "")) index += 1; };
  const scanString = (): string => {
    const start = index++;
    while (index < text.length) {
      if (text[index] === "\\") index += 2;
      else if (text[index++] === '"') return JSON.parse(text.slice(start, index));
    }
    throw new Error("invalid profile JSON string");
  };
  const scanValue = (): void => {
    whitespace();
    if (text[index] === "{") scanObject();
    else if (text[index] === "[") scanArray();
    else if (text[index] === '"') void scanString();
    else while (index < text.length && !/[,}\]]/.test(text[index])) index += 1;
  };
  const scanArray = (): void => {
    index += 1; whitespace();
    while (text[index] !== "]") {
      scanValue(); whitespace();
      if (text[index] === ",") { index += 1; whitespace(); }
    }
    index += 1;
  };
  const scanObject = (): void => {
    const keys = new Set<string>();
    index += 1; whitespace();
    while (text[index] !== "}") {
      const key = scanString();
      if (keys.has(key)) throw new Error(`duplicate JSON field: ${key}`);
      keys.add(key); whitespace(); index += 1; scanValue(); whitespace();
      if (text[index] === ",") { index += 1; whitespace(); }
    }
    index += 1;
  };
  scanValue();
}
