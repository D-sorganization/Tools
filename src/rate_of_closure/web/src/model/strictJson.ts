/** Parse strict JSON while rejecting duplicate fields at every object depth. */
export function parseUniqueJson(
  text: string, documentName = "profile JSON", maximumDepth = 32,
): unknown {
  if (typeof text !== "string") throw new TypeError(`${documentName} source must be text`);
  if (typeof documentName !== "string" || documentName.trim() === "") {
    throw new TypeError("documentName must be nonempty text");
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`invalid ${documentName}: ${String(error)}`);
  }
  assertUniqueKeys(text, documentName, maximumDepth);
  return value;
}

function assertUniqueKeys(text: string, documentName: string, maximumDepth: number): void {
  let index = 0;
  const whitespace = () => { while (/\s/.test(text[index] ?? "")) index += 1; };
  const scanString = (): string => {
    const start = index++;
    while (index < text.length) {
      if (text[index] === "\\") index += 2;
      else if (text[index++] === '"') return JSON.parse(text.slice(start, index));
    }
    throw new Error(`invalid ${documentName} string`);
  };
  const scanValue = (depth: number): void => {
    if (depth > maximumDepth) throw new RangeError(`${documentName} exceeds the JSON depth limit`);
    whitespace();
    if (text[index] === "{") scanObject(depth);
    else if (text[index] === "[") scanArray(depth);
    else if (text[index] === '"') void scanString();
    else while (index < text.length && !/[,}\]]/.test(text[index])) index += 1;
  };
  const scanArray = (depth: number): void => {
    index += 1; whitespace();
    while (text[index] !== "]") {
      scanValue(depth + 1); whitespace();
      if (text[index] === ",") { index += 1; whitespace(); }
    }
    index += 1;
  };
  const scanObject = (depth: number): void => {
    const keys = new Set<string>();
    index += 1; whitespace();
    while (text[index] !== "}") {
      const key = scanString();
      if (keys.has(key)) throw new Error(`duplicate JSON field: ${key}`);
      keys.add(key); whitespace(); index += 1; scanValue(depth + 1); whitespace();
      if (text[index] === ",") { index += 1; whitespace(); }
    }
    index += 1;
  };
  scanValue(0);
}
