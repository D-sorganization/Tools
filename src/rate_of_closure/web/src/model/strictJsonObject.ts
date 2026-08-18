/** JSON parser that rejects duplicate object keys before standard parsing. */

class JsonDuplicateKeyScanner {
  private index = 0;

  constructor(private readonly text: string) {}

  scan(): void {
    this.skipWhitespace();
    this.scanValue();
    this.skipWhitespace();
    if (this.index !== this.text.length) {
      throw new Error("JSON contains trailing content");
    }
  }

  private scanValue(): void {
    this.skipWhitespace();
    const token = this.text[this.index];
    if (token === "{") {
      this.scanObject();
      return;
    }
    if (token === "[") {
      this.scanArray();
      return;
    }
    if (token === '"') {
      this.scanString();
      return;
    }
    this.scanPrimitive();
  }

  private scanObject(): void {
    this.index += 1;
    this.skipWhitespace();
    const keys = new Set<string>();
    if (this.consume("}")) return;
    while (true) {
      this.skipWhitespace();
      const rawKey = this.scanString();
      const key = JSON.parse(rawKey) as string;
      if (keys.has(key))
        throw new Error(`JSON contains duplicate field ${key}`);
      keys.add(key);
      this.skipWhitespace();
      this.expect(":");
      this.scanValue();
      this.skipWhitespace();
      if (this.consume("}")) return;
      this.expect(",");
    }
  }

  private scanArray(): void {
    this.index += 1;
    this.skipWhitespace();
    if (this.consume("]")) return;
    while (true) {
      this.scanValue();
      this.skipWhitespace();
      if (this.consume("]")) return;
      this.expect(",");
    }
  }

  private scanString(): string {
    const start = this.index;
    this.expect('"');
    while (this.index < this.text.length) {
      const token = this.text[this.index];
      this.index += 1;
      if (token === '"') return this.text.slice(start, this.index);
      if (token === "\\") {
        this.index += 1;
      }
    }
    throw new Error("JSON contains an unterminated string");
  }

  private scanPrimitive(): void {
    const start = this.index;
    while (this.index < this.text.length) {
      const token = this.text[this.index];
      if (
        token === "," ||
        token === "]" ||
        token === "}" ||
        /\s/u.test(token)
      ) {
        break;
      }
      this.index += 1;
    }
    if (this.index === start) throw new Error("JSON value is missing");
  }

  private skipWhitespace(): void {
    while (/\s/u.test(this.text[this.index] ?? "")) this.index += 1;
  }

  private consume(expected: string): boolean {
    if (this.text[this.index] !== expected) return false;
    this.index += 1;
    return true;
  }

  private expect(expected: string): void {
    if (!this.consume(expected)) throw new Error(`JSON expected ${expected}`);
  }
}

/** Parse one JSON document and reject ambiguous duplicate object fields. */
export function parseStrictJson(text: string): unknown {
  if (typeof text !== "string") throw new Error("JSON input must be text");
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "invalid JSON";
    throw new Error(`binding must contain valid JSON: ${detail}`);
  }
  new JsonDuplicateKeyScanner(text).scan();
  return parsed;
}
