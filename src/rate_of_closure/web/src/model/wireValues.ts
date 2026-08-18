/** Strict JSON-wire readers. They never coerce strings or Booleans. */
export const wireRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};

export const wireArray = (value: unknown, label: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value;
};

export const wireFiniteNumber = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
};

export const wireInteger = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${label} must be an integer`);
  }
  return value;
};

export const wireString = (value: unknown, label: string): string => {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  return value;
};

const isControlFree = (value: string): boolean =>
  [...value].every((character) => {
    const codePoint = character.codePointAt(0)!;
    return codePoint >= 32 && !(codePoint >= 127 && codePoint <= 159);
  });

export const wireStableId = (value: unknown, label: string): string => {
  const result = wireString(value, label);
  if (result.length === 0 || result.trim() !== result || !isControlFree(result)) {
    throw new Error(`${label} must be a non-empty, trimmed, control-free stable string`);
  }
  return result;
};

export const wireStableIdArray = (value: unknown, label: string): string[] => {
  const result = wireArray(value, label).map((entry, index) =>
    wireStableId(entry, `${label}[${index}]`),
  );
  if (new Set(result).size !== result.length) {
    throw new Error(`${label} must contain unique stable strings`);
  }
  return result;
};

export const wireNumberArray = (value: unknown, label: string): number[] =>
  wireArray(value, label).map((entry, index) =>
    wireFiniteNumber(entry, `${label}[${index}]`),
  );
