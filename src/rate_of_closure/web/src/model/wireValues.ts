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

export const wireNumberArray = (value: unknown, label: string): number[] =>
  wireArray(value, label).map((entry, index) =>
    wireFiniteNumber(entry, `${label}[${index}]`),
  );
