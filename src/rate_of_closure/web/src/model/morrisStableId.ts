/** Exact portable stable identifiers shared by Morris wire consumers. */

const STABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export const morrisStableId = (value: unknown, name: string): string => {
  if (typeof value === "string"
      && Array.from(value).some((character) => {
        const code = character.codePointAt(0) ?? 0;
        return code <= 0x1f || (code >= 0x7f && code <= 0x9f);
      })) {
    throw new RangeError(`${name} must not contain control characters`);
  }
  if (typeof value !== "string" || !STABLE_ID.test(value)) {
    throw new RangeError(`${name} must be a portable stable identifier`);
  }
  return value;
};
