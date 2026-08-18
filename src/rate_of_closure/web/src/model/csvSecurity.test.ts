import { describe, expect, it } from "vitest";

import { spreadsheetSafeCsvCell } from "./csvSecurity";

describe("spreadsheet-safe CSV cells", () => {
  it("neutralizes every formula prefix while preserving numeric negatives", () => {
    for (const text of ["=1+1", "+SUM(A:A)", "-cmd", "@name", "\tformula", "\rformula"]) {
      expect(spreadsheetSafeCsvCell(text)).toContain(`'${text}`);
    }
    expect(spreadsheetSafeCsvCell(-3.25)).toBe("-3.25");
  });
});
