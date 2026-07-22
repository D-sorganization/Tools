import { describe, it, expect } from "vitest";
import { parseCsv } from "./csv";

describe("parseCsv — index detection", () => {
  it("uses a 'timestamp' column as the epoch-ms index (ISO)", () => {
    const text =
      "timestamp,temp\n" +
      "2021-01-01T00:00:00Z,10\n" +
      "2021-01-01T00:00:01Z,11\n";
    const { index, columns } = parseCsv(text);
    expect(index).not.toBeNull();
    expect(index).toEqual([
      Date.parse("2021-01-01T00:00:00Z"),
      Date.parse("2021-01-01T00:00:01Z"),
    ]);
    // The time column is excluded from value columns.
    expect(columns).toHaveLength(1);
    expect(columns[0].name).toBe("temp");
    expect(columns[0].values).toEqual([10, 11]);
  });

  it("treats a numeric epoch in a named time column as milliseconds", () => {
    const text = "time,v\n1000,1\n2000,2\n3000,3\n";
    const { index, columns } = parseCsv(text);
    expect(index).toEqual([1000, 2000, 3000]);
    expect(columns[0].name).toBe("v");
    expect(columns[0].values).toEqual([1, 2, 3]);
  });

  it("is case-insensitive about the time header name", () => {
    const text = "DateTime,x\n2021-06-01,5\n2021-06-02,6\n";
    const { index } = parseCsv(text);
    expect(index).toEqual([
      Date.parse("2021-06-01"),
      Date.parse("2021-06-02"),
    ]);
  });

  it("auto-detects an unnamed ISO-date column as the index", () => {
    const text = "when,reading\n2020-03-01T00:00:00Z,7\n2020-03-01T01:00:00Z,8\n";
    const { index, columns } = parseCsv(text);
    expect(index).toEqual([
      Date.parse("2020-03-01T00:00:00Z"),
      Date.parse("2020-03-01T01:00:00Z"),
    ]);
    expect(columns.map((c) => c.name)).toEqual(["reading"]);
  });

  it("returns index=null when there is no time column", () => {
    const text = "a,b\n1,2\n3,4\n";
    const { index, columns } = parseCsv(text);
    expect(index).toBeNull();
    expect(columns.map((c) => c.name)).toEqual(["a", "b"]);
    expect(columns[0].values).toEqual([1, 3]);
    expect(columns[1].values).toEqual([2, 4]);
  });

  it("does NOT hijack a plain numeric column as a time index", () => {
    // No name match and the column is plain integers -> not a date column.
    const text = "count,value\n1,100\n2,200\n3,300\n";
    const { index, columns } = parseCsv(text);
    expect(index).toBeNull();
    expect(columns).toHaveLength(2);
  });
});

describe("parseCsv — numeric cells & gaps", () => {
  it("maps non-numeric / empty cells to null", () => {
    const text = "a,b\n1,foo\n,2\nbar,3\n";
    const { columns } = parseCsv(text);
    const a = columns.find((c) => c.name === "a")!;
    const b = columns.find((c) => c.name === "b")!;
    expect(a.values).toEqual([1, null, null]);
    expect(b.values).toEqual([null, 2, 3]);
  });

  it("parses negative, decimal, and scientific notation", () => {
    const text = "x\n-1.5\n2e3\n0.25\n";
    const { columns } = parseCsv(text);
    expect(columns[0].values).toEqual([-1.5, 2000, 0.25]);
  });
});

describe("parseCsv — quoting & delimiters", () => {
  it("handles quoted fields containing commas", () => {
    const text = 'label,value\n"a,b",1\n"c,d",2\n';
    const { columns } = parseCsv(text);
    const label = columns.find((c) => c.name === "label")!;
    // Non-numeric strings become null but the row count is preserved.
    expect(label.values).toEqual([null, null]);
    const value = columns.find((c) => c.name === "value")!;
    expect(value.values).toEqual([1, 2]);
  });

  it("unescapes doubled quotes inside a quoted field", () => {
    const text = 'name,n\n"say ""hi""",1\n';
    const { columns } = parseCsv(text);
    // Only assert it parsed without throwing and produced one data row.
    expect(columns.find((c) => c.name === "n")!.values).toEqual([1]);
  });

  it("handles \\r\\n line endings and a trailing newline", () => {
    const text = "a,b\r\n1,2\r\n3,4\r\n";
    const { columns } = parseCsv(text);
    expect(columns[0].values).toEqual([1, 3]);
    expect(columns[1].values).toEqual([2, 4]);
  });

  it("handles a final row without a trailing newline", () => {
    const text = "a,b\n1,2\n3,4";
    const { columns } = parseCsv(text);
    expect(columns[0].values).toEqual([1, 3]);
  });
});

describe("parseCsv — DbC error paths", () => {
  it("throws TypeError when text is not a string", () => {
    // @ts-expect-error testing runtime guard
    expect(() => parseCsv(123)).toThrow(TypeError);
  });

  it("throws on empty input", () => {
    expect(() => parseCsv("")).toThrow(Error);
    expect(() => parseCsv("   \n  \n")).toThrow(Error);
  });

  it("throws on header-only input (no data rows)", () => {
    expect(() => parseCsv("a,b\n")).toThrow(/no data rows/i);
    expect(() => parseCsv("only,header")).toThrow(/no data rows/i);
  });
});
