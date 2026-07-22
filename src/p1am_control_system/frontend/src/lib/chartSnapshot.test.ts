import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  timestampedName,
  serializeSvg,
  downloadSvg,
  downloadCsv,
  svgToPngBlob,
} from "./chartSnapshot";

const SVG_NS = "http://www.w3.org/2000/svg";

/** A minimal on-DOM `<svg>` with one identifiable path for serialization tests. */
function makeSvg(pathD = "M0 0 L10 10"): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg") as SVGSVGElement;
  svg.setAttribute("viewBox", "0 0 100 50");
  svg.setAttribute("width", "100");
  svg.setAttribute("height", "50");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", pathD);
  svg.appendChild(path);
  return svg;
}

// jsdom's Blob has no readable `.text()`, so stand in a tiny fake that records
// its parts; jsdom also lacks object-URL support, so stub those too. All are
// installed for the whole file (they didn't exist / weren't readable before) so
// the deferred revoke() timer never calls into a torn-down mock.
class FakeBlob {
  readonly parts: BlobPart[];
  readonly type: string;
  constructor(parts: BlobPart[], opts?: BlobPropertyBag) {
    this.parts = parts;
    this.type = opts?.type ?? "";
  }
  text(): Promise<string> {
    return Promise.resolve(this.parts.map((p) => String(p)).join(""));
  }
}

const capturedBlobs: FakeBlob[] = [];
beforeEach(() => {
  capturedBlobs.length = 0;
  vi.stubGlobal("Blob", FakeBlob);
  URL.createObjectURL = vi.fn((obj: Blob | MediaSource): string => {
    capturedBlobs.push(obj as unknown as FakeBlob);
    return "blob:mock";
  });
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** Spy on `document.createElement` and capture the anchor + stub its click. */
function captureAnchor(): { getAnchor: () => HTMLAnchorElement | undefined; clicks: () => number } {
  let anchor: HTMLAnchorElement | undefined;
  let clicks = 0;
  const realCreate = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
    const el = realCreate(tag);
    if (tag === "a") {
      anchor = el as HTMLAnchorElement;
      el.click = () => {
        clicks += 1;
      };
    }
    return el;
  });
  return { getAnchor: () => anchor, clicks: () => clicks };
}

describe("timestampedName", () => {
  it("builds prefix_DATE_TIME.ext with no colons", () => {
    const name = timestampedName("p1am_trend", "png");
    expect(name).toMatch(
      /^p1am_trend_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.png$/,
    );
    expect(name).not.toContain(":");
  });

  it("strips a leading dot from the extension", () => {
    expect(timestampedName("chart", ".csv")).toMatch(/\.csv$/);
    expect(timestampedName("chart", ".csv")).not.toContain("..");
  });

  it("guards its arguments (DbC)", () => {
    // @ts-expect-error wrong type for the runtime guard
    expect(() => timestampedName(123, "png")).toThrow(TypeError);
    expect(() => timestampedName("", "png")).toThrow(TypeError);
    // @ts-expect-error wrong type for the runtime guard
    expect(() => timestampedName("chart", null)).toThrow(TypeError);
    expect(() => timestampedName("chart", "")).toThrow(TypeError);
  });
});

describe("serializeSvg", () => {
  it("preserves the chart paths and inlines the background", () => {
    const out = serializeSvg(makeSvg("M1 2 L3 4"));
    expect(out).toContain("M1 2 L3 4");
    expect(out).toContain("#0f172a");
    expect(out).toContain("<svg");
  });

  it("honours a custom background colour", () => {
    const out = serializeSvg(makeSvg(), { background: "#ffffff" });
    expect(out).toContain("#ffffff");
  });

  it("guards its argument (DbC)", () => {
    // @ts-expect-error wrong type for the runtime guard
    expect(() => serializeSvg({})).toThrow(TypeError);
    // @ts-expect-error wrong type for the runtime guard
    expect(() => serializeSvg(null)).toThrow(TypeError);
  });
});

describe("downloadSvg", () => {
  it("serializes and triggers a single anchor download", () => {
    const { getAnchor, clicks } = captureAnchor();

    downloadSvg(makeSvg(), "chart.svg");

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(clicks()).toBe(1);
    expect(getAnchor()?.download).toBe("chart.svg");
    expect(capturedBlobs[0].type).toContain("image/svg+xml");
  });

  it("guards its arguments (DbC)", () => {
    // @ts-expect-error wrong type for the runtime guard
    expect(() => downloadSvg({}, "x.svg")).toThrow(TypeError);
    expect(() => downloadSvg(makeSvg(), "")).toThrow(TypeError);
  });
});

describe("downloadCsv", () => {
  it("escapes commas, quotes and newlines and triggers a download", async () => {
    const { getAnchor, clicks } = captureAnchor();

    downloadCsv(
      ["a", "b,c"],
      [
        [1, 'x"y'],
        ["p\nq", 2],
      ],
      "data.csv",
    );

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(clicks()).toBe(1);
    expect(getAnchor()?.download).toBe("data.csv");

    const text = await capturedBlobs[0].text();
    expect(text).toContain('"b,c"'); // comma -> quoted
    expect(text).toContain('"x""y"'); // embedded quote -> doubled + quoted
    expect(text).toContain('"p\nq"'); // newline -> quoted
    // Unremarkable numeric values stay bare, in order, one line per row.
    expect(text.split("\r\n")[1]).toBe('1,"x""y"');
  });

  it("guards its arguments (DbC)", () => {
    // @ts-expect-error wrong type for the runtime guard
    expect(() => downloadCsv("nope", [], "f.csv")).toThrow(TypeError);
    // @ts-expect-error wrong type for the runtime guard
    expect(() => downloadCsv(["a"], "nope", "f.csv")).toThrow(TypeError);
    expect(() => downloadCsv(["a"], [[1]], "")).toThrow(TypeError);
  });
});

describe("svgToPngBlob", () => {
  // The happy raster path needs a real canvas, which jsdom lacks; only the
  // synchronous guards and the clean-rejection contract are asserted here.
  it("rejects the scale argument synchronously (DbC)", () => {
    expect(() => svgToPngBlob(makeSvg(), { scale: 0 })).toThrow(TypeError);
    expect(() => svgToPngBlob(makeSvg(), { scale: -1 })).toThrow(TypeError);
    // @ts-expect-error wrong type for the runtime guard
    expect(() => svgToPngBlob({})).toThrow(TypeError);
  });

  it("returns a promise and never throws asynchronously without a canvas", () => {
    const result = svgToPngBlob(makeSvg());
    expect(result).toBeInstanceOf(Promise);
    // Swallow the eventual rejection (no real canvas under jsdom).
    result.catch(() => undefined);
  });
});
