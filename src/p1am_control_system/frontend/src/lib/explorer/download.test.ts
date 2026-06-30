import { afterEach, describe, expect, it } from "vitest";

import { downloadBlob, serializeSvg, serializeSvgString } from "./download";

const SVG_NS = "http://www.w3.org/2000/svg";

function makeSvg(): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg") as SVGSVGElement;
  svg.setAttribute("width", "100");
  svg.setAttribute("height", "80");
  const text = document.createElementNS(SVG_NS, "text");
  text.setAttribute("fill", "var(--text-primary)");
  text.textContent = "hi";
  svg.appendChild(text);
  return svg;
}

afterEach(() => {
  document.documentElement.style.removeProperty("--accent-cyan");
});

describe("serializeSvgString", () => {
  it("rejects a non-SVG argument", () => {
    // @ts-expect-error intentional wrong type
    expect(() => serializeSvgString({})).toThrow(TypeError);
  });

  it("produces standalone SVG with xmlns", () => {
    const out = serializeSvgString(makeSvg());
    expect(out).toContain("<svg");
    expect(out).toContain(`xmlns="${SVG_NS}"`);
  });

  it("bakes resolved theme CSS variables into the exported SVG", () => {
    document.documentElement.style.setProperty("--accent-cyan", "#abcdef");
    const out = serializeSvgString(makeSvg());
    // The injected <style> redeclares the active theme vars so var(--x)
    // resolves in a standalone file (regression: export was colorless).
    expect(out).toContain("--accent-cyan: #abcdef");
    expect(out).toContain("<style");
  });
});

describe("serializeSvg", () => {
  it("returns an image/svg+xml Blob", () => {
    const blob = serializeSvg(makeSvg());
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toContain("image/svg+xml");
  });
});

describe("downloadBlob", () => {
  it("validates its arguments", () => {
    // @ts-expect-error intentional wrong type
    expect(() => downloadBlob("nope", "f.txt")).toThrow(TypeError);
    expect(() => downloadBlob(new Blob(["x"]), "")).toThrow(TypeError);
  });
});
