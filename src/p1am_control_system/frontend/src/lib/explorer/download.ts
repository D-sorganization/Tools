/**
 * Browser download / SVG-export glue for the Data Explorer.
 *
 * {@link downloadBlob} triggers a file save for any `Blob`,
 * {@link serializeSvg} freezes an `<svg>` element into a standalone SVG `Blob`
 * (namespace declarations injected so the file opens outside the DOM), and
 * {@link svgToPngBlob} rasterizes that SVG onto a canvas to produce a PNG.
 *
 * This is the one DOM-touching module in the explorer lib; the math lives in
 * its siblings. Functions validate their arguments (DbC).
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const XLINK_NS = "http://www.w3.org/1999/xlink";

/**
 * Trigger a browser download of `blob` as `filename`.
 *
 * Creates a temporary object URL and a synthetic `<a download>` click, then
 * revokes the URL on the next tick.
 *
 * @throws TypeError if `blob` is not a Blob or `filename` is not a non-empty
 *   string.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  if (!(blob instanceof Blob)) {
    throw new TypeError("downloadBlob: blob must be a Blob");
  }
  if (typeof filename !== "string" || filename.trim() === "") {
    throw new TypeError("downloadBlob: filename must be a non-empty string");
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Defer revoke so the navigation/download has a chance to start.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Serialize `svg` to a standalone `image/svg+xml` `Blob`.
 *
 * The element is cloned (the live DOM is left untouched) and the SVG/XLink
 * namespace attributes are ensured so the output is a valid standalone file.
 *
 * @throws TypeError if `svg` is not an SVGSVGElement.
 */
export function serializeSvg(svg: SVGSVGElement): Blob {
  return new Blob([serializeSvgString(svg)], {
    type: "image/svg+xml;charset=utf-8",
  });
}

/**
 * Serialize `svg` to an XML string with namespace declarations injected.
 *
 * Exposed for testing/reuse; {@link serializeSvg} wraps it in a `Blob`.
 *
 * @throws TypeError if `svg` is not an SVGSVGElement.
 */
export function serializeSvgString(svg: SVGSVGElement): string {
  if (typeof SVGSVGElement !== "undefined" && !(svg instanceof SVGSVGElement)) {
    throw new TypeError("serializeSvg: svg must be an SVGSVGElement");
  }
  if (svg == null || typeof svg.cloneNode !== "function") {
    throw new TypeError("serializeSvg: svg must be an SVGSVGElement");
  }
  const clone = svg.cloneNode(true) as SVGSVGElement;
  if (!clone.getAttribute("xmlns")) {
    clone.setAttribute("xmlns", SVG_NS);
  }
  if (!clone.getAttribute("xmlns:xlink")) {
    clone.setAttribute("xmlns:xlink", XLINK_NS);
  }
  const xml = new XMLSerializer().serializeToString(clone);
  return xml.startsWith("<?xml")
    ? xml
    : `<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n${xml}`;
}

/** Read an explicit pixel size from the SVG, falling back to its bounding box. */
function svgPixelSize(svg: SVGSVGElement): { width: number; height: number } {
  const attrWidth = Number.parseFloat(svg.getAttribute("width") ?? "");
  const attrHeight = Number.parseFloat(svg.getAttribute("height") ?? "");
  if (Number.isFinite(attrWidth) && Number.isFinite(attrHeight)) {
    return { width: attrWidth, height: attrHeight };
  }
  const viewBox = svg.viewBox?.baseVal;
  if (viewBox && viewBox.width > 0 && viewBox.height > 0) {
    return { width: viewBox.width, height: viewBox.height };
  }
  const rect =
    typeof svg.getBoundingClientRect === "function"
      ? svg.getBoundingClientRect()
      : { width: 0, height: 0 };
  return {
    width: rect.width || 640,
    height: rect.height || 480,
  };
}

/**
 * Rasterize `svg` to a PNG `Blob` via an offscreen canvas.
 *
 * The SVG is serialized, loaded into an `Image`, drawn onto a canvas sized to
 * the SVG (scaled by `scale` for higher-DPI exports), and read back as PNG.
 *
 * @param scale device-pixel multiplier (default 2); must be > 0.
 * @throws TypeError if `svg` is not an SVGSVGElement or `scale` is invalid.
 */
export function svgToPngBlob(svg: SVGSVGElement, scale = 2): Promise<Blob> {
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new TypeError("svgToPngBlob: scale must be a positive finite number");
  }
  const svgString = serializeSvgString(svg);
  const { width, height } = svgPixelSize(svg);

  return new Promise<Blob>((resolve, reject) => {
    const svgBlob = new Blob([svgString], {
      type: "image/svg+xml;charset=utf-8",
    });
    const url = URL.createObjectURL(svgBlob);
    const image = new Image();
    image.onload = (): void => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(url);
          reject(new Error("svgToPngBlob: 2D canvas context unavailable"));
          return;
        }
        ctx.scale(scale, scale);
        ctx.drawImage(image, 0, 0, width, height);
        URL.revokeObjectURL(url);
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error("svgToPngBlob: canvas.toBlob returned null"));
        }, "image/png");
      } catch (err) {
        URL.revokeObjectURL(url);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    };
    image.onerror = (): void => {
      URL.revokeObjectURL(url);
      reject(new Error("svgToPngBlob: failed to load SVG image"));
    };
    image.src = url;
  });
}
