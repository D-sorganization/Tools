/**
 * One shared snapshot/export library for every SVG chart in the HMI.
 *
 * This is the single DOM-touching module that turns any on-screen `<svg>` chart
 * into a shareable artifact: a standalone SVG, a rasterized PNG, or (where the
 * caller has the underlying rows) a CSV. Every graph exports through here via
 * the reusable {@link file://./../components/SnapshotButton.tsx} so there is one
 * place that owns filename shaping, serialization, canvas rasterization and the
 * browser-download click (DRY).
 *
 * No external dependencies: the app's CSP forbids CDN libraries, so the PNG is
 * produced with native `XMLSerializer` + an offscreen `<canvas>` and the CSV is
 * assembled by hand. Every exported function validates its arguments (DbC) and
 * throws `TypeError` on the wrong type so misuse fails loudly in tests, not
 * silently in the field.
 */

/** Default standalone-SVG background — matches the app's dark slate canvas. */
const DEFAULT_BACKGROUND = "#0f172a";
/** Readable default text colour baked into standalone exports. */
const DEFAULT_TEXT_COLOR = "#f8fafc";
/** Font stack inlined so exported files render without the app's stylesheet. */
const DEFAULT_FONT_FAMILY =
  "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const SVG_NS = "http://www.w3.org/2000/svg";
/** Default device-pixel multiplier for PNG rasterization (crisper on HiDPI). */
const DEFAULT_PNG_SCALE = 2;

/** Zero-pad a positive integer to two digits (e.g. 5 -> "05"). */
function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

/** Throw a TypeError unless `value` is a non-empty string. */
function assertNonEmptyString(value: unknown, fn: string, param: string): void {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${fn}: ${param} must be a non-empty string`);
  }
}

/** Throw a TypeError unless `svg` is an `SVGSVGElement`. */
function assertSvgElement(svg: unknown, fn: string): asserts svg is SVGSVGElement {
  const ok =
    typeof SVGSVGElement !== "undefined"
      ? svg instanceof SVGSVGElement
      : svg != null &&
        typeof (svg as { cloneNode?: unknown }).cloneNode === "function";
  if (!ok) {
    throw new TypeError(`${fn}: svg must be an SVGSVGElement`);
  }
}

/**
 * Build a timestamped download filename, e.g.
 * `timestampedName("p1am_trend", "png")` -> `p1am_trend_2026-07-08_20-31-05.png`.
 *
 * The timestamp is read at call time (never at module load) so each export is
 * stamped when it happens. Colons are stripped so the name is filesystem-safe.
 *
 * @throws TypeError if `prefix` or `ext` is not a non-empty string.
 */
export function timestampedName(prefix: string, ext: string): string {
  assertNonEmptyString(prefix, "timestampedName", "prefix");
  assertNonEmptyString(ext, "timestampedName", "ext");
  // Read the clock inside the function — never at module scope.
  const now = new Date();
  const date = `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(
    now.getDate(),
  )}`;
  const time = `${pad2(now.getHours())}-${pad2(now.getMinutes())}-${pad2(
    now.getSeconds(),
  )}`;
  const cleanExt = ext.replace(/^\.+/, "");
  return `${prefix}_${date}_${time}.${cleanExt}`.replace(/:/g, "-");
}

/**
 * Serialize `svg` to a standalone `image/svg+xml` XML string.
 *
 * The element is cloned (the live DOM is untouched), the SVG namespace is
 * ensured, and a background + text colour + font-family are inlined so the file
 * renders correctly outside the app (as a saved file or rasterized via `<img>`).
 *
 * @param opts.background CSS background colour (default {@link DEFAULT_BACKGROUND}).
 * @throws TypeError if `svg` is not an SVGSVGElement or `background` is not a string.
 */
export function serializeSvg(
  svg: SVGSVGElement,
  opts: { background?: string } = {},
): string {
  assertSvgElement(svg, "serializeSvg");
  const background = opts.background ?? DEFAULT_BACKGROUND;
  if (typeof background !== "string") {
    throw new TypeError("serializeSvg: background must be a string");
  }
  const clone = svg.cloneNode(true) as SVGSVGElement;
  if (!clone.getAttribute("xmlns")) {
    clone.setAttribute("xmlns", SVG_NS);
  }
  const existing = clone.getAttribute("style") ?? "";
  const baked =
    `background-color:${background};color:${DEFAULT_TEXT_COLOR};` +
    `font-family:${DEFAULT_FONT_FAMILY};`;
  clone.setAttribute("style", existing ? `${existing};${baked}` : baked);
  const xml = new XMLSerializer().serializeToString(clone);
  return xml.startsWith("<?xml")
    ? xml
    : `<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n${xml}`;
}

/**
 * Trigger a browser download of `blob` as `filename` via a synthetic anchor.
 *
 * The one place the object-URL / `<a download>` click dance lives, so every
 * export path shares it (DRY). The URL is revoked on the next tick so the
 * download has a chance to start.
 */
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Serialize `svg` and trigger a download of the standalone SVG file.
 *
 * @throws TypeError if `svg` is not an SVGSVGElement or `filename` is empty.
 */
export function downloadSvg(svg: SVGSVGElement, filename: string): void {
  assertSvgElement(svg, "downloadSvg");
  assertNonEmptyString(filename, "downloadSvg", "filename");
  const blob = new Blob([serializeSvg(svg)], {
    type: "image/svg+xml;charset=utf-8",
  });
  triggerDownload(blob, filename);
}

/** UTF-8-safe base64 (btoa only accepts Latin-1) for the data-URL image src. */
function utf8ToBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/** On-screen pixel size, falling back to width/height attrs then the viewBox. */
function clientSize(svg: SVGSVGElement): { width: number; height: number } {
  const rect =
    typeof svg.getBoundingClientRect === "function"
      ? svg.getBoundingClientRect()
      : { width: 0, height: 0 };
  let width = rect.width;
  let height = rect.height;
  if (!width || !height) {
    const aw = Number.parseFloat(svg.getAttribute("width") ?? "");
    const ah = Number.parseFloat(svg.getAttribute("height") ?? "");
    if (aw > 0 && ah > 0) {
      width = aw;
      height = ah;
    }
  }
  if (!width || !height) {
    const vb = svg.viewBox?.baseVal;
    if (vb && vb.width > 0 && vb.height > 0) {
      width = vb.width;
      height = vb.height;
    }
  }
  return { width: width || 640, height: height || 480 };
}

/**
 * Rasterize `svg` to a PNG `Blob` via an offscreen canvas.
 *
 * The SVG is serialized, loaded into an `Image` from a base64 `data:` URL, drawn
 * onto a `<canvas>` sized to the SVG's client size times `scale`, and read back
 * as PNG. Rejects cleanly (never throws asynchronously) if the canvas or
 * `toBlob` is unavailable — e.g. under jsdom.
 *
 * @param opts.scale device-pixel multiplier (default {@link DEFAULT_PNG_SCALE}).
 * @param opts.background standalone background colour.
 * @throws TypeError (synchronously) if `svg` is not an SVGSVGElement or `scale`
 *   is not a positive finite number.
 */
export function svgToPngBlob(
  svg: SVGSVGElement,
  opts: { scale?: number; background?: string } = {},
): Promise<Blob> {
  assertSvgElement(svg, "svgToPngBlob");
  const scale = opts.scale ?? DEFAULT_PNG_SCALE;
  if (typeof scale !== "number" || !Number.isFinite(scale) || scale <= 0) {
    throw new TypeError("svgToPngBlob: scale must be a positive finite number");
  }
  const svgString = serializeSvg(svg, { background: opts.background });
  const { width, height } = clientSize(svg);

  return new Promise<Blob>((resolve, reject) => {
    let canvas: HTMLCanvasElement;
    try {
      canvas = document.createElement("canvas");
    } catch (err) {
      reject(err instanceof Error ? err : new Error(String(err)));
      return;
    }
    if (
      typeof canvas.getContext !== "function" ||
      typeof canvas.toBlob !== "function"
    ) {
      reject(new Error("svgToPngBlob: canvas rasterization is unavailable"));
      return;
    }

    const url = `data:image/svg+xml;base64,${utf8ToBase64(svgString)}`;
    const image = new Image();
    image.onload = (): void => {
      try {
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("svgToPngBlob: 2D canvas context unavailable"));
          return;
        }
        ctx.scale(scale, scale);
        ctx.drawImage(image, 0, 0, width, height);
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error("svgToPngBlob: canvas.toBlob returned null"));
        }, "image/png");
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    };
    image.onerror = (): void => {
      reject(new Error("svgToPngBlob: failed to rasterize the SVG"));
    };
    image.src = url;
  });
}

/**
 * Rasterize `svg` to PNG and trigger a download.
 *
 * @throws TypeError (synchronously) if `svg`/`filename`/`scale` are invalid; the
 *   returned promise rejects if rasterization fails.
 */
export async function downloadPng(
  svg: SVGSVGElement,
  filename: string,
  opts?: { scale?: number; background?: string },
): Promise<void> {
  assertSvgElement(svg, "downloadPng");
  assertNonEmptyString(filename, "downloadPng", "filename");
  const blob = await svgToPngBlob(svg, opts);
  triggerDownload(blob, filename);
}

/** Escape one CSV field: wrap + double-quote when it holds `,` `"` or newlines. */
function escapeCsvValue(value: string | number): string {
  const s = String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * Assemble CSV text from a header row and body rows (CRLF line endings).
 *
 * @throws TypeError if `headers`/`rows` are not arrays of the expected shape.
 */
export function downloadCsv(
  headers: string[],
  rows: (string | number)[][],
  filename: string,
): void {
  if (!Array.isArray(headers) || headers.some((h) => typeof h !== "string")) {
    throw new TypeError("downloadCsv: headers must be an array of strings");
  }
  if (!Array.isArray(rows) || rows.some((r) => !Array.isArray(r))) {
    throw new TypeError("downloadCsv: rows must be an array of arrays");
  }
  assertNonEmptyString(filename, "downloadCsv", "filename");

  // ⚡ Bolt Optimization: Build CSV string with a single-pass loop and string concatenation
  // to avoid allocating intermediate arrays on every row with .map().join().
  let csvString = "";
  for (let i = 0; i < headers.length; i++) {
    csvString += escapeCsvValue(headers[i]);
    if (i < headers.length - 1) csvString += ",";
  }

  for (let i = 0; i < rows.length; i++) {
    csvString += "\r\n";
    const row = rows[i];
    for (let j = 0; j < row.length; j++) {
      csvString += escapeCsvValue(row[j]);
      if (j < row.length - 1) csvString += ",";
    }
  }

  const blob = new Blob([csvString], {
    type: "text/csv;charset=utf-8",
  });
  triggerDownload(blob, filename);
}
