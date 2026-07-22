import React, { useCallback, useEffect, useState } from "react";
import { Camera, Image, FileSpreadsheet } from "lucide-react";
import {
  downloadCsv,
  downloadPng,
  downloadSvg,
  timestampedName,
} from "../lib/chartSnapshot";

/**
 * One reusable "share this chart" control for every SVG graph in the HMI.
 *
 * Renders a compact PNG / SVG (/ optional CSV) button group that exports the
 * chart referenced by `targetRef` through the shared {@link file://./../lib/chartSnapshot.ts}
 * library (DRY — no component builds its own download code). Buttons are real
 * `<button>`s (keyboard + screen-reader accessible) with `aria-label`s; the
 * image buttons disable while the SVG ref is unattached, and the PNG button
 * shows a brief busy state while it rasterizes. Export failures are logged and
 * swallowed so a bad export never crashes the HMI.
 */
export interface SnapshotButtonProps {
  /** Ref to the chart's root `<svg>` (image exports read `.current`). */
  targetRef: React.RefObject<SVGSVGElement | null>;
  /** Filename prefix, e.g. `"heater_trend"`; the lib appends a timestamp + ext. */
  filename: string;
  /** Optional tabular data; when present a CSV button is shown. */
  csv?: { headers: string[]; rows: (string | number)[][] };
  /** Accessible group label (default "Export chart snapshot"). */
  label?: string;
}

const groupStyle: React.CSSProperties = {
  display: "inline-flex",
  gap: "0.3rem",
  alignItems: "center",
};

const btnStyle: React.CSSProperties = {
  padding: "0.25rem 0.5rem",
  fontSize: "0.7rem",
};

export const SnapshotButton: React.FC<SnapshotButtonProps> = ({
  targetRef,
  filename,
  csv,
  label,
}) => {
  const [busy, setBusy] = useState(false);

  // A ref attaches after the first commit and does not itself trigger a
  // re-render, so the initial `disabled` read would be stale. Nudge exactly one
  // re-render on mount so the disabled state reflects the now-attached <svg>.
  const [, bumpMount] = useState(0);
  useEffect(() => {
    bumpMount((n) => n + 1);
  }, []);

  const svgMissing = targetRef.current == null;

  const handlePng = useCallback(async () => {
    const svg = targetRef.current;
    if (!svg) return;
    setBusy(true);
    try {
      await downloadPng(svg, timestampedName(filename, "png"));
    } catch (err) {
      console.error("SnapshotButton: PNG export failed", err);
    } finally {
      setBusy(false);
    }
  }, [targetRef, filename]);

  const handleSvg = useCallback(() => {
    const svg = targetRef.current;
    if (!svg) return;
    try {
      downloadSvg(svg, timestampedName(filename, "svg"));
    } catch (err) {
      console.error("SnapshotButton: SVG export failed", err);
    }
  }, [targetRef, filename]);

  const handleCsv = useCallback(() => {
    if (!csv) return;
    try {
      downloadCsv(csv.headers, csv.rows, timestampedName(filename, "csv"));
    } catch (err) {
      console.error("SnapshotButton: CSV export failed", err);
    }
  }, [csv, filename]);

  return (
    <div style={groupStyle} role="group" aria-label={label ?? "Export chart snapshot"}>
      <button
        type="button"
        className="btn"
        style={btnStyle}
        onClick={handlePng}
        disabled={svgMissing || busy}
        title="Download a PNG image of this chart"
        aria-label="Download chart as PNG image"
      >
        <Camera size={12} aria-hidden="true" />
        <span>{busy ? "…" : "PNG"}</span>
      </button>
      <button
        type="button"
        className="btn"
        style={btnStyle}
        onClick={handleSvg}
        disabled={svgMissing}
        title="Download a standalone SVG of this chart"
        aria-label="Download chart as SVG vector"
      >
        <Image size={12} aria-hidden="true" />
        <span>SVG</span>
      </button>
      {csv && (
        <button
          type="button"
          className="btn"
          style={btnStyle}
          onClick={handleCsv}
          title="Download this chart's data as CSV"
          aria-label="Download chart data as CSV"
        >
          <FileSpreadsheet size={12} aria-hidden="true" />
          <span>CSV</span>
        </button>
      )}
    </div>
  );
};
