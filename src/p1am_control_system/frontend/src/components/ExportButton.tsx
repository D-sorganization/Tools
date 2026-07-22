import React, { useState } from "react";
import { Download } from "lucide-react";
import {
  RANGE_PRESETS,
  type RangePresetId,
  resolveRange,
  buildExportUrl,
} from "../lib/dataExport";

/**
 * Universal one-click CSV export for any panel: pick a time range, click, and
 * the browser downloads the chosen tags from the historian (`/api/export`).
 * Every panel passes the tags it shows, so the export is always pre-scoped —
 * no manual tag entry. Built on the shared dataExport lib (DRY).
 */
export const ExportButton: React.FC<{
  /** Tag ids/names this panel should export (pre-filled for the operator). */
  tags: ReadonlyArray<number | string>;
  label?: string;
  defaultRange?: RangePresetId;
  /** Surface a problem (e.g. empty tag set) to the host's toast system. */
  onError?: (message: string) => void;
}> = ({ tags, label = "Export CSV", defaultRange = "today", onError }) => {
  const [range, setRange] = useState<RangePresetId>(defaultRange);

  const handleExport = (): void => {
    try {
      const url = buildExportUrl(tags, resolveRange(range, Date.now()));
      window.open(url, "_blank", "noopener");
    } catch (err) {
      onError?.(err instanceof Error ? err.message : "Export failed");
    }
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.7rem" }}>
      <select
        value={range}
        onChange={(e) => setRange(e.target.value as RangePresetId)}
        aria-label="Export time range"
        title="Time range to export"
        style={{ fontSize: "0.7rem", padding: "0.12rem 0.25rem" }}
      >
        {RANGE_PRESETS.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label}
          </option>
        ))}
      </select>
      <button
        type="button"
        className="btn"
        onClick={handleExport}
        disabled={tags.length === 0}
        title="Download captured data for these tags as CSV"
        style={{ padding: "0.15rem 0.45rem", fontSize: "0.7rem", display: "flex", alignItems: "center", gap: "0.25rem" }}
      >
        <Download size={13} />
        {label}
      </button>
    </div>
  );
};
