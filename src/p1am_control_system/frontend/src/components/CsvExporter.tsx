import React, { useState, useId } from "react";
import { Download, FileText } from "lucide-react";
import type { TriggerNotification } from "../types";

/** Local datetime-local default string offset by `minutesAgo` from now. */
function localDateTimeInput(minutesAgo: number): string {
  const d = new Date();
  d.setMinutes(d.getMinutes() - minutesAgo);
  const tzOffset = d.getTimezoneOffset() * 60000;
  return new Date(d.getTime() - tzOffset).toISOString().slice(0, 16);
}

/**
 * Historical telemetry CSV downloader (#3543).
 *
 * Extracted from App.tsx's default-sidebar view; owns its own form state and
 * opens the backend `/api/export` streaming endpoint in a new tab.
 */
export const CsvExporter: React.FC<{
  triggerNotification: TriggerNotification;
}> = ({ triggerNotification }) => {
  const [exportTags, setExportTags] = useState<string>("0, 1, 10");
  const [exportStart, setExportStart] = useState<string>(() =>
    localDateTimeInput(15),
  );
  const [exportEnd, setExportEnd] = useState<string>(() =>
    localDateTimeInput(0),
  );

  const tagsId = useId();
  const startId = useId();
  const endId = useId();

  const handleDownloadCSV = () => {
    const startISO = new Date(exportStart).toISOString();
    const endISO = new Date(exportEnd).toISOString();
    const cleanedTags = exportTags
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .join(",");

    if (!cleanedTags) {
      triggerNotification("Please enter at least one Tag ID.", "error");
      return;
    }

    const url = `/api/export?tag_ids=${encodeURIComponent(
      cleanedTags,
    )}&start_time=${encodeURIComponent(startISO)}&end_time=${encodeURIComponent(
      endISO,
    )}`;
    window.open(url, "_blank");
  };

  return (
    <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "1rem" }}>
      <h3
        style={{
          fontSize: "0.85rem",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
          marginBottom: "0.75rem",
          display: "flex",
          alignItems: "center",
          gap: "0.3rem",
        }}
      >
        <FileText size={14} color="var(--accent-purple)" />
        <span>CSV Data Exporter</span>
      </h3>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        <div className="input-group">
          <label htmlFor={tagsId} className="input-label">Tags (comma-separated)</label>
          <input
            id={tagsId}
            type="text"
            className="form-input"
            value={exportTags}
            onChange={(e) => setExportTags(e.target.value)}
            placeholder="e.g. 0,1,10"
          />
        </div>
        <div className="input-group">
          <label htmlFor={startId} className="input-label">Start Time</label>
          <input
            id={startId}
            type="datetime-local"
            className="form-input"
            value={exportStart}
            onChange={(e) => setExportStart(e.target.value)}
          />
        </div>
        <div className="input-group">
          <label htmlFor={endId} className="input-label">End Time</label>
          <input
            id={endId}
            type="datetime-local"
            className="form-input"
            value={exportEnd}
            onChange={(e) => setExportEnd(e.target.value)}
          />
        </div>
        <button
          type="button"
          onClick={handleDownloadCSV}
          className="btn"
          style={{ width: "100%", padding: "0.45rem", fontSize: "0.8rem" }}
        >
          <Download size={14} />
          Export Log Data
        </button>
      </div>
    </div>
  );
};
