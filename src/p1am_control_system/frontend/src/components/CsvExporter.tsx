import React, { useState } from "react";
import { Download, FileText } from "lucide-react";
import type { TriggerNotification } from "../types";
import { buildExportUrl } from "../lib/dataExport";

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

  const handleDownloadCSV = () => {
    const tags = exportTags
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

    if (tags.length === 0) {
      triggerNotification("Please enter at least one Tag ID.", "error");
      return;
    }

    try {
      const url = buildExportUrl(tags, {
        startMs: new Date(exportStart).getTime(),
        endMs: new Date(exportEnd).getTime(),
      });
      window.open(url, "_blank", "noopener");
    } catch {
      triggerNotification("End time must be after start time.", "error");
    }
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
      <form
        style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}
        onSubmit={(e) => {
          e.preventDefault();
          handleDownloadCSV();
        }}
      >
        <div className="input-group">
          <label htmlFor="csv-tags" className="input-label">Tags (comma-separated)</label>
          <input
            id="csv-tags"
            type="text"
            className="form-input"
            value={exportTags}
            onChange={(e) => setExportTags(e.target.value)}
            placeholder="e.g. 0,1,10"
          />
        </div>
        <div className="input-group">
          <label htmlFor="csv-start" className="input-label">Start Time</label>
          <input
            id="csv-start"
            type="datetime-local"
            className="form-input"
            value={exportStart}
            onChange={(e) => setExportStart(e.target.value)}
          />
        </div>
        <div className="input-group">
          <label htmlFor="csv-end" className="input-label">End Time</label>
          <input
            id="csv-end"
            type="datetime-local"
            className="form-input"
            value={exportEnd}
            onChange={(e) => setExportEnd(e.target.value)}
          />
        </div>
        <button
          type="submit"
          className="btn"
          style={{ width: "100%", padding: "0.45rem", fontSize: "0.8rem" }}
        >
          <Download size={14} />
          Export Log Data
        </button>
      </form>
    </div>
  );
};
