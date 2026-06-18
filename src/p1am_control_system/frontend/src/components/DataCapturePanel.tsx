import React, { useCallback, useEffect, useState } from "react";
import { Circle, Download, Trash2, Database } from "lucide-react";
import "./DataCapturePanel.css";
import { getCaptureStatus, clearCapture } from "../api/endpoints";
import type { CaptureStatus } from "../api/schemas";
import { fmtBytes, fmtDuration } from "../lib/format";
import { TAG_INDICES, tagName } from "../lib/tags";

export type { CaptureStatus };

/**
 * Data Capture panel.
 *
 * The backend historian logs every tag on every scan, so capture is automatic
 * and always-on whenever the system is running — this panel surfaces that:
 * live capture status, one-click CSV export of a chosen window, and a guarded
 * "clear" that purges the historian and reclaims disk so a long test campaign
 * cannot overflow the storage device.
 */

const ALL_TAGS = TAG_INDICES.map(tagName).join(",");

export const DataCapturePanel: React.FC = () => {
  const [status, setStatus] = useState<CaptureStatus | null>(null);
  const [windowMinutes, setWindowMinutes] = useState<number>(0); // 0 = all captured
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getCaptureStatus());
    } catch {
      /* transient — keep last good status */
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [refresh]);

  const flash = useCallback((m: string) => {
    setMsg(m);
    setTimeout(() => setMsg(null), 4000);
  }, []);

  const handleExport = useCallback(() => {
    const end = new Date();
    const start =
      windowMinutes > 0
        ? new Date(end.getTime() - windowMinutes * 60_000)
        : status?.oldest_timestamp
          ? new Date(status.oldest_timestamp)
          : new Date(end.getTime() - 3600_000);
    const url =
      `/api/export?tag_ids=${encodeURIComponent(ALL_TAGS)}` +
      `&start_time=${encodeURIComponent(start.toISOString())}` +
      `&end_time=${encodeURIComponent(end.toISOString())}`;
    // Streaming CSV with Content-Disposition — navigating triggers a download.
    window.open(url, "_blank");
    flash("Export started — check your downloads.");
  }, [windowMinutes, status, flash]);

  const handleClear = useCallback(async () => {
    if (
      !window.confirm(
        "Clear ALL captured data (tags + events) and reclaim disk? " +
          "This cannot be undone. Capture resumes immediately.",
      )
    )
      return;
    setBusy(true);
    try {
      const r = await clearCapture(true);
      flash(
        `Cleared ${r.tag_rows_deleted.toLocaleString()} rows · ` +
          `freed ${fmtBytes(r.db_bytes_before - r.db_bytes_after)}.`,
      );
      refresh();
    } catch (e) {
      flash(`Clear failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, [flash, refresh]);

  return (
    <div className="dc">
      <div className="dc-card">
        <div className="dc-head">
          <Database size={16} />
          <span>Data capture</span>
          <span className={`dc-rec ${status?.capturing ? "on" : ""}`}>
            <Circle size={9} fill="currentColor" /> {status?.capturing ? "REC" : "OFF"}
          </span>
        </div>
        <p className="dc-sub">
          Every tag is logged on every scan automatically while the system runs —
          no need to start it. Export a window for analysis, or clear the cache to
          free space on the capture device.
        </p>

        <div className="dc-stats">
          <Stat label="Samples" value={(status?.total_rows ?? 0).toLocaleString()} />
          <Stat label="Tags" value={`${status?.distinct_tags ?? 0}`} />
          <Stat label="Span" value={fmtDuration(status?.span_seconds ?? 0)} />
          <Stat label="On disk" value={fmtBytes(status?.db_bytes ?? 0)} />
          <Stat label="Events" value={(status?.event_rows ?? 0).toLocaleString()} />
        </div>
      </div>

      <div className="dc-grid">
        <div className="dc-card">
          <div className="dc-card-title">Export dataset (CSV)</div>
          <label className="dc-field">
            <span>Window</span>
            <select
              value={windowMinutes}
              onChange={(e) => setWindowMinutes(Number(e.target.value))}
            >
              <option value={0}>All captured</option>
              <option value={1}>Last 1 min</option>
              <option value={5}>Last 5 min</option>
              <option value={15}>Last 15 min</option>
              <option value={60}>Last 60 min</option>
            </select>
          </label>
          <button className="btn btn-primary dc-btn" onClick={handleExport}>
            <Download size={14} /> Download CSV
          </button>
        </div>

        <div className="dc-card dc-danger">
          <div className="dc-card-title">Clear captured data</div>
          <p className="dc-sub">
            Purges all samples and events, then VACUUMs to return disk space.
            Capture continues automatically afterward.
          </p>
          <button className="btn dc-btn dc-clear" onClick={handleClear} disabled={busy}>
            <Trash2 size={14} /> Clear Cache
          </button>
        </div>
      </div>

      {msg && <div className="dc-toast">{msg}</div>}
    </div>
  );
};

const Stat: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="dc-stat">
    <div className="dc-stat-label">{label}</div>
    <div className="dc-stat-value">{value}</div>
  </div>
);
