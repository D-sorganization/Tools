import React, { useState } from "react";

import { exportDataset } from "../../api/explorer";
import {
  EXPORT_FORMATS,
  type DatasetResponse,
  type ExportFormat,
} from "../../api/explorerSchemas";
import { downloadBlob } from "../../lib/explorer/download";
import type { ExplorerSession, NotifyFn } from "./explorerState";
import { Btn, ErrorText, Field, Row, Select, TextInput } from "./ui";

/**
 * Export the processed dataset (CSV/JSON) and save/restore named analysis
 * sessions (source + pipeline + plot config) in localStorage.
 */

export interface ExportSessionPanelProps {
  dataset: DatasetResponse | null;
  getSession: () => ExplorerSession;
  applySession: (s: ExplorerSession) => void;
  triggerNotification: NotifyFn;
}

const SESSION_KEY = "p1am.explorer.sessions.v1";

function loadSessions(): Record<string, ExplorerSession> {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? (JSON.parse(raw) as Record<string, ExplorerSession>) : {};
  } catch {
    return {};
  }
}

function saveSessions(map: Record<string, ExplorerSession>): void {
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(map));
  } catch {
    /* storage unavailable — non-fatal */
  }
}

export const ExportSessionPanel: React.FC<ExportSessionPanelProps> = ({
  dataset,
  getSession,
  applySession,
  triggerNotification,
}) => {
  const [format, setFormat] = useState<ExportFormat>("csv");
  const [filename, setFilename] = useState("dataset");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Record<string, ExplorerSession>>(loadSessions);
  const [sessionName, setSessionName] = useState("");

  const doExport = async () => {
    if (!dataset) return;
    setBusy(true);
    setError(null);
    try {
      const blob = await exportDataset({
        index: dataset.index,
        columns: dataset.columns,
        format,
        filename: `${filename}.${format}`,
      });
      downloadBlob(blob, `${filename}.${format}`);
      triggerNotification("Dataset exported", "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const saveCurrent = () => {
    const name = sessionName.trim();
    if (!name) return;
    const next = { ...sessions, [name]: getSession() };
    setSessions(next);
    saveSessions(next);
    triggerNotification(`Session "${name}" saved`, "success");
  };

  const remove = (name: string) => {
    const next = { ...sessions };
    delete next[name];
    setSessions(next);
    saveSessions(next);
  };

  const names = Object.keys(sessions).sort();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
      <div>
        <span style={{ fontSize: "0.74rem", color: "var(--text-secondary)" }}>
          Export processed dataset
        </span>
        <Row>
          <Field label="filename">
            <TextInput
              value={filename}
              onChange={(e) => setFilename(e.target.value || "dataset")}
              style={{ width: "12rem" }}
            />
          </Field>
          <Field label="format">
            <Select value={format} onChange={(e) => setFormat(e.target.value as ExportFormat)}>
              {EXPORT_FORMATS.map((f) => (
                <option key={f} value={f}>{f.toUpperCase()}</option>
              ))}
            </Select>
          </Field>
          <Btn variant="primary" onClick={() => void doExport()} disabled={!dataset || busy}>
            {busy ? "Exporting…" : "Export"}
          </Btn>
        </Row>
        <ErrorText>{error}</ErrorText>
      </div>

      <div>
        <span style={{ fontSize: "0.74rem", color: "var(--text-secondary)" }}>
          Analysis sessions
        </span>
        <Row>
          <TextInput
            value={sessionName}
            placeholder="session name"
            onChange={(e) => setSessionName(e.target.value)}
            style={{ width: "12rem" }}
          />
          <Btn onClick={saveCurrent} disabled={!sessionName.trim()}>
            Save current
          </Btn>
        </Row>
        {names.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", marginTop: "0.4rem" }}>
            {names.map((n) => (
              <Row key={n} gap="0.4rem" wrap={false}>
                <span style={{ flex: 1, fontSize: "0.76rem" }}>{n}</span>
                <Btn onClick={() => applySession(sessions[n])}>Load</Btn>
                <Btn variant="danger" onClick={() => remove(n)}>✕</Btn>
              </Row>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
