import React, { useRef } from "react";

import type { SignalInfo } from "../../api/explorerSchemas";
import { parseCsv } from "../../lib/explorer/csv";
import type { CsvSource, HistorianForm, SourceMode } from "./explorerState";
import { Btn, Check, ErrorText, Field, NumInput, Row, TextInput } from "./ui";

/**
 * Choose where the dataset comes from: the live historian (pick tags + a time
 * window) or a CSV file parsed in the browser. Controlled — all state lives in
 * the {@link DataExplorer} container.
 */

export interface SourcePanelProps {
  signals: SignalInfo[];
  mode: SourceMode;
  onModeChange: (mode: SourceMode) => void;
  historian: HistorianForm;
  onHistorianChange: (form: HistorianForm) => void;
  csv: CsvSource | null;
  onCsvLoaded: (csv: CsvSource | null) => void;
  onError: (message: string) => void;
  onBuild: () => void;
  loading: boolean;
  error: string | null;
}

const PRESETS: { label: string; seconds: number | "all" }[] = [
  { label: "15m", seconds: 15 * 60 },
  { label: "1h", seconds: 3600 },
  { label: "6h", seconds: 6 * 3600 },
  { label: "24h", seconds: 24 * 3600 },
  { label: "All", seconds: "all" },
];

/** Local datetime-input string (`YYYY-MM-DDTHH:mm:ss`) for an epoch-ms instant. */
function toLocalInput(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

export const SourcePanel: React.FC<SourcePanelProps> = ({
  signals,
  mode,
  onModeChange,
  historian,
  onHistorianChange,
  csv,
  onCsvLoaded,
  onError,
  onBuild,
  loading,
  error,
}) => {
  const fileRef = useRef<HTMLInputElement>(null);

  const toggleTag = (name: string) => {
    const set = new Set(historian.tags);
    if (set.has(name)) set.delete(name);
    else set.add(name);
    onHistorianChange({ ...historian, tags: [...set] });
  };

  const applyPreset = (seconds: number | "all") => {
    const now = Date.now();
    if (seconds === "all") {
      // ⚡ Bolt Optimization: Replace chained map/filter and Math.min(...spread)
      // with a single-pass loop to avoid intermediate allocations and call stack limits.
      let start = now - 3600_000;
      let hasStart = false;
      for (let i = 0; i < signals.length; i++) {
        const startTime = signals[i].start_time;
        if (startTime) {
          const t = Date.parse(startTime);
          if (Number.isFinite(t)) {
            if (!hasStart || t < start) {
              start = t;
              hasStart = true;
            }
          }
        }
      }
      onHistorianChange({
        ...historian,
        start: toLocalInput(start),
        end: toLocalInput(now),
      });
      return;
    }
    onHistorianChange({
      ...historian,
      start: toLocalInput(now - seconds * 1000),
      end: toLocalInput(now),
    });
  };

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      const text = await file.text();
      const table = parseCsv(text);
      onCsvLoaded({ name: file.name, index: table.index, columns: table.columns });
    } catch (err) {
      // Surface parse failures instead of silently dropping the selection.
      onCsvLoaded(null);
      onError(
        `Could not parse ${file.name}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  const canBuild =
    !loading &&
    (mode === "historian"
      ? historian.tags.length > 0 && !!historian.start && !!historian.end
      : !!csv && csv.columns.length > 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem" }}>
      <Row>
        <Btn
          variant={mode === "historian" ? "primary" : "ghost"}
          onClick={() => onModeChange("historian")}
        >
          Historian
        </Btn>
        <Btn
          variant={mode === "csv" ? "primary" : "ghost"}
          onClick={() => onModeChange("csv")}
        >
          Upload CSV
        </Btn>
      </Row>

      {mode === "historian" ? (
        <>
          <div>
            <span
              style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}
            >
              Signals ({historian.tags.length} selected)
            </span>
            <div
              style={{
                marginTop: "0.3rem",
                maxHeight: "8.5rem",
                overflowY: "auto",
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(8rem, 1fr))",
                gap: "0.15rem 0.6rem",
                border: "1px solid var(--panel-border)",
                borderRadius: "5px",
                padding: "0.4rem 0.5rem",
                background: "var(--input-bg)",
              }}
            >
              {signals.length === 0 && (
                <span
                  style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}
                >
                  No historian signals yet.
                </span>
              )}
              {signals.map((s) => (
                <Check
                  key={s.name}
                  label={`${s.name} (${s.count})`}
                  checked={historian.tags.includes(s.name)}
                  onChange={() => toggleTag(s.name)}
                />
              ))}
            </div>
          </div>

          <Row>
            {PRESETS.map((p) => (
              <Btn key={p.label} onClick={() => applyPreset(p.seconds)}>
                {p.label}
              </Btn>
            ))}
          </Row>

          <Row>
            <Field label="Start">
              <TextInput
                value={historian.start}
                placeholder="YYYY-MM-DDTHH:mm:ss"
                onChange={(e) =>
                  onHistorianChange({ ...historian, start: e.target.value })
                }
                style={{ width: "12rem" }}
              />
            </Field>
            <Field label="End">
              <TextInput
                value={historian.end}
                placeholder="YYYY-MM-DDTHH:mm:ss"
                onChange={(e) =>
                  onHistorianChange({ ...historian, end: e.target.value })
                }
                style={{ width: "12rem" }}
              />
            </Field>
            <Field label="Max points">
              <NumInput
                min={10}
                max={200000}
                value={historian.maxPoints}
                onChange={(e) =>
                  onHistorianChange({
                    ...historian,
                    maxPoints: Number(e.target.value) || 5000,
                  })
                }
              />
            </Field>
          </Row>
        </>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            style={{ display: "none" }}
            onChange={(e) => {
              void handleFile(e.target.files?.[0]);
              e.target.value = ""; // allow re-selecting the same file
            }}
          />
          <Row>
            <Btn onClick={() => fileRef.current?.click()}>Choose CSV…</Btn>
            {csv && (
              <span
                style={{ fontSize: "0.74rem", color: "var(--text-secondary)" }}
              >
                {csv.name} — {csv.columns.length} columns
                {csv.index ? " (time index)" : ""}
              </span>
            )}
          </Row>
          <span style={{ fontSize: "0.66rem", color: "var(--text-muted)" }}>
            A column named time/timestamp/date is used as the x-axis; otherwise a
            row index is synthesized.
          </span>
        </div>
      )}

      <Row>
        <Btn variant="primary" onClick={onBuild} disabled={!canBuild}>
          {loading ? "Building…" : "Build dataset"}
        </Btn>
      </Row>
      <ErrorText>{error}</ErrorText>
    </div>
  );
};
