import { useCallback, useEffect, useState } from "react";
import type { SystemHealth } from "../api/schemas";
import * as api from "../api/endpoints";

export function SystemHealthPanel() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [checksum, setChecksum] = useState("");
  const [reason, setReason] = useState("Synthetic recovery exercise");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setHealth(await api.getSystemHealth());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Health query failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const backup = async () => {
    setBusy(true);
    try {
      const artifact = await api.downloadRecoveryPackage();
      setChecksum(artifact.sha256);
      const url = URL.createObjectURL(artifact.payload);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `p1am-recovery-${artifact.configurationRevision}.zip`;
      anchor.click();
      URL.revokeObjectURL(url);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Backup failed");
    } finally {
      setBusy(false);
    }
  };

  const restore = async () => {
    if (!file || !checksum.trim()) {
      setError("Select a package and provide its SHA-256 checksum");
      return;
    }
    setBusy(true);
    try {
      await api.restoreRecoveryPackage(file, checksum.trim(), reason);
      setError(null);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Restore failed");
    } finally {
      setBusy(false);
    }
  };

  const runAcceptance = async () => {
    setBusy(true);
    try {
      const artifact = await api.runRepresentativeScenario();
      const url = URL.createObjectURL(artifact.payload);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${artifact.evidenceId}.zip`;
      anchor.click();
      URL.revokeObjectURL(url);
      setError(artifact.passed ? null : "Scenario completed with failed evidence");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Scenario run failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-label="System health and recovery" style={{ padding: "1rem" }}>
      <div className="panel-header">
        <span>System Health & Recovery</span>
        <button className="btn" onClick={() => void refresh()} disabled={busy}>
          Refresh
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {health && (
        <>
          <p>
            Overall: <strong>{health.overall}</strong> · software {health.identity.software_revision} · configuration {health.identity.configuration_revision}
          </p>
          <ul>
            {health.checks.map((check) => (
              <li key={check.name}>{check.name}: {check.status} — {check.detail}</li>
            ))}
          </ul>
        </>
      )}
      <p style={{ color: "var(--text-muted)" }}>
        Recovery packages exclude energized state and restore into a draft only.
      </p>
      <button className="btn btn-primary" onClick={() => void backup()} disabled={busy}>
        Download Verified Recovery Package
      </button>
      <button className="btn" onClick={() => void runAcceptance()} disabled={busy}>
        Run Synthetic Acceptance Scenario
      </button>
      <div style={{ display: "grid", gap: "0.5rem", marginTop: "0.75rem" }}>
        <label>
          <span className="input-label">Recovery package</span>
          <input type="file" accept=".zip,application/zip" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
        </label>
        <label>
          <span className="input-label">Package SHA-256</span>
          <input className="form-input" value={checksum} onChange={(event) => setChecksum(event.target.value)} />
        </label>
        <label>
          <span className="input-label">Restore reason</span>
          <input className="form-input" value={reason} onChange={(event) => setReason(event.target.value)} />
        </label>
        <button className="btn" onClick={() => void restore()} disabled={busy}>
          Verify & Restore as Draft
        </button>
      </div>
    </section>
  );
}
