import { useCallback, useEffect, useState } from "react";
import type {
  ConfigurationDiffEntry,
  ConfigurationRevision,
} from "../api/schemas";
import * as api from "../api/endpoints";

const actionLabel: Record<string, string> = {
  draft: "Validate",
  validated: "Submit for review",
  in_review: "Approve",
  approved: "Activate",
};

export function ConfigurationWorkflowPanel() {
  const [revisions, setRevisions] = useState<ConfigurationRevision[]>([]);
  const [diff, setDiff] = useState<ConfigurationDiffEntry[]>([]);
  const [reason, setReason] = useState("Reviewed representative configuration");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = revisions[revisions.length - 1];

  const refresh = useCallback(async () => {
    try {
      const next = await api.getConfigurationRevisions();
      setRevisions(next);
      const candidate = next[next.length - 1];
      setDiff(candidate ? await api.getConfigurationDiff(candidate.revision_id) : []);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Configuration query failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const advance = async () => {
    if (!latest) return;
    setBusy(true);
    try {
      if (latest.state === "draft") {
        await api.validateConfiguration(latest.revision_id);
      } else if (latest.state === "validated") {
        await api.reviewConfiguration(latest.revision_id);
      } else if (latest.state === "in_review") {
        await api.approveConfiguration(latest.revision_id, reason);
      } else if (latest.state === "approved") {
        await api.activateConfiguration(latest.revision_id);
      }
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Configuration action failed");
    } finally {
      setBusy(false);
    }
  };

  const rollback = async (revision: ConfigurationRevision) => {
    setBusy(true);
    try {
      await api.rollbackConfiguration(revision.revision_id, reason);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Rollback failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-label="Protected configuration workflow" style={{ padding: "1rem" }}>
      <div className="panel-header">
        <span>Protected Configuration Workflow</span>
        <button className="btn" onClick={() => void refresh()} disabled={busy}>
          Refresh
        </button>
      </div>
      <p style={{ color: "var(--text-muted)" }}>
        Drafts require validation, review, approval, and identified activation.
      </p>
      <label style={{ display: "block", marginBottom: "0.75rem" }}>
        <span className="input-label">Review or rollback reason</span>
        <input
          className="form-input"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      {!latest ? (
        <p>No protected revisions yet. Create a draft from an editor.</p>
      ) : (
        <>
          <p>
            <strong>{latest.revision_id}</strong> · {latest.state} · SHA-256 {latest.payload_sha256.slice(0, 12)}…
          </p>
          <p>{diff.length} changed configuration fields in the current diff.</p>
          {actionLabel[latest.state] && (
            <button className="btn btn-primary" onClick={() => void advance()} disabled={busy}>
              {busy ? "Working…" : actionLabel[latest.state]}
            </button>
          )}
        </>
      )}
      <div style={{ marginTop: "0.75rem" }}>
        {revisions
          .filter((revision) => revision.state === "superseded")
          .slice(-3)
          .map((revision) => (
            <button
              key={revision.revision_id}
              className="btn"
              disabled={busy}
              onClick={() => void rollback(revision)}
            >
              Roll back to {revision.revision_id}
            </button>
          ))}
      </div>
    </section>
  );
}
