import { useCallback, useEffect, useState } from "react";
import type { ProfessionalAlarm } from "../api/schemas";
import * as api from "../api/endpoints";

export function ProfessionalAlarmPanel() {
  const [alarms, setAlarms] = useState<ProfessionalAlarm[]>([]);
  const [reason, setReason] = useState("Synthetic maintenance");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setAlarms(await api.getProfessionalAlarms());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Alarm query failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const mutate = async (operation: () => Promise<unknown>) => {
    try {
      await operation();
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Alarm action failed");
    }
  };

  return (
    <section aria-label="Professional alarm lifecycle" style={{ padding: "1rem" }}>
      <div className="panel-header">
        <span>Professional Alarm Lifecycle</span>
        <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
          Supervisory demonstration — not independent protection
        </span>
      </div>
      <label style={{ display: "block", marginBottom: "0.75rem" }}>
        <span className="input-label">Shelving reason</span>
        <input
          className="form-input"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      {alarms.length === 0 ? (
        <p style={{ color: "var(--text-muted)" }}>No active lifecycle alarms.</p>
      ) : (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {alarms.map((alarm) => (
            <article
              key={alarm.tag}
              style={{ border: "1px solid var(--panel-border)", padding: "0.75rem" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <strong>{alarm.tag}</strong>
                <span>{alarm.priority} · {alarm.lifecycle}</span>
              </div>
              <p>Condition: {alarm.condition}</p>
              <p>{alarm.first_out_sequence ? `First-out #${alarm.first_out_sequence}` : "No first-out order"}</p>
              <p style={{ color: "var(--text-secondary)" }}>{alarm.help_text}</p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button
                  className="btn"
                  aria-label={`Acknowledge ${alarm.tag}`}
                  onClick={() => void mutate(() => api.acknowledgeProfessionalAlarm(alarm.tag))}
                >
                  Acknowledge
                </button>
                <button
                  className="btn"
                  onClick={() => void mutate(() => api.shelfProfessionalAlarm(alarm.tag, reason, 900))}
                >
                  Shelf 15 min
                </button>
                {alarm.lifecycle === "shelved" && (
                  <button
                    className="btn"
                    onClick={() => void mutate(() => api.unshelveProfessionalAlarm(alarm.tag))}
                  >
                    Unshelve
                  </button>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
