import { useEffect, useState } from "react";
import { getOperatorOverview, getProtectionSnapshot } from "../api/endpoints";
import type {
  AssetFaceplate,
  ProcessOverview,
  ProtectionSnapshot,
} from "../api/schemas";

const cardStyle = {
  border: "1px solid var(--panel-border)",
  borderRadius: "0.65rem",
  background: "var(--panel-bg)",
  padding: "0.8rem",
} as const;

function Faceplate({ asset, onClose }: { asset: AssetFaceplate; onClose: () => void }) {
  return (
    <section
      role="dialog"
      aria-label={`${asset.label} faceplate`}
      aria-modal="true"
      style={{ ...cardStyle, position: "fixed", right: "1rem", top: "5rem", zIndex: 30, minWidth: 300 }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
        <div>
          <strong>{asset.label}</strong>
          <div style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>{asset.asset_id}</div>
        </div>
        <button type="button" onClick={onClose} aria-label="Close faceplate">×</button>
      </header>
      <p style={{ fontSize: "1.7rem", margin: "0.8rem 0" }}>
        {asset.primary_value.value} {asset.primary_value.unit}
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem" }}>
        <div>Quality {asset.quality}</div>
        <div>Mode {asset.mode}</div>
        <div>Alarm {asset.alarm_state}</div>
        <div>Interlock {asset.interlock_state}</div>
      </div>
      <button type="button" style={{ marginTop: "0.75rem" }}>
        Open trend drill-down
      </button>
    </section>
  );
}

function ProtectionView({ snapshot }: { snapshot: ProtectionSnapshot }) {
  return (
    <section aria-labelledby="protection-heading" style={{ marginTop: "1rem" }}>
      <h3 id="protection-heading">Protection, permissive, and first-out context</h3>
      {snapshot.active_bypasses.map((bypass) => (
        <div key={bypass.protection_id} role="alert" style={{ ...cardStyle, borderColor: "var(--color-warning)" }}>
          <strong>ACTIVE MANAGED BYPASS</strong> — {bypass.protection_id}: {bypass.reason}. Expires {bypass.expires_at}.
        </div>
      ))}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "0.7rem", marginTop: "0.7rem" }}>
        {snapshot.definitions.map((definition) => {
          const trip = snapshot.trips.find((item) => item.protection_id === definition.protection_id);
          return (
            <article key={definition.protection_id} style={cardStyle}>
              <strong>{definition.category.replace("_", " ")}</strong>
              {trip?.first_out && <div style={{ color: "var(--color-error)", fontWeight: 800 }}>FIRST OUT</div>}
              <div>{definition.protection_id}</div>
              <ul>{definition.consequences.map((item) => <li key={item}>{item}</li>)}</ul>
              {!definition.bypassable && <span>Non-bypassable</span>}
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function OperatorWorkspace() {
  const [overview, setOverview] = useState<ProcessOverview | null>(null);
  const [protections, setProtections] = useState<ProtectionSnapshot | null>(null);
  const [selected, setSelected] = useState<AssetFaceplate | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([getOperatorOverview(), getProtectionSnapshot()])
      .then(([nextOverview, nextProtections]) => {
        if (active) {
          setOverview(nextOverview);
          setProtections(nextProtections);
        }
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Operator workspace unavailable");
      });
    return () => { active = false; };
  }, []);

  if (error) return <div role="alert">{error}</div>;
  if (!overview || !protections) return <div>Loading representative operator workspace…</div>;

  return (
    <main aria-labelledby="operator-workspace-heading">
      <header>
        <h2 id="operator-workspace-heading">{overview.title}</h2>
        <p><strong>Synthetic demonstration only.</strong> Not for live control.</p>
      </header>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.8rem" }}>
        {overview.areas.map((area) => (
          <section key={area.area_id} style={cardStyle} aria-labelledby={`${area.area_id}-heading`}>
            <h3 id={`${area.area_id}-heading`}>{area.label}</h3>
            {area.assets.map((asset) => (
              <button
                key={asset.asset_id}
                type="button"
                onClick={() => setSelected(asset)}
                style={{ display: "block", width: "100%", textAlign: "left", marginTop: "0.45rem", padding: "0.65rem" }}
              >
                {asset.label} — {asset.primary_value.value} {asset.primary_value.unit}
              </button>
            ))}
          </section>
        ))}
      </div>
      <ProtectionView snapshot={protections} />
      {selected && <Faceplate asset={selected} onClose={() => setSelected(null)} />}
    </main>
  );
}
