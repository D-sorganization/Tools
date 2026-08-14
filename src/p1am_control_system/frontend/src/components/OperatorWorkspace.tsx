import { useEffect, useState } from "react";
import {
  getOperatorOverview,
  getProtectionSnapshot,
  getRepresentativeAssetHealth,
  getShiftEntries,
  getProductStatus,
  getRepresentativeAdvisory,
  recordAdvisoryDisposition,
} from "../api/endpoints";
import type {
  AssetFaceplate,
  AssetHealthReport,
  ProcessOverview,
  ProtectionSnapshot,
  ProductStatus,
  AdvisoryResult,
  ShiftEntry,
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
  const [assetHealth, setAssetHealth] = useState<AssetHealthReport | null>(null);
  const [shiftEntries, setShiftEntries] = useState<ShiftEntry[]>([]);
  const [productStatus, setProductStatus] = useState<ProductStatus | null>(null);
  const [advisory, setAdvisory] = useState<AdvisoryResult | null>(null);
  const [dispositionStatus, setDispositionStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([
      getOperatorOverview(),
      getProtectionSnapshot(),
      getRepresentativeAssetHealth(),
      getShiftEntries(),
      getProductStatus(),
      getRepresentativeAdvisory(),
    ])
      .then(([nextOverview, nextProtections, nextHealth, nextEntries, nextProduct, nextAdvisory]) => {
        if (active) {
          setOverview(nextOverview);
          setProtections(nextProtections);
          setAssetHealth(nextHealth);
          setShiftEntries(nextEntries);
          setProductStatus(nextProduct);
          setAdvisory(nextAdvisory);
        }
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Operator workspace unavailable");
      });
    return () => { active = false; };
  }, []);

  if (error) return <div role="alert">{error}</div>;
  if (!overview || !protections || !assetHealth || !productStatus || !advisory) return <div>Loading representative operator workspace…</div>;

  const disposition = async (
    decision: "accepted_for_review" | "rejected" | "deferred",
  ) => {
    try {
      const record = await recordAdvisoryDisposition(
        advisory.advisory_id,
        decision,
        "Operator disposition from representative advisory workspace",
      );
      setDispositionStatus(`${record.decision.replace(/_/g, " ")} by ${record.actor}; no control value applied.`);
    } catch (reason: unknown) {
      setDispositionStatus(reason instanceof Error ? reason.message : "Disposition failed");
    }
  };

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
      <section aria-labelledby="maintenance-heading" style={{ marginTop: "1rem" }}>
        <h3 id="maintenance-heading">Asset health & maintenance</h3>
        <p>
          {assetHealth.asset_id}: {assetHealth.counters.runtime_seconds} runtime seconds, {assetHealth.counters.start_count} starts.
          Advisories are maintenance records, never authoritative trips.
        </p>
        <ul>
          {assetHealth.advisories.map((advisory) => (
            <li key={advisory.code}><strong>{advisory.code.replace(/_/g, " ")}</strong>: {advisory.detail}</li>
          ))}
        </ul>
      </section>
      <section aria-labelledby="investigation-heading" style={{ marginTop: "1rem" }}>
        <h3 id="investigation-heading">Investigations & reporting</h3>
        <p>
          Saved synthetic investigations retain query bounds, tag metadata, transformations,
          charts, annotations, event context, explicit bad-data handling, and export checksums.
        </p>
      </section>
      <section aria-labelledby="handover-heading" style={{ marginTop: "1rem" }}>
        <h3 id="handover-heading">Shift log & handover</h3>
        {shiftEntries.length === 0 ? (
          <p>No synthetic handover entries.</p>
        ) : (
          <ul>{shiftEntries.map((entry) => <li key={entry.entry_id}>{entry.summary}</li>)}</ul>
        )}
        <p>Signed entries are append-only; receiving operators acknowledge unresolved work explicitly.</p>
      </section>
      <section aria-labelledby="product-heading" style={{ marginTop: "1rem" }}>
        <h3 id="product-heading">Reusable control product</h3>
        <p>Procedure state: <strong>{productStatus.procedure_state}</strong>. Simulator-only transitions are bounded and attributable.</p>
        <ul>
          {productStatus.connectors.map((connector) => {
            const samples = Object.values(productStatus.samples).filter(
              (sample) => sample.connector_id === connector.connector_id,
            );
            const quality = samples.some((sample) => sample.quality === "bad") ? "bad" : "good";
            return <li key={connector.connector_id}>{connector.connector_id}: {quality}</li>;
          })}
        </ul>
        <p>
          Notifications escalate from {productStatus.notification_policy.primary_recipient} to {productStatus.notification_policy.escalation_recipient}; deliveries are delayed, suppressed, rate-limited, redacted, and audited.
        </p>
        <p>
          Recovery objectives: RTO {productStatus.availability.recovery_time_objective_seconds}s / RPO {productStatus.availability.recovery_point_objective_seconds}s. One command authority; energizing commands fail closed without the HMI.
        </p>
      </section>
      <section aria-labelledby="advisory-heading" style={{ marginTop: "1rem" }}>
        <h3 id="advisory-heading">Advisory optimization &amp; digital twin</h3>
        <p><strong>Review only. No authoritative write path.</strong> Synthetic demonstration; not validated against a plant.</p>
        <p>
          Model {advisory.model.model_id} v{advisory.model.version}; dataset {advisory.data.dataset_id}.
        </p>
        <p>
          Recommendation: {advisory.recommended_setpoint} {advisory.constraints.unit} within {advisory.constraints.minimum}–{advisory.constraints.maximum}.
          {" "}{advisory.confidence.level * 100}% confidence: {advisory.confidence.lower}–{advisory.confidence.upper}.
        </p>
        <p>Replay verified: {String(advisory.replay.verified)}; result checksum {advisory.replay.result_sha256.slice(0, 12)}…</p>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={() => void disposition("accepted_for_review")}>Accept for review</button>
          <button type="button" onClick={() => void disposition("deferred")}>Defer</button>
          <button type="button" onClick={() => void disposition("rejected")}>Reject</button>
        </div>
        {dispositionStatus && <p role="status">{dispositionStatus}</p>}
      </section>
      {selected && <Faceplate asset={selected} onClose={() => setSelected(null)} />}
    </main>
  );
}
