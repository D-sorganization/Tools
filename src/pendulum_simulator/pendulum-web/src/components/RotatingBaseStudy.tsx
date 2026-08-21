import { useMemo, useState } from "react";

import {
  ROTATING_BASE_BOUNDARIES,
  ROTATING_BASE_MODEL_TIER,
  ROTATING_BASE_SOURCE_REVISION,
  ROTATING_BASE_STUDY,
  registeredCase,
  type MatchingRule,
  type TorsoProfile,
} from "../rotatingBaseStudy";
import {
  ROTATING_BASE_RUN_CATALOG,
  registeredRun,
} from "../rotatingBaseRunCatalog";
import "./RotatingBaseStudy.css";
import { RotatingBaseTraceCharts } from "./RotatingBaseTraceCharts";

const PROFILE_LABELS: Record<TorsoProfile, string> = {
  accelerate: "Accelerate (+55 N·m)",
  constant_rate: "Zero Torso Command (0 N·m)",
  decelerate: "Decelerate (−55 N·m)",
};
const RULE_LABELS: Record<MatchingRule, string> = {
  relative_club_rate: "Relative Club Rate",
  absolute_club_rate: "Absolute Club Rate",
};
const METRICS = [
  ["impact_speed_m_s", "Delivery Speed", "m/s"],
  ["contact_work_on_club_j", "Contact Work on Club", "J"],
  ["braking_grip_work_j", "Braking Grip Work", "J"],
  ["force_couple_work_j", "Force-Couple Work", "J"],
  ["negative_along_path_impulse_ns", "Negative Along-Path Impulse", "N·s"],
  ["bilateral_wrist_work_j", "Bilateral Wrist Work", "J"],
  ["total_control_work_j", "Total Control Work", "J"],
  ["distal_energy_gain_j", "Distal Energy Gain", "J"],
  ["peak_grip_force_n", "Peak Grip Force", "N"],
  ["maximum_constraint_residual_m", "Position Closure", "m"],
  ["maximum_velocity_constraint_residual_m_s", "Velocity Closure", "m/s"],
  ["maximum_contact_power_identity_residual_w", "Power Identity Residual", "W"],
  ["work_energy_closure_j", "Work–Energy Closure", "J"],
] as const;

export function RotatingBaseStudy() {
  const [profile, setProfile] = useState<TorsoProfile>("accelerate");
  const [matchingRule, setMatchingRule] =
    useState<MatchingRule>("relative_club_rate");
  const [torsoRate, setTorsoRate] = useState(1.5);
  const selected = useMemo(
    () => registeredCase(profile, matchingRule, torsoRate),
    [profile, matchingRule, torsoRate],
  );
  const selectedRun = useMemo(
    () => registeredRun(ROTATING_BASE_RUN_CATALOG, profile, matchingRule, torsoRate),
    [profile, matchingRule, torsoRate],
  );
  const killswitch = ROTATING_BASE_STUDY.same_state_killswitch;

  function exportSelectedCase() {
    const payload = JSON.stringify(
      {
        schema_id: "swing-sim/rotating-base-governed-export",
        schema_version: 1,
        source_revision: ROTATING_BASE_SOURCE_REVISION,
        model_tier: ROTATING_BASE_MODEL_TIER,
        boundaries: ROTATING_BASE_BOUNDARIES,
        study_sha256: ROTATING_BASE_RUN_CATALOG.study_sha256,
        run: selectedRun,
      },
      null,
      2,
    );
    const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `rotating_base_case_${selected.case_index}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="rotating-base-study" aria-labelledby="rotating-base-title">
      <header>
        <div>
          <h2 id="rotating-base-title">Qualified Rotating-Base Study</h2>
          <p>
            Evidence browser for {ROTATING_BASE_STUDY.attempted_case_count} retained
            registered rows ({ROTATING_BASE_STUDY.valid_case_count} valid). The web
            surface reads the same immutable authority as the desktop provider.
          </p>
          <p>
            Model tier: <code>{ROTATING_BASE_MODEL_TIER}</code>; source revision:{" "}
            <code>{ROTATING_BASE_SOURCE_REVISION.slice(0, 12)}</code>.
          </p>
        </div>
        <span className={selected.valid ? "valid" : "adverse"}>
          {selected.valid ? "Valid Registered Row" : "Invalid/Adverse Retained"}
        </span>
      </header>
      <div className="rotating-base-study__scope">
        <strong>Scientific Scope:</strong> nonanatomical finite-inertia model
        coordinate; no governed human validation; no coaching recommendation.
      </div>
      <div className="rotating-base-study__controls">
        <label>
          Torso Program
          <select value={profile} onChange={(event) => setProfile(event.target.value as TorsoProfile)}>
            {Object.entries(PROFILE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <label>
          Matching Rule
          <select value={matchingRule} onChange={(event) => setMatchingRule(event.target.value as MatchingRule)}>
            {Object.entries(RULE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <label>
          Initial Torso Rate
          <select value={torsoRate} onChange={(event) => setTorsoRate(Number(event.target.value))}>
            {[1.5, 3.5, 5.5].map((rate) => (
              <option key={rate} value={rate}>{rate} rad/s</option>
            ))}
          </select>
        </label>
        <button className="btn btn-secondary" type="button" onClick={exportSelectedCase}>
          Export Governed Row
        </button>
      </div>
      {!selected.valid && (
        <p className="rotating-base-study__exclusion">
          Exclusion: {selected.exclusion_reasons.join(", ")}
        </p>
      )}
      <dl className="rotating-base-study__metrics">
        {METRICS.map(([key, label, unit]) => (
          <div key={key}>
            <dt>{label}</dt>
            <dd>{selected[key].toPrecision(6)} {unit}</dd>
          </div>
        ))}
      </dl>
      <div className="rotating-base-study__killswitches">
        <h3>Exact Same-State Killswitches</h3>
        <p>Branch time: {killswitch.branch_time_s} s</p>
        <ul>
          {Object.entries(killswitch.channels).map(([name, channel]) => (
            <li key={name}>
              <strong>{name.replace(/_/g, " ")}:</strong> Δdelivery speed {channel.delivery_speed_difference_m_s.toPrecision(4)} m/s; Δcontact work {channel.post_branch_contact_work_difference_j.toPrecision(4)} J
            </li>
          ))}
        </ul>
      </div>
      <RotatingBaseTraceCharts run={selectedRun} />
    </section>
  );
}
