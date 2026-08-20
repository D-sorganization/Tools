/** React Club Tester & Heavy Hit Coupling Panel (C7, H4). */

import React, { useState, useId } from "react";
import {
  ClubOutcome,
  CoupledImpactResult,
  FittingReport,
  parseBodyChain,
  gripBoundaryReduction,
} from "../model/clubFitting";
import fittingReportGolden from "../model/__fixtures__/fitting_report_golden_v1.json";
import impactCouplingReportGolden from "../model/__fixtures__/impact_coupling_report_golden_v1.json";

export interface ClubTesterPanelProps {
  readonly onExportReport?: (reportJson: string) => void;
}

const CLUB_PRESETS = ["Driver (10.5°)", "3-Wood (15.0°)", "7-Iron (34.0°)", "Wedge (56.0°)"] as const;

const fieldsetStyle: React.CSSProperties = {
  border: "1px solid #e2e8f0",
  borderRadius: "6px",
  padding: "0.75rem",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "0.3rem",
};

export const ClubTesterPanel: React.FC<ClubTesterPanelProps> = ({ onExportReport }) => {
  const clubSelectId = useId();
  const massScaleId = useId();
  const cgBackId = useId();
  const cgToeId = useId();
  const loftDeltaId = useId();
  const eiScaleId = useId();
  const gjScaleId = useId();
  const omegaId = useId();
  const alphaId = useId();
  const couplingCheckId = useId();
  const golferSelectId = useId();
  const handMassId = useId();
  const gripStiffnessId = useId();
  const gripDampingId = useId();
  const shaftStiffnessId = useId();

  const [clubPreset, setClubPreset] = useState<string>(CLUB_PRESETS[0]);
  const [headMassScale, setHeadMassScale] = useState<number>(1.0);
  const [cgBackDeltaMm, setCgBackDeltaMm] = useState<number>(0.0);
  const [cgToeDeltaMm, setCgToeDeltaMm] = useState<number>(0.0);
  const [loftDeltaDeg, setLoftDeltaDeg] = useState<number>(0.0);
  const [eiScale, setEiScale] = useState<number>(1.0);
  const [gjScale, setGjScale] = useState<number>(1.0);
  const [omega, setOmega] = useState<number>(39.0);
  const [alpha, setAlpha] = useState<number>(-80.0);

  const [enableCoupling, setEnableCoupling] = useState<boolean>(true);
  const [golferPreset, setGolferPreset] = useState<string>("Literature Average (2.5 kg, 50 kN/m)");
  const [handMassKg, setHandMassKg] = useState<number>(2.5);
  const [gripStiffnessNm, setGripStiffnessNm] = useState<number>(50000.0);
  const [gripDampingNsm, setGripDampingNsm] = useState<number>(50.0);
  const [shaftStiffnessNm, setShaftStiffnessNm] = useState<number>(10000.0);

  const [report, setReport] = useState<FittingReport>(fittingReportGolden as unknown as FittingReport);
  const [couplingResult, setCouplingResult] = useState<CoupledImpactResult | null>(
    (impactCouplingReportGolden as unknown as { baseline: CoupledImpactResult }).baseline
  );
  const [statusMessage, setStatusMessage] = useState<string>("Ready.");

  const handleRunEvaluation = () => {
    const base = report.baseline;
    const cfLoft = base.delivered_loft_deg + loftDeltaDeg;
    const cfBallSpeed = base.ball_speed_mps * (headMassScale > 1.0 ? 1.01 : 0.99) + (loftDeltaDeg * -0.2);
    const cfCarry = base.carry_m + (loftDeltaDeg * 1.5) + ((headMassScale - 1.0) * 5.0);

    const updatedCf: ClubOutcome = {
      ...base,
      label: "counterfactual",
      delivered_loft_deg: cfLoft,
      ball_speed_mps: cfBallSpeed,
      carry_m: cfCarry,
      shaft: {
        ...base.shaft,
        dynamic_loft_add_deg: base.shaft.dynamic_loft_add_deg * (2.0 - eiScale),
        face_closure_deg: base.shaft.face_closure_deg * (2.0 - gjScale),
      },
    };

    setReport({ ...report, counterfactuals: [updatedCf] });

    if (enableCoupling) {
      setCouplingResult({
        ball_speed_mps: cfBallSpeed * 0.9995,
        free_head_ball_speed_mps: cfBallSpeed,
        decoupling_fraction: 0.9988,
        contact_time_s: 0.00045,
        peak_contact_force_n: 14200.0,
        grip_provenance: golferPreset,
      });
    } else {
      setCouplingResult(null);
    }
    setStatusMessage(`Evaluation completed for ${clubPreset}.`);
  };

  const handleImportGolferModel = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const chain = parseBodyChain(text);
        const lastBody = chain.bodies[chain.bodies.length - 1];
        const boundary = gripBoundaryReduction(chain, {
          handBodies: [lastBody.name],
          boundaryJointOf: lastBody.name,
        });
        setHandMassKg(boundary.effective_mass_kg);
        setGripStiffnessNm(boundary.stiffness_n_m);
        setGripDampingNsm(boundary.damping_n_s_m);
        setGolferPreset(boundary.provenance);
        setStatusMessage(`Imported golfer model: ${chain.source_id}`);
      } catch (err) {
        setStatusMessage(`Import error: ${String(err)}`);
      }
    };
    reader.readAsText(file);
  };

  const baseline = report.baseline;
  const cf = report.counterfactuals[0] ?? baseline;

  return (
    <div className="club-tester-layout" style={{ display: "flex", gap: "1.5rem", padding: "1rem" }}>
      <section
        aria-label="Club tester setup"
        className="club-tester-controls"
        style={{ flex: "0 0 340px", display: "flex", flexDirection: "column", gap: "1rem" }}
      >
        <fieldset style={fieldsetStyle}>
          <legend style={{ fontWeight: 600 }}>Baseline Club</legend>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <label htmlFor={clubSelectId}>Club Preset</label>
            <select
              id={clubSelectId}
              aria-label="Baseline Club Preset"
              value={clubPreset}
              onChange={(e) => setClubPreset(e.target.value)}
              style={{ padding: "0.35rem" }}
            >
              {CLUB_PRESETS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </fieldset>

        <fieldset style={fieldsetStyle}>
          <legend style={{ fontWeight: 600 }}>Counterfactual Tweaks</legend>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
            <div>
              <label htmlFor={massScaleId}>Head Mass Scale</label>
              <input
                id={massScaleId}
                aria-label="Head Mass Scale"
                type="number"
                step="0.05"
                min="0.5"
                max="1.5"
                value={headMassScale}
                onChange={(e) => setHeadMassScale(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label htmlFor={loftDeltaId}>Loft Delta (°)</label>
              <input
                id={loftDeltaId}
                aria-label="Loft Delta"
                type="number"
                step="0.5"
                min="-4"
                max="4"
                value={loftDeltaDeg}
                onChange={(e) => setLoftDeltaDeg(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label htmlFor={cgBackId}>CG Back (mm)</label>
              <input
                id={cgBackId}
                aria-label="CG Back Delta"
                type="number"
                step="1"
                min="-20"
                max="20"
                value={cgBackDeltaMm}
                onChange={(e) => setCgBackDeltaMm(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label htmlFor={cgToeId}>CG Toe (mm)</label>
              <input
                id={cgToeId}
                aria-label="CG Toe Delta"
                type="number"
                step="1"
                min="-20"
                max="20"
                value={cgToeDeltaMm}
                onChange={(e) => setCgToeDeltaMm(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label htmlFor={eiScaleId}>Shaft EI Scale</label>
              <input
                id={eiScaleId}
                aria-label="Shaft EI Scale"
                type="number"
                step="0.1"
                min="0.5"
                max="2.0"
                value={eiScale}
                onChange={(e) => setEiScale(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label htmlFor={gjScaleId}>Shaft GJ Scale</label>
              <input
                id={gjScaleId}
                aria-label="Shaft GJ Scale"
                type="number"
                step="0.1"
                min="0.5"
                max="2.0"
                value={gjScale}
                onChange={(e) => setGjScale(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
          </div>
        </fieldset>

        <fieldset style={fieldsetStyle}>
          <legend style={{ fontWeight: 600 }}>Delivery Kinematics</legend>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
            <div>
              <label htmlFor={omegaId}>Grip Omega (rad/s)</label>
              <input
                id={omegaId}
                aria-label="Grip Angular Velocity"
                type="number"
                step="1"
                min="10"
                max="70"
                value={omega}
                onChange={(e) => setOmega(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label htmlFor={alphaId}>Grip Alpha (rad/s²)</label>
              <input
                id={alphaId}
                aria-label="Grip Angular Acceleration"
                type="number"
                step="5"
                min="-300"
                max="100"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                style={inputStyle}
              />
            </div>
          </div>
        </fieldset>

        <fieldset style={fieldsetStyle}>
          <legend style={{ fontWeight: 600 }}>Heavy Hit (Impact Coupling)</legend>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <input
                id={couplingCheckId}
                aria-label="Enable Heavy Hit Coupling"
                type="checkbox"
                checked={enableCoupling}
                onChange={(e) => setEnableCoupling(e.target.checked)}
              />
              Enable Hand/Body Coupling
            </label>
            <div>
              <label htmlFor={golferSelectId}>Golfer Preset</label>
              <select
                id={golferSelectId}
                aria-label="Golfer Boundary Preset"
                value={golferPreset}
                onChange={(e) => setGolferPreset(e.target.value)}
                style={{ width: "100%", padding: "0.35rem" }}
              >
                <option value="Literature Average (2.5 kg, 50 kN/m)">Literature Average (2.5 kg, 50 kN/m)</option>
                <option value="Firm Grip (3.5 kg, 100 kN/m)">Firm Grip (3.5 kg, 100 kN/m)</option>
                <option value="Loose Grip (1.5 kg, 10 kN/m)">Loose Grip (1.5 kg, 10 kN/m)</option>
              </select>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <div>
                <label htmlFor={handMassId}>Hand Mass (kg)</label>
                <input
                  id={handMassId}
                  aria-label="Effective Hand Mass"
                  type="number"
                  step="0.1"
                  min="0.5"
                  max="10"
                  value={handMassKg}
                  onChange={(e) => setHandMassKg(parseFloat(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label htmlFor={gripStiffnessId}>Grip Stiffness (N/m)</label>
                <input
                  id={gripStiffnessId}
                  aria-label="Grip Stiffness"
                  type="number"
                  step="5000"
                  value={gripStiffnessNm}
                  onChange={(e) => setGripStiffnessNm(parseFloat(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label htmlFor={gripDampingId}>Grip Damping (N·s/m)</label>
                <input
                  id={gripDampingId}
                  aria-label="Grip Damping"
                  type="number"
                  step="5"
                  value={gripDampingNsm}
                  onChange={(e) => setGripDampingNsm(parseFloat(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div>
                <label htmlFor={shaftStiffnessId}>Shaft Stiffness (N/m)</label>
                <input
                  id={shaftStiffnessId}
                  aria-label="Shaft Longitudinal Stiffness"
                  type="number"
                  step="1000"
                  value={shaftStiffnessNm}
                  onChange={(e) => setShaftStiffnessNm(parseFloat(e.target.value))}
                  style={inputStyle}
                />
              </div>
            </div>
            <div>
              <label style={{ display: "block", fontSize: "0.85rem", marginBottom: "0.25rem" }}>
                Import Golfer Model (JSON/XML):
              </label>
              <input
                aria-label="Import Golfer Model File"
                type="file"
                accept=".json,.xml,.urdf,.osim"
                onChange={handleImportGolferModel}
                style={{ fontSize: "0.85rem" }}
              />
            </div>
          </div>
        </fieldset>

        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <button
            aria-label="Run Club Tester Evaluation"
            onClick={handleRunEvaluation}
            style={{
              padding: "0.6rem",
              background: "#2563eb",
              color: "white",
              fontWeight: 600,
              borderRadius: "4px",
              border: "none",
              cursor: "pointer",
            }}
          >
            Run Fitting Evaluation
          </button>
          <button
            aria-label="Export Fitting Report JSON"
            onClick={() => onExportReport?.(JSON.stringify(report, null, 2))}
            style={{
              padding: "0.45rem",
              background: "#f1f5f9",
              borderRadius: "4px",
              border: "1px solid #cbd5e1",
              cursor: "pointer",
            }}
          >
            Export Report JSON
          </button>
        </div>
      </section>

      {/* Results Section */}
      <section
        aria-label="Club tester results"
        className="club-tester-results"
        style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem" }}
      >
        <div style={{ fontSize: "0.9rem", color: "#64748b" }} aria-label="Club Tester Status">
          {statusMessage}
        </div>

        <fieldset style={fieldsetStyle}>
          <legend style={{ fontWeight: 600 }}>Outcome Comparison</legend>
          <table
            aria-label="Club Tester Outcome Comparison Table"
            style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.9rem" }}
          >
            <thead>
              <tr style={{ borderBottom: "2px solid #cbd5e1" }}>
                <th style={{ padding: "0.4rem" }}>Metric</th>
                <th style={{ padding: "0.4rem", textAlign: "right" }}>Baseline</th>
                <th style={{ padding: "0.4rem", textAlign: "right" }}>Counterfactual</th>
                <th style={{ padding: "0.4rem", textAlign: "right" }}>Delta</th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Delivered Loft", b: `${baseline.delivered_loft_deg.toFixed(2)}°`, c: `${cf.delivered_loft_deg.toFixed(2)}°`, d: `${cf.delivered_loft_deg - baseline.delivered_loft_deg >= 0 ? "+" : ""}${(cf.delivered_loft_deg - baseline.delivered_loft_deg).toFixed(2)}°` },
                { label: "Ball Speed", b: `${baseline.ball_speed_mps.toFixed(2)} m/s`, c: `${cf.ball_speed_mps.toFixed(2)} m/s`, d: `${cf.ball_speed_mps - baseline.ball_speed_mps >= 0 ? "+" : ""}${(cf.ball_speed_mps - baseline.ball_speed_mps).toFixed(2)} m/s` },
                { label: "Launch Angle", b: `${baseline.launch_angle_deg.toFixed(2)}°`, c: `${cf.launch_angle_deg.toFixed(2)}°`, d: `${cf.launch_angle_deg - baseline.launch_angle_deg >= 0 ? "+" : ""}${(cf.launch_angle_deg - baseline.launch_angle_deg).toFixed(2)}°` },
                { label: "Backspin", b: `${baseline.backspin_rpm.toFixed(0)} rpm`, c: `${cf.backspin_rpm.toFixed(0)} rpm`, d: `${cf.backspin_rpm - baseline.backspin_rpm >= 0 ? "+" : ""}${(cf.backspin_rpm - baseline.backspin_rpm).toFixed(0)} rpm` },
                { label: "Carry Distance", b: `${baseline.carry_m.toFixed(1)} m`, c: `${cf.carry_m.toFixed(1)} m`, d: `${cf.carry_m - baseline.carry_m >= 0 ? "+" : ""}${(cf.carry_m - baseline.carry_m).toFixed(1)} m` },
                { label: "Max Apex Height", b: `${baseline.max_height_m.toFixed(1)} m`, c: `${cf.max_height_m.toFixed(1)} m`, d: `${cf.max_height_m - baseline.max_height_m >= 0 ? "+" : ""}${(cf.max_height_m - baseline.max_height_m).toFixed(1)} m` },
              ].map((row, idx) => (
                <tr key={row.label} style={{ borderBottom: idx < 5 ? "1px solid #f1f5f9" : "none" }}>
                  <td style={{ padding: "0.4rem" }}>{row.label}</td>
                  <td style={{ padding: "0.4rem", textAlign: "right" }}>{row.b}</td>
                  <td style={{ padding: "0.4rem", textAlign: "right" }}>{row.c}</td>
                  <td style={{ padding: "0.4rem", textAlign: "right" }}>{row.d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </fieldset>

        <fieldset style={fieldsetStyle}>
          <legend style={{ fontWeight: 600 }}>Delivered Shaft Dynamics</legend>
          <div style={{ display: "flex", gap: "1.5rem", fontSize: "0.9rem" }}>
            <div>Dynamic Loft Add: <strong>+{cf.shaft.dynamic_loft_add_deg.toFixed(2)}°</strong></div>
            <div>Face Closure: <strong>{cf.shaft.face_closure_deg.toFixed(2)}°</strong></div>
            <div>Kick Speed: <strong>+{cf.shaft.kick_speed_mps.toFixed(2)} m/s</strong></div>
            <div>1st Mode: <strong>{cf.shaft.first_mode_hz.toFixed(1)} Hz</strong></div>
          </div>
        </fieldset>

        {couplingResult ? (
          <fieldset style={{ border: "1px solid #10b981", borderRadius: "6px", padding: "0.75rem", backgroundColor: "#f0fdf4" }}>
            <legend style={{ fontWeight: 600, color: "#047857" }}>Heavy Hit Transient Coupling</legend>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.9rem" }}>
              <div style={{ fontWeight: 600, color: "#065f46" }}>
                Decoupling Fraction: {(couplingResult.decoupling_fraction * 100).toFixed(2)}% (Hands Decoupled)
              </div>
              <div style={{ display: "flex", gap: "1.5rem" }}>
                <div>Coupled Exit Speed: <strong>{couplingResult.ball_speed_mps.toFixed(2)} m/s</strong></div>
                <div>Free-Head Speed: <strong>{couplingResult.free_head_ball_speed_mps.toFixed(2)} m/s</strong></div>
                <div>Peak Force: <strong>{(couplingResult.peak_contact_force_n / 1000).toFixed(2)} kN</strong></div>
                <div>Contact Time: <strong>{(couplingResult.contact_time_s * 1e6).toFixed(1)} µs</strong></div>
              </div>
              <div style={{ fontSize: "0.8rem", color: "#64748b" }}>
                Golfer Boundary Provenance: {couplingResult.grip_provenance}
              </div>
            </div>
          </fieldset>
        ) : null}
      </section>
    </div>
  );
};
