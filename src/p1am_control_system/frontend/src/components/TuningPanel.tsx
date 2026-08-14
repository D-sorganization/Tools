import React from "react";
import { Sliders, Info } from "lucide-react";
import type { MpcSimResult, RoutingConfig, TuningResult } from "../types";

/**
 * PID auto-tuning and MPC-vs-PID simulation tab.
 *
 * Extracted verbatim from App.tsx, which had grown to 1690 lines — 190 over the
 * repo's 1500-line source budget — and mixed this self-contained reference tab
 * in with the live control surfaces. All state and the REST calls stay in App;
 * this component is presentation plus callbacks, so the tuning UI can change
 * without touching the file that owns telemetry and safety.
 */

export interface MpcParams {
  prediction_horizon: number;
  control_horizon: number;
  setpoint: number;
  rho: number;
  process_gain: number;
  process_tau: number;
  process_delay: number;
}

interface Props {
  config: RoutingConfig;
  selectedTuningLoop: number;
  setSelectedTuningLoop: (index: number) => void;
  isTuningMode: boolean;
  tuningStepVal: string;
  setTuningStepVal: (value: string) => void;
  tuningResults: TuningResult | null;
  startTuning: (index: number) => void;
  stepTuning: (index: number, value: number) => void;
  stopTuning: (index: number) => void;
  applyRecommendedGains: (index: number) => void;
  mpcParams: MpcParams;
  setMpcParams: React.Dispatch<React.SetStateAction<MpcParams>>;
  mpcSimData: MpcSimResult | null;
  runMpcSimulation: () => void;
}

export const TuningPanel: React.FC<Props> = ({
  config,
  selectedTuningLoop,
  setSelectedTuningLoop,
  isTuningMode,
  tuningStepVal,
  setTuningStepVal,
  tuningResults,
  startTuning,
  stepTuning,
  stopTuning,
  applyRecommendedGains,
  mpcParams,
  setMpcParams,
  mpcSimData,
  runMpcSimulation,
}) => (
  <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
    {/* PID Loop Tuning Section */}
    <div className="glass-panel">
      <div className="panel-header">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Sliders size={16} color="var(--accent-cyan)" />
          <span>Auto-Tuning & Transient Response Identification</span>
        </div>
        <span className="tooltip-container">
          <Info size={14} color="var(--text-muted)" />
          <span className="tooltip-text">
            Decouples PID loop automatic control, registers step change, and solves First Order Plus Dead Time parameters.
          </span>
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        <div>
          <div className="input-group" style={{ marginBottom: "1rem" }}>
            <label className="input-label">Select Controller Loop</label>
            <select
              className="form-input"
              value={selectedTuningLoop}
              onChange={(e) => setSelectedTuningLoop(Number(e.target.value))}
              disabled={isTuningMode}
            >
              {config.pids.map((_, idx) => (
                <option key={idx} value={idx}>Loop {idx + 1}</option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem" }}>
            {!isTuningMode ? (
              <button
                type="button"
                className="btn btn-primary"
                style={{ flex: 1, backgroundColor: "var(--accent-purple)", borderColor: "var(--accent-purple)" }}
                onClick={() => startTuning(selectedTuningLoop)}
              >
                Start Tuning Mode
              </button>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem", width: "100%" }}>
                <div style={{ padding: "0.75rem", background: "rgba(255, 179, 0, 0.1)", border: "1px solid rgba(255, 179, 0, 0.3)", borderRadius: "4px", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  <strong>Tuning Active:</strong> PID automatic calculations are paused. Setpoint tracking is decoupled.
                </div>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <input
                    type="number"
                    step="1"
                    className="form-input"
                    value={tuningStepVal}
                    onChange={(e) => setTuningStepVal(e.target.value)}
                    placeholder="Step CV"
                    style={{ width: "100px" }}
                  />
                  <button
                    type="button"
                    className="btn"
                    onClick={() => stepTuning(selectedTuningLoop, Number(tuningStepVal))}
                  >
                    Apply Step
                  </button>
                </div>
                <button
                  type="button"
                  className="btn btn-primary"
                  style={{ backgroundColor: "var(--color-error)", borderColor: "var(--color-error)" }}
                  onClick={() => stopTuning(selectedTuningLoop)}
                >
                  Stop Tuning & Solve FOPDT
                </button>
              </div>
            )}
          </div>
        </div>

        <div style={{ borderLeft: "1px solid var(--panel-border)", paddingLeft: "1.5rem" }}>
          <h4 style={{ fontSize: "0.85rem", fontWeight: 700, textTransform: "uppercase", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
            Identification & Recommendations
          </h4>
          {tuningResults ? (
            <div>
              {tuningResults.status === "success" ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                    <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem", borderRadius: "4px" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Process Gain (Kp)</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700 }}>{tuningResults.parameters.kp.toFixed(3)}</div>
                    </div>
                    <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem", borderRadius: "4px" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Time Const (Tau)</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700 }}>{tuningResults.parameters.tau.toFixed(2)} s</div>
                    </div>
                    <div style={{ background: "rgba(255,255,255,0.02)", padding: "0.5rem", borderRadius: "4px", gridColumn: "span 2" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Dead Time (Theta)</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700 }}>{tuningResults.parameters.theta.toFixed(2)} s</div>
                    </div>
                  </div>

                  <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.5rem", marginTop: "0.25rem" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>Cohen-Coon Recommended Gains:</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem" }}>
                      <div>
                        <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Kp</div>
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--accent-cyan)" }}>{tuningResults.recommended_pid.kp.toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Ki</div>
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--accent-purple)" }}>{tuningResults.recommended_pid.ki.toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Kd</div>
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--color-warning)" }}>{tuningResults.recommended_pid.kd.toFixed(2)}</div>
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{ width: "100%", fontSize: "0.75rem", padding: "0.4rem" }}
                      onClick={() => applyRecommendedGains(selectedTuningLoop)}
                    >
                      Load Gains into Controller
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: "0.8rem", color: "var(--color-error)" }}>
                  {tuningResults.message}
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontStyle: "italic" }}>
              No active or past identification results. Start tuning mode and apply a step change to compute model variables.
            </div>
          )}
        </div>
      </div>
    </div>

    {/* MPC & Advanced Control Section */}
    <div className="glass-panel">
      <div className="panel-header">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Sliders size={16} color="var(--accent-magenta)" />
          <span>Model Predictive Control (MPC) Solver Groundwork</span>
        </div>
        <span className="tooltip-container">
          <Info size={14} color="var(--text-muted)" />
          <span className="tooltip-text">
            Run projected gradient descent MPC solver comparisons against standard PID loop control to evaluate dynamic constraint optimization.
          </span>
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "1.5rem" }}>
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
            <div className="input-group">
              <label className="input-label" style={{ fontSize: "0.65rem" }}>Pred Horizon (Np)</label>
              <input
                type="number"
                className="form-input"
                value={mpcParams.prediction_horizon}
                onChange={(e) => setMpcParams({ ...mpcParams, prediction_horizon: Number(e.target.value) })}
              />
            </div>
            <div className="input-group">
              <label className="input-label" style={{ fontSize: "0.65rem" }}>Ctrl Horizon (Nc)</label>
              <input
                type="number"
                className="form-input"
                value={mpcParams.control_horizon}
                onChange={(e) => setMpcParams({ ...mpcParams, control_horizon: Number(e.target.value) })}
              />
            </div>
            <div className="input-group">
              <label className="input-label" style={{ fontSize: "0.65rem" }}>Setpoint SP</label>
              <input
                type="number"
                className="form-input"
                value={mpcParams.setpoint}
                onChange={(e) => setMpcParams({ ...mpcParams, setpoint: Number(e.target.value) })}
              />
            </div>
            <div className="input-group">
              <label className="input-label" style={{ fontSize: "0.65rem" }}>Input Penalty (Rho)</label>
              <input
                type="number"
                step="0.01"
                className="form-input"
                value={mpcParams.rho}
                onChange={(e) => setMpcParams({ ...mpcParams, rho: Number(e.target.value) })}
              />
            </div>
            <div className="input-group">
              <label className="input-label" style={{ fontSize: "0.65rem" }}>Proc Gain (Kp)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={mpcParams.process_gain}
                onChange={(e) => setMpcParams({ ...mpcParams, process_gain: Number(e.target.value) })}
              />
            </div>
            <div className="input-group">
              <label className="input-label" style={{ fontSize: "0.65rem" }}>Proc Tau (s)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={mpcParams.process_tau}
                onChange={(e) => setMpcParams({ ...mpcParams, process_tau: Number(e.target.value) })}
              />
            </div>
          </div>

          <button
            type="button"
            className="btn btn-primary"
            style={{ width: "100%", marginTop: "1rem" }}
            onClick={runMpcSimulation}
          >
            Run Predictive Simulation
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", height: "100%", justifyContent: "center" }}>
          {mpcSimData ? (
            <div style={{ position: "relative", width: "100%", height: "200px" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-secondary)", marginBottom: "0.25rem", textAlign: "center" }}>
                Transient Comparison (PID vs MPC)
              </div>
              {/* Draw SVG Graph comparing the paths */}
              <svg style={{ width: "100%", height: "100%", background: "rgba(0,0,0,0.15)", borderRadius: "4px" }}>
                {/* Grid lines */}
                <line x1="0" y1="50" x2="350" y2="50" stroke="rgba(255,255,255,0.05)" />
                <line x1="0" y1="100" x2="350" y2="100" stroke="rgba(255,255,255,0.05)" />
                <line x1="0" y1="150" x2="350" y2="150" stroke="rgba(255,255,255,0.05)" />

                {/* Setpoint (dashed line) */}
                {(() => {
                  const spY = 200 - (mpcParams.setpoint * 2);
                  return <line x1="0" y1={spY} x2="350" y2={spY} stroke="var(--text-muted)" strokeDasharray="3,3" strokeWidth="1" />;
                })()}

                {/* PID PV path (purple) */}
                {(() => {
                  const points = mpcSimData.time.map((_, idx) => {
                    const x = (idx / (mpcSimData.time.length - 1)) * 340 + 5;
                    const y = 200 - (mpcSimData.pid.pv[idx] * 2);
                    return `${x},${y}`;
                  }).join(" ");
                  return <polyline fill="none" stroke="var(--accent-purple)" strokeWidth="2" points={points} />;
                })()}

                {/* MPC PV path (cyan) */}
                {(() => {
                  const points = mpcSimData.time.map((_, idx) => {
                    const x = (idx / (mpcSimData.time.length - 1)) * 340 + 5;
                    const y = 200 - (mpcSimData.mpc.pv[idx] * 2);
                    return `${x},${y}`;
                  }).join(" ");
                  return <polyline fill="none" stroke="var(--accent-cyan)" strokeWidth="2" points={points} />;
                })()}
              </svg>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.25rem", fontSize: "0.65rem", color: "var(--text-muted)" }}>
                <span>Time: 0s</span>
                <span style={{ display: "flex", gap: "0.75rem" }}>
                  <span style={{ color: "var(--accent-purple)" }}>● PID</span>
                  <span style={{ color: "var(--accent-cyan)" }}>● MPC</span>
                  <span style={{ color: "var(--text-muted)" }}>-- Setpoint</span>
                </span>
                <span>25s</span>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", border: "1px dashed var(--panel-border)", padding: "2rem", borderRadius: "4px", color: "var(--text-muted)", fontSize: "0.8rem" }}>
              Run the predictive simulation to compare tracking trajectories.
            </div>
          )}
        </div>
      </div>
    </div>
  </div>
);
