import React from "react";
import { Info, Sliders } from "lucide-react";
import type { MpcSimResult, PIDConfig, TuningResult } from "../types";

type MpcParams = {
  prediction_horizon: number;
  control_horizon: number;
  setpoint: number;
  rho: number;
  process_gain: number;
  process_tau: number;
  process_delay: number;
};

type TuningPanelProps = {
  pids: PIDConfig[];
  selectedTuningLoop: number;
  isTuningMode: boolean;
  tuningStepVal: string;
  tuningResults: TuningResult | null;
  mpcParams: MpcParams;
  mpcSimData: MpcSimResult | null;
  onSelectedTuningLoopChange: (index: number) => void;
  onTuningStepValChange: (value: string) => void;
  onMpcParamsChange: (params: MpcParams) => void;
  onStartTuning: (index: number) => void;
  onStepTuning: (index: number, value: number) => void;
  onStopTuning: (index: number) => void;
  onApplyRecommendedGains: (index: number) => void;
  onRunMpcSimulation: () => void;
};

export const TuningPanel: React.FC<TuningPanelProps> = ({
  pids,
  selectedTuningLoop,
  isTuningMode,
  tuningStepVal,
  tuningResults,
  mpcParams,
  mpcSimData,
  onSelectedTuningLoopChange,
  onTuningStepValChange,
  onMpcParamsChange,
  onStartTuning,
  onStepTuning,
  onStopTuning,
  onApplyRecommendedGains,
  onRunMpcSimulation,
}) => (
  <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
    <div className="glass-panel">
      <div className="panel-header">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Sliders size={16} color="var(--accent-cyan)" />
          <span>Auto-Tuning & Transient Response Identification</span>
        </div>
        <span className="tooltip-container">
          <Info size={14} color="var(--text-muted)" />
          <span className="tooltip-text">
            Decouples PID loop automatic control, registers step change, and
            solves First Order Plus Dead Time parameters.
          </span>
        </span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "1.5rem",
        }}
      >
        <div>
          <div className="input-group" style={{ marginBottom: "1rem" }}>
            <label className="input-label">Select Controller Loop</label>
            <select
              className="form-input"
              value={selectedTuningLoop}
              onChange={(e) =>
                onSelectedTuningLoopChange(Number(e.target.value))
              }
              disabled={isTuningMode}
            >
              {pids.map((_, idx) => (
                <option key={idx} value={idx}>
                  Loop {idx + 1}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem" }}>
            {!isTuningMode ? (
              <button
                type="button"
                className="btn btn-primary"
                style={{
                  flex: 1,
                  backgroundColor: "var(--accent-purple)",
                  borderColor: "var(--accent-purple)",
                }}
                onClick={() => onStartTuning(selectedTuningLoop)}
              >
                Start Tuning Mode
              </button>
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "1rem",
                  width: "100%",
                }}
              >
                <div
                  style={{
                    padding: "0.75rem",
                    background: "rgba(255, 179, 0, 0.1)",
                    border: "1px solid rgba(255, 179, 0, 0.3)",
                    borderRadius: "4px",
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  <strong>Tuning Active:</strong> PID automatic calculations are
                  paused. Setpoint tracking is decoupled.
                </div>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <input
                    type="number"
                    step="1"
                    className="form-input"
                    value={tuningStepVal}
                    onChange={(e) => onTuningStepValChange(e.target.value)}
                    placeholder="Step CV"
                    style={{ width: "100px" }}
                  />
                  <button
                    type="button"
                    className="btn"
                    onClick={() =>
                      onStepTuning(selectedTuningLoop, Number(tuningStepVal))
                    }
                  >
                    Apply Step
                  </button>
                </div>
                <button
                  type="button"
                  className="btn btn-primary"
                  style={{
                    backgroundColor: "var(--color-error)",
                    borderColor: "var(--color-error)",
                  }}
                  onClick={() => onStopTuning(selectedTuningLoop)}
                >
                  Stop Tuning & Solve FOPDT
                </button>
              </div>
            )}
          </div>
        </div>

        <div
          style={{
            borderLeft: "1px solid var(--panel-border)",
            paddingLeft: "1.5rem",
          }}
        >
          <h4
            style={{
              fontSize: "0.85rem",
              fontWeight: 700,
              textTransform: "uppercase",
              marginBottom: "0.75rem",
              color: "var(--text-secondary)",
            }}
          >
            Identification & Recommendations
          </h4>
          {tuningResults ? (
            <div>
              {tuningResults.status === "success" ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.75rem",
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "0.5rem",
                    }}
                  >
                    <ResultMetric
                      label="Process Gain (Kp)"
                      value={tuningResults.parameters.kp.toFixed(3)}
                    />
                    <ResultMetric
                      label="Time Const (Tau)"
                      value={`${tuningResults.parameters.tau.toFixed(2)} s`}
                    />
                    <ResultMetric
                      label="Dead Time (Theta)"
                      value={`${tuningResults.parameters.theta.toFixed(2)} s`}
                      wide
                    />
                  </div>

                  <div
                    style={{
                      borderTop: "1px solid var(--panel-border)",
                      paddingTop: "0.5rem",
                      marginTop: "0.25rem",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.7rem",
                        color: "var(--text-muted)",
                        marginBottom: "0.35rem",
                      }}
                    >
                      Cohen-Coon Recommended Gains:
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr 1fr",
                        gap: "0.5rem",
                        marginBottom: "0.75rem",
                      }}
                    >
                      <GainMetric
                        label="Kp"
                        color="var(--accent-cyan)"
                        value={tuningResults.recommended_pid.kp.toFixed(2)}
                      />
                      <GainMetric
                        label="Ki"
                        color="var(--accent-purple)"
                        value={tuningResults.recommended_pid.ki.toFixed(2)}
                      />
                      <GainMetric
                        label="Kd"
                        color="var(--color-warning)"
                        value={tuningResults.recommended_pid.kd.toFixed(2)}
                      />
                    </div>
                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{
                        width: "100%",
                        fontSize: "0.75rem",
                        padding: "0.4rem",
                      }}
                      onClick={() =>
                        onApplyRecommendedGains(selectedTuningLoop)
                      }
                    >
                      Load Gains into Controller
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  style={{ fontSize: "0.8rem", color: "var(--color-error)" }}
                >
                  {tuningResults.message}
                </div>
              )}
            </div>
          ) : (
            <div
              style={{
                fontSize: "0.8rem",
                color: "var(--text-muted)",
                fontStyle: "italic",
              }}
            >
              No active or past identification results. Start tuning mode and
              apply a step change to compute model variables.
            </div>
          )}
        </div>
      </div>
    </div>

    <div className="glass-panel">
      <div className="panel-header">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Sliders size={16} color="var(--accent-magenta)" />
          <span>Model Predictive Control (MPC) Solver Groundwork</span>
        </div>
        <span className="tooltip-container">
          <Info size={14} color="var(--text-muted)" />
          <span className="tooltip-text">
            Run projected gradient descent MPC solver comparisons against
            standard PID loop control to evaluate dynamic constraint
            optimization.
          </span>
        </span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1.5fr",
          gap: "1.5rem",
        }}
      >
        <div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0.5rem",
            }}
          >
            <MpcNumberInput
              label="Pred Horizon (Np)"
              value={mpcParams.prediction_horizon}
              onChange={(value) =>
                onMpcParamsChange({ ...mpcParams, prediction_horizon: value })
              }
            />
            <MpcNumberInput
              label="Ctrl Horizon (Nc)"
              value={mpcParams.control_horizon}
              onChange={(value) =>
                onMpcParamsChange({ ...mpcParams, control_horizon: value })
              }
            />
            <MpcNumberInput
              label="Setpoint SP"
              value={mpcParams.setpoint}
              onChange={(value) =>
                onMpcParamsChange({ ...mpcParams, setpoint: value })
              }
            />
            <MpcNumberInput
              label="Input Penalty (Rho)"
              value={mpcParams.rho}
              step="0.01"
              onChange={(value) =>
                onMpcParamsChange({ ...mpcParams, rho: value })
              }
            />
            <MpcNumberInput
              label="Proc Gain (Kp)"
              value={mpcParams.process_gain}
              step="0.1"
              onChange={(value) =>
                onMpcParamsChange({ ...mpcParams, process_gain: value })
              }
            />
            <MpcNumberInput
              label="Proc Tau (s)"
              value={mpcParams.process_tau}
              step="0.1"
              onChange={(value) =>
                onMpcParamsChange({ ...mpcParams, process_tau: value })
              }
            />
          </div>

          <button
            type="button"
            className="btn btn-primary"
            style={{ width: "100%", marginTop: "1rem" }}
            onClick={onRunMpcSimulation}
          >
            Run Predictive Simulation
          </button>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            justifyContent: "center",
          }}
        >
          {mpcSimData ? (
            <MpcComparisonChart mpcParams={mpcParams} mpcSimData={mpcSimData} />
          ) : (
            <div
              style={{
                textAlign: "center",
                border: "1px dashed var(--panel-border)",
                padding: "2rem",
                borderRadius: "4px",
                color: "var(--text-muted)",
                fontSize: "0.8rem",
              }}
            >
              Run the predictive simulation to compare tracking trajectories.
            </div>
          )}
        </div>
      </div>
    </div>
  </div>
);

const ResultMetric: React.FC<{
  label: string;
  value: string;
  wide?: boolean;
}> = ({ label, value, wide = false }) => (
  <div
    style={{
      background: "rgba(255,255,255,0.02)",
      padding: "0.5rem",
      borderRadius: "4px",
      gridColumn: wide ? "span 2" : undefined,
    }}
  >
    <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
      {label}
    </div>
    <div style={{ fontSize: "1rem", fontWeight: 700 }}>{value}</div>
  </div>
);

const GainMetric: React.FC<{ label: string; value: string; color: string }> = ({
  label,
  value,
  color,
}) => (
  <div>
    <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
      {label}
    </div>
    <div style={{ fontSize: "0.85rem", fontWeight: 700, color }}>{value}</div>
  </div>
);

const MpcNumberInput: React.FC<{
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: string;
}> = ({ label, value, onChange, step }) => (
  <div className="input-group">
    <label className="input-label" style={{ fontSize: "0.65rem" }}>
      {label}
    </label>
    <input
      type="number"
      step={step}
      className="form-input"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  </div>
);

const MpcComparisonChart: React.FC<{
  mpcParams: MpcParams;
  mpcSimData: MpcSimResult;
}> = ({ mpcParams, mpcSimData }) => (
  <div style={{ position: "relative", width: "100%", height: "200px" }}>
    <div
      style={{
        fontSize: "0.75rem",
        fontWeight: 700,
        color: "var(--text-secondary)",
        marginBottom: "0.25rem",
        textAlign: "center",
      }}
    >
      Transient Comparison (PID vs MPC)
    </div>
    <svg
      style={{
        width: "100%",
        height: "100%",
        background: "rgba(0,0,0,0.15)",
        borderRadius: "4px",
      }}
    >
      <line x1="0" y1="50" x2="350" y2="50" stroke="rgba(255,255,255,0.05)" />
      <line x1="0" y1="100" x2="350" y2="100" stroke="rgba(255,255,255,0.05)" />
      <line x1="0" y1="150" x2="350" y2="150" stroke="rgba(255,255,255,0.05)" />
      <line
        x1="0"
        y1={200 - mpcParams.setpoint * 2}
        x2="350"
        y2={200 - mpcParams.setpoint * 2}
        stroke="var(--text-muted)"
        strokeDasharray="3,3"
        strokeWidth="1"
      />
      <polyline
        fill="none"
        stroke="var(--accent-purple)"
        strokeWidth="2"
        points={seriesPoints(mpcSimData.time.length, mpcSimData.pid.pv)}
      />
      <polyline
        fill="none"
        stroke="var(--accent-cyan)"
        strokeWidth="2"
        points={seriesPoints(mpcSimData.time.length, mpcSimData.mpc.pv)}
      />
    </svg>
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        marginTop: "0.25rem",
        fontSize: "0.65rem",
        color: "var(--text-muted)",
      }}
    >
      <span>Time: 0s</span>
      <span style={{ display: "flex", gap: "0.75rem" }}>
        <span style={{ color: "var(--accent-purple)" }}>PID</span>
        <span style={{ color: "var(--accent-cyan)" }}>MPC</span>
        <span style={{ color: "var(--text-muted)" }}>-- Setpoint</span>
      </span>
      <span>25s</span>
    </div>
  </div>
);

function seriesPoints(length: number, values: number[]): string {
  const denominator = Math.max(1, length - 1);
  return values
    .map((value, idx) => {
      const x = (idx / denominator) * 340 + 5;
      const y = 200 - value * 2;
      return `${x},${y}`;
    })
    .join(" ");
}
