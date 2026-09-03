import React, { useState } from "react";
import type { RoutingConfig, PIDConfig, InterlockConfig } from "../App";
import { limitInputValue, parseLimitInput } from "../lib/limits";
import { ShieldAlert, Cpu, HardDriveDownload } from "lucide-react";
import { TAG_INDICES } from "./RoutingMatrix";

interface ControlDashboardProps {
  config: RoutingConfig;
  onUpdate: (updatedConfig: RoutingConfig) => void;
  onDeploy: () => Promise<void>;
  deploying: boolean;
  forcedTab?: "pid" | "interlocks";
}

export const ControlDashboard: React.FC<ControlDashboardProps> = ({
  config,
  onUpdate,
  onDeploy,
  deploying,
  forcedTab,
}) => {
  const [localTab, setLocalTab] = useState<"pid" | "interlocks">("pid");
  const [selectedPidIdx, setSelectedPidIdx] = useState<number>(0);
  const [selectedInterlockIdx, setSelectedInterlockIdx] = useState<number>(0);

  const activeTab = forcedTab || localTab;


  // Helper to update active PID config
  const handlePidChange = (field: keyof PIDConfig, value: number) => {
    const updatedPids = config.pids.map((pid, idx) => {
      if (idx === selectedPidIdx) {
        return { ...pid, [field]: value };
      }
      return pid;
    });
    onUpdate({ ...config, pids: updatedPids });
  };

  // Helper to update active Interlock config
  const handleInterlockChange = (field: keyof InterlockConfig, value: number | null) => {
    const updatedInterlocks = config.interlocks.map((interlock, idx) => {
      if (idx === selectedInterlockIdx) {
        return { ...interlock, [field]: value };
      }
      return interlock;
    });
    onUpdate({ ...config, interlocks: updatedInterlocks });
  };

  const activePid = config.pids[selectedPidIdx];
  const activeInterlock = config.interlocks[selectedInterlockIdx];

  return (
    <div className="glass-panel" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Tabs Header */}
      {!forcedTab && (
        <div style={{ display: "flex", borderBottom: "1px solid var(--panel-border)", marginBottom: "1.25rem" }}>
          <button
            type="button"
            onClick={() => setLocalTab("pid")}
            style={{
              background: "none",
              border: "none",
              color: activeTab === "pid" ? "var(--accent-cyan)" : "var(--text-secondary)",
              padding: "0.75rem 1.25rem",
              fontSize: "0.9rem",
              fontWeight: 600,
              cursor: "pointer",
              borderBottom: activeTab === "pid" ? "2px solid var(--accent-cyan)" : "2px solid transparent",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              transition: "all var(--transition-fast)",
            }}
          >
            <Cpu size={16} />
            PID Control Loops
          </button>
          <button
            type="button"
            onClick={() => setLocalTab("interlocks")}
            style={{
              background: "none",
              border: "none",
              color: activeTab === "interlocks" ? "var(--accent-purple)" : "var(--text-secondary)",
              padding: "0.75rem 1.25rem",
              fontSize: "0.9rem",
              fontWeight: 600,
              cursor: "pointer",
              borderBottom: activeTab === "interlocks" ? "2px solid var(--accent-purple)" : "2px solid transparent",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              transition: "all var(--transition-fast)",
            }}
          >
            <ShieldAlert size={16} />
            Safety Interlocks
          </button>
        </div>
      )}

      {/* Tab Contents */}
      <div style={{ flex: 1, minHeight: "220px" }}>
        {activeTab === "pid" && activePid && (
          <div>
            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
              {config.pids.map((_, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setSelectedPidIdx(idx)}
                  className="btn"
                  style={{
                    padding: "0.4rem 0.8rem",
                    fontSize: "0.8rem",
                    borderColor: selectedPidIdx === idx ? "var(--accent-cyan)" : "rgba(255,255,255,0.08)",
                    background: selectedPidIdx === idx ? "rgba(0, 242, 254, 0.1)" : "rgba(0,0,0,0.2)",
                    color: selectedPidIdx === idx ? "var(--accent-cyan)" : "var(--text-secondary)",
                  }}
                  aria-pressed={selectedPidIdx === idx}
                >
                  Loop {idx + 1}
                </button>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div className="input-group">
                <label className="input-label">Process Variable (PV) Tag</label>
                <select
                  className="form-input"
                  value={activePid.pv_tag_id}
                  onChange={(e) => handlePidChange("pv_tag_id", Number(e.target.value))}
                >
                  {TAG_INDICES.map((i) => (
                    <option key={i} value={i}>Tag {i}</option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Control Variable (CV) Tag</label>
                <select
                  className="form-input"
                  value={activePid.cv_tag_id}
                  onChange={(e) => handlePidChange("cv_tag_id", Number(e.target.value))}
                >
                  {TAG_INDICES.map((i) => (
                    <option key={i} value={i}>Tag {i}</option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Setpoint</label>
                <input
                  type="number"
                  step="0.1"
                  className="form-input"
                  value={activePid.setpoint}
                  onChange={(e) => handlePidChange("setpoint", Number(e.target.value))}
                />
              </div>

              <div className="input-group">
                <label className="input-label">Proportional Gain (Kp)</label>
                <input
                  type="number"
                  step="0.01"
                  className="form-input"
                  value={activePid.kp}
                  onChange={(e) => handlePidChange("kp", Number(e.target.value))}
                />
              </div>

              <div className="input-group">
                <label className="input-label">Integral Gain (Ki)</label>
                <input
                  type="number"
                  step="0.01"
                  className="form-input"
                  value={activePid.ki}
                  onChange={(e) => handlePidChange("ki", Number(e.target.value))}
                />
              </div>

              <div className="input-group">
                <label className="input-label">Derivative Gain (Kd)</label>
                <input
                  type="number"
                  step="0.01"
                  className="form-input"
                  value={activePid.kd}
                  onChange={(e) => handlePidChange("kd", Number(e.target.value))}
                />
              </div>
            </div>
          </div>
        )}

        {activeTab === "interlocks" && activeInterlock && (
          <div>
            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem", flexWrap: "wrap", maxHeight: "80px", overflowY: "auto" }}>
              <div className="input-group" style={{ width: "100%", marginBottom: "0.5rem" }}>
                <label className="input-label">Select Signal Broker Tag to Secure</label>
                <select
                  className="form-input"
                  value={selectedInterlockIdx}
                  onChange={(e) => setSelectedInterlockIdx(Number(e.target.value))}
                >
                  {TAG_INDICES.map((i) => (
                    <option key={i} value={i}>Tag {i} (Safety Configuration)</option>
                  ))}
                </select>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div className="input-group">
                <label className="input-label">High Trip Limit (Alarm High)</label>
                <input
                  type="number"
                  step="0.5"
                  className="form-input"
                  value={limitInputValue(activeInterlock.high_limit)}
                  placeholder="disabled"
                  onChange={(e) => handleInterlockChange("high_limit", parseLimitInput(e.target.value))}
                />
              </div>

              <div className="input-group">
                <label className="input-label">Low Trip Limit (Alarm Low)</label>
                <input
                  type="number"
                  step="0.5"
                  className="form-input"
                  value={limitInputValue(activeInterlock.low_limit)}
                  placeholder="disabled"
                  onChange={(e) => handleInterlockChange("low_limit", parseLimitInput(e.target.value))}
                />
              </div>
            </div>
            
            <div style={{ marginTop: "1rem", padding: "0.75rem", background: "rgba(255, 23, 68, 0.08)", border: "1px solid rgba(255, 23, 68, 0.2)", borderRadius: "6px", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              <strong>Safety Inhibit Active:</strong> Exceeding either boundary forces outputs to 0V and triggers physical GPIO trip inhibitors on the P1AM controller.
            </div>
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.05)", paddingTop: "1.25rem", marginTop: "1.25rem", display: "flex", gap: "1rem", justifyContent: "flex-end" }}>
        <button
          type="button"
          onClick={onDeploy}
          disabled={deploying}
          className="btn btn-primary"
          style={{ width: "100%" }}
        >
          <HardDriveDownload size={18} />
          {deploying ? "Deploying Configuration..." : "Deploy Config & Write to NVRAM"}
        </button>
      </div>
    </div>
  );
};
