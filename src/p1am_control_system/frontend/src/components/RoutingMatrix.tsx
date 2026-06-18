import React from "react";
import type { RoutingConfig } from "../App";
import { TAG_INDICES } from "../lib/tags";

interface RoutingMatrixProps {
  config: RoutingConfig;
  onUpdate: (updatedConfig: RoutingConfig) => void;
  tagValues: number[];
}

const INPUT_LABELS = [
  "Thermo T0 (°C)",
  "Thermo T1 (°C)",
  "Thermo T2 (°C)",
  "Thermo T3 (°C)",
  "Analog In A0 (V)",
  "Analog In A1 (V)",
];

const OUTPUT_LABELS = ["Analog Out V0 (V)", "Analog Out V1 (V)"];

// Re-exported from the centralized tag helpers (#3546) so existing importers
// (ControlDashboard, TrendChart) keep working while the source of truth is one place.
export { TAG_INDICES };

export const RoutingMatrix: React.FC<RoutingMatrixProps> = ({
  config,
  onUpdate,
  tagValues,
}) => {
  const handleInputRoute = (inputIdx: number, tagId: number) => {
    const nextRouting = [...config.input_routing];
    nextRouting[inputIdx] = tagId;
    onUpdate({ ...config, input_routing: nextRouting });
  };

  const handleOutputRoute = (outputIdx: number, tagId: number) => {
    const nextRouting = [...config.output_routing];
    nextRouting[outputIdx] = tagId;
    onUpdate({ ...config, output_routing: nextRouting });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      {/* Input Routing Crossbar */}
      <div className="glass-panel">
        <div className="panel-header">
          <span>Input Routing Matrix (Sensor ➜ Tag)</span>
          <span
            style={{
              fontSize: "0.8rem",
              color: "var(--accent-cyan)",
              fontWeight: 500,
            }}
          >
            Active Signal Broker Routing
          </span>
        </div>
        <div className="matrix-container">
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                <th
                  style={{
                    width: "180px",
                    textAlign: "left",
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  Physical Channel
                </th>
                {TAG_INDICES.map((i) => (
                  <th key={i} style={{ padding: "0.25rem" }}>
                    <div className="col-label" style={{ height: "45px" }}>
                      Tag {i}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {INPUT_LABELS.map((label, rowIdx) => (
                <tr
                  key={rowIdx}
                  style={{
                    borderBottom: "1px solid rgba(255, 255, 255, 0.03)",
                  }}
                >
                  <td
                    style={{
                      padding: "0.5rem 0",
                      fontSize: "0.85rem",
                      fontWeight: 500,
                    }}
                  >
                    {label}
                  </td>
                  {TAG_INDICES.map((colIdx) => {
                    const isActive = config.input_routing[rowIdx] === colIdx;
                    return (
                      <td
                        key={colIdx}
                        style={{ padding: "0.15rem", textAlign: "center" }}
                      >
                        <button
                          type="button"
                          className={`matrix-cell ${isActive ? "active-input" : ""}`}
                          onClick={() => handleInputRoute(rowIdx, colIdx)}
                          title={`Route ${label} to Tag ${colIdx} (Current val: ${tagValues[colIdx]?.toFixed(2) ?? "0.00"})`}
                          aria-pressed={isActive}
                        >
                          {isActive && (
                            <div
                              style={{
                                width: "6px",
                                height: "6px",
                                borderRadius: "50%",
                                background: "#0b0d12",
                              }}
                            />
                          )}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Output Routing Crossbar */}
      <div className="glass-panel">
        <div className="panel-header">
          <span>Output Routing Matrix (Tag ➜ Actuator)</span>
          <span
            style={{
              fontSize: "0.8rem",
              color: "var(--accent-purple)",
              fontWeight: 500,
            }}
          >
            Active Actuator Mappings
          </span>
        </div>
        <div className="matrix-container">
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                <th
                  style={{
                    width: "180px",
                    textAlign: "left",
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  Physical Channel
                </th>
                {TAG_INDICES.map((i) => (
                  <th key={i} style={{ padding: "0.25rem" }}>
                    <div className="col-label" style={{ height: "45px" }}>
                      Tag {i}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {OUTPUT_LABELS.map((label, rowIdx) => (
                <tr
                  key={rowIdx}
                  style={{
                    borderBottom: "1px solid rgba(255, 255, 255, 0.03)",
                  }}
                >
                  <td
                    style={{
                      padding: "0.5rem 0",
                      fontSize: "0.85rem",
                      fontWeight: 500,
                    }}
                  >
                    {label}
                  </td>
                  {TAG_INDICES.map((colIdx) => {
                    const isActive = config.output_routing[rowIdx] === colIdx;
                    return (
                      <td
                        key={colIdx}
                        style={{ padding: "0.15rem", textAlign: "center" }}
                      >
                        <button
                          type="button"
                          className={`matrix-cell ${isActive ? "active-output" : ""}`}
                          onClick={() => handleOutputRoute(rowIdx, colIdx)}
                          title={`Route Tag ${colIdx} (Current val: ${tagValues[colIdx]?.toFixed(2) ?? "0.00"}) to ${label}`}
                          aria-pressed={isActive}
                        >
                          {isActive && (
                            <div
                              style={{
                                width: "6px",
                                height: "6px",
                                borderRadius: "50%",
                                background: "#0b0d12",
                              }}
                            />
                          )}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
