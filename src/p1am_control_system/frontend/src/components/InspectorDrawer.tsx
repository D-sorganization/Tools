import React from "react";
import { BookOpen, ShieldAlert, Sliders, X } from "lucide-react";
import type { LadderTagInfo } from "../api/schemas";
import { TABS, type TabId } from "../lib/tabs";
import { TAG_INDICES, tagName } from "../lib/tags";
import type {
  AlicatMFCState,
  InterlockConfig,
  PIDConfig,
  RoutingConfig,
  TriggerNotification,
} from "../types";
import { CsvExporter } from "./CsvExporter";
import { DataCapturePanel } from "./DataCapturePanel";

export type InspectorState =
  | { type: "none" }
  | { type: "tag"; tagId: number }
  | { type: "custom_tag"; tagName: string }
  | { type: "pid"; index: number }
  | { type: "routing" }
  | { type: "alicat"; deviceId: string }
  | { type: "settings" }
  | { type: "export" };

type InspectorDrawerProps = {
  inspectorView: InspectorState;
  setInspectorView: React.Dispatch<React.SetStateAction<InspectorState>>;
  config: RoutingConfig;
  deploying: boolean;
  allTags: LadderTagInfo[];
  tagsDict: Record<string, number>;
  tagValues: Record<number, number>;
  overrideVal: string;
  showOverrideConfirm: boolean;
  alicats: AlicatMFCState[];
  alicatSetpointVal: string;
  visibleTabs: Record<TabId, boolean>;
  setOverrideVal: React.Dispatch<React.SetStateAction<string>>;
  setShowOverrideConfirm: React.Dispatch<React.SetStateAction<boolean>>;
  setAlicatSetpointVal: React.Dispatch<React.SetStateAction<string>>;
  executeOverride: (tagId: number | string) => Promise<void>;
  handleInterlockChange: (
    tagId: number,
    field: keyof InterlockConfig,
    value: number,
  ) => void;
  handlePidChange: (
    index: number,
    field: keyof PIDConfig,
    value: number,
  ) => void;
  handleDeploy: () => Promise<void>;
  handleTabVisibilityToggle: (tab: TabId) => void;
  handleAlicatSetpoint: (deviceId: string, val: number) => Promise<void>;
  handleAlicatGas: (deviceId: string, gas: string) => Promise<void>;
  triggerNotification: TriggerNotification;
};

export const InspectorDrawer: React.FC<InspectorDrawerProps> = ({
  inspectorView,
  setInspectorView,
  config,
  deploying,
  allTags,
  tagsDict,
  tagValues,
  overrideVal,
  showOverrideConfirm,
  alicats,
  alicatSetpointVal,
  visibleTabs,
  setOverrideVal,
  setShowOverrideConfirm,
  setAlicatSetpointVal,
  executeOverride,
  handleInterlockChange,
  handlePidChange,
  handleDeploy,
  handleTabVisibilityToggle,
  handleAlicatSetpoint,
  handleAlicatGas,
  triggerNotification,
}) => (
  <>
    {/* Inspector / data-export drawer — slides in from the right on demand */}
    {inspectorView.type !== "none" && (
      <div
        onClick={() => setInspectorView({ type: "none" })}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.4)",
          zIndex: 199,
        }}
      />
    )}
    <aside
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        height: "100vh",
        width: "440px",
        maxWidth: "92vw",
        transform:
          inspectorView.type !== "none" ? "translateX(0)" : "translateX(100%)",
        transition: "transform 0.25s ease",
        zIndex: 200,
        background: "var(--bg-color)",
        borderLeft: "1px solid var(--panel-border)",
        boxShadow: "-10px 0 28px rgba(0,0,0,0.35)",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "1.25rem",
        padding: "1.25rem",
      }}
    >
      {/* Main inspector panel */}
      <div
        className="glass-panel"
        style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
      >
        <div
          className="panel-header"
          style={{
            borderBottom: "1px solid var(--panel-border)",
            paddingBottom: "0.5rem",
          }}
        >
          <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            {inspectorView.type === "export"
              ? "Data export"
              : inspectorView.type === "settings"
                ? "Settings"
                : ""}
          </span>
          <button
            type="button"
            onClick={() => setInspectorView({ type: "none" })}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-secondary)",
              cursor: "pointer",
            }}
            aria-label="Close panel"
          >
            <X size={16} />
          </button>
        </div>

        {inspectorView.type === "export" && <DataCapturePanel />}

        {/* Legacy default content (kept for tag/pid detail flows; the
              drawer only opens on an explicit selection or the Export button) */}
        {inspectorView.type === "none" && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "1.25rem",
            }}
          >
            {/* CSV Log Exporter inside Default Sidebar view */}
            <CsvExporter triggerNotification={triggerNotification} />

            {/* Hardware Reference Docs Drawer */}
            <div
              style={{
                borderTop: "1px solid var(--panel-border)",
                paddingTop: "1rem",
              }}
            >
              <h3
                style={{
                  fontSize: "0.85rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  marginBottom: "0.6rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.3rem",
                }}
              >
                <BookOpen size={14} color="var(--accent-magenta)" />
                <span>P1AM Hardware Reference Docs</span>
              </h3>
              <ul
                style={{
                  fontSize: "0.8rem",
                  listStyleType: "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.4rem",
                }}
              >
                <li>
                  <a
                    href="https://www.automationdirect.com/pn/P1AM-100"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "var(--accent-cyan)",
                      textDecoration: "none",
                    }}
                  >
                    P1AM-100 CPU module Manual
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.automationdirect.com/pn/P1AM-ETH"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "var(--accent-cyan)",
                      textDecoration: "none",
                    }}
                  >
                    P1AM-ETH Ethernet module
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.automationdirect.com/pn/P1-04THM"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "var(--accent-cyan)",
                      textDecoration: "none",
                    }}
                  >
                    P1-04THM Thermocouple card
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.automationdirect.com/pn/P1-4ADL2DAL-1"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "var(--accent-cyan)",
                      textDecoration: "none",
                    }}
                  >
                    P1-4ADL2DAL-1 Analog I/O
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.automationdirect.com/pn/P1AM-GPIO"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "var(--accent-cyan)",
                      textDecoration: "none",
                    }}
                  >
                    P1AM-GPIO interface shield
                  </a>
                </li>
              </ul>
            </div>
          </div>
        )}

        {/* Tag register editor (Unified for both legacy tag index and custom named tags) */}
        {(inspectorView.type === "tag" ||
          inspectorView.type === "custom_tag") &&
          (() => {
            const isCustom = inspectorView.type === "custom_tag";
            const tagLabel = isCustom
              ? inspectorView.tagName
              : tagName(inspectorView.tagId);
            const tagInfo = allTags.find((t) => t.name === tagLabel);

            // Get current value
            const currentVal = isCustom
              ? (tagsDict[tagLabel] ?? 0.0)
              : (tagValues[inspectorView.tagId] ?? 0.0);

            // Check if writable (legacy tags are Read/Write, custom tags lookup from registry)
            const rwMode = tagInfo ? tagInfo.rw_mode : "Read/Write";
            const isWritable = rwMode === "Read/Write";

            return (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "1rem",
                }}
              >
                <div>
                  <h3
                    style={{
                      fontSize: "1rem",
                      fontWeight: 700,
                      color: "var(--accent-cyan)",
                      textTransform: "uppercase",
                    }}
                  >
                    Tag Inspector
                  </h3>
                  <div
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--text-primary)",
                      fontWeight: 700,
                      marginTop: "0.25rem",
                    }}
                    className="mono-text"
                  >
                    {tagLabel}
                  </div>
                  {tagInfo && tagInfo.description && (
                    <p
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--text-secondary)",
                        marginTop: "0.25rem",
                        lineHeight: 1.4,
                      }}
                    >
                      {tagInfo.description}
                    </p>
                  )}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      background: "var(--input-bg)",
                      padding: "0.5rem",
                      borderRadius: "4px",
                      marginTop: "0.5rem",
                      border: "1px solid var(--panel-border)",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.8rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      Current Value:
                    </span>
                    <span
                      className="mono-text"
                      style={{
                        fontSize: "0.9rem",
                        fontWeight: 700,
                        color: "var(--accent-cyan)",
                      }}
                    >
                      {currentVal.toFixed(2)}
                    </span>
                  </div>
                </div>

                {tagInfo && (
                  <div
                    style={{
                      fontSize: "0.75rem",
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "0.5rem",
                      background: "rgba(255,255,255,0.01)",
                      padding: "0.5rem",
                      borderRadius: "4px",
                      border: "1px solid var(--panel-border)",
                    }}
                  >
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>
                        Reg Type:
                      </span>{" "}
                      <strong style={{ color: "var(--text-secondary)" }}>
                        {tagInfo.register_type || "None"}
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>
                        Reg Num:
                      </span>{" "}
                      <strong style={{ color: "var(--text-secondary)" }}>
                        {tagInfo.register_num ?? "None"}
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>
                        Format:
                      </span>{" "}
                      <strong style={{ color: "var(--text-secondary)" }}>
                        {tagInfo.data_format || "None"}
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>
                        RW Mode:
                      </span>{" "}
                      <strong style={{ color: "var(--text-secondary)" }}>
                        {tagInfo.rw_mode}
                      </strong>
                    </div>
                  </div>
                )}

                {/* Section 1: Manual Value Override (Write Force) */}
                {isWritable && (
                  <div
                    style={{
                      borderTop: "1px solid var(--panel-border)",
                      paddingTop: "0.75rem",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.3rem",
                        marginBottom: "0.5rem",
                      }}
                    >
                      <ShieldAlert size={14} color="var(--color-warning)" />
                      <span
                        style={{
                          fontSize: "0.8rem",
                          fontWeight: 700,
                          textTransform: "uppercase",
                        }}
                      >
                        Manual Force Override
                      </span>
                    </div>
                    <div className="input-group">
                      <label className="input-label">
                        Override Force Value
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        className="form-input"
                        value={overrideVal}
                        onChange={(e) => setOverrideVal(e.target.value)}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowOverrideConfirm(true)}
                      className="btn"
                      style={{
                        width: "100%",
                        color: "var(--color-warning)",
                        borderColor: "var(--color-warning)",
                        padding: "0.45rem",
                        fontSize: "0.8rem",
                      }}
                    >
                      Force Register Write
                    </button>

                    {showOverrideConfirm && (
                      <div
                        style={{
                          background: "rgba(239, 68, 68, 0.1)",
                          border: "1px solid var(--color-error)",
                          borderRadius: "4px",
                          padding: "0.6rem",
                          marginTop: "0.5rem",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "0.75rem",
                            fontWeight: 700,
                            color: "var(--color-error)",
                            marginBottom: "0.25rem",
                          }}
                        >
                          Confirm Direct Write
                        </div>
                        <p
                          style={{
                            fontSize: "0.7rem",
                            color: "var(--text-secondary)",
                            marginBottom: "0.6rem",
                          }}
                        >
                          Forcing Tag {tagLabel} to {overrideVal} will overwrite
                          normal logic. Continue?
                        </p>
                        <div style={{ display: "flex", gap: "0.4rem" }}>
                          <button
                            type="button"
                            onClick={() => {
                              const idToSubmit = isCustom
                                ? tagLabel
                                : inspectorView.tagId;
                              executeOverride(idToSubmit);
                            }}
                            className="btn"
                            style={{
                              background: "var(--color-error)",
                              border: "none",
                              color: "#ffffff",
                              padding: "0.2rem 0.5rem",
                              fontSize: "0.75rem",
                            }}
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            onClick={() => setShowOverrideConfirm(false)}
                            className="btn"
                            style={{
                              padding: "0.2rem 0.5rem",
                              fontSize: "0.75rem",
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Section 2: Safety Interlocks Boundaries (Only for legacy tags currently, or if tag is in config.interlocks) */}
                {!isCustom && (
                  <div
                    style={{
                      borderTop: "1px solid var(--panel-border)",
                      paddingTop: "0.75rem",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.3rem",
                        marginBottom: "0.5rem",
                      }}
                    >
                      <Sliders size={14} color="var(--accent-purple)" />
                      <span
                        style={{
                          fontSize: "0.8rem",
                          fontWeight: 700,
                          textTransform: "uppercase",
                        }}
                      >
                        Safety Limits Interlocks
                      </span>
                    </div>
                    <div className="input-group">
                      <label className="input-label">
                        High Trip Limit (Alarm High)
                      </label>
                      <input
                        type="number"
                        step="0.5"
                        className="form-input"
                        value={
                          config.interlocks[inspectorView.tagId]?.high_limit ??
                          100.0
                        }
                        onChange={(e) =>
                          handleInterlockChange(
                            inspectorView.tagId,
                            "high_limit",
                            Number(e.target.value),
                          )
                        }
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">
                        Low Trip Limit (Alarm Low)
                      </label>
                      <input
                        type="number"
                        step="0.5"
                        className="form-input"
                        value={
                          config.interlocks[inspectorView.tagId]?.low_limit ??
                          0.0
                        }
                        onChange={(e) =>
                          handleInterlockChange(
                            inspectorView.tagId,
                            "low_limit",
                            Number(e.target.value),
                          )
                        }
                      />
                    </div>
                    <button
                      type="button"
                      onClick={handleDeploy}
                      disabled={deploying}
                      className="btn btn-primary"
                      style={{
                        width: "100%",
                        padding: "0.5rem",
                        fontSize: "0.85rem",
                        marginTop: "0.5rem",
                      }}
                    >
                      {deploying
                        ? "Deploying Configuration..."
                        : "Commit Safety Limits"}
                    </button>
                  </div>
                )}
              </div>
            );
          })()}

        {/* PID Loop Tune editor */}
        {inspectorView.type === "pid" && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "1rem",
            }}
          >
            <div>
              <h3
                style={{
                  fontSize: "1rem",
                  fontWeight: 700,
                  color: "var(--accent-purple)",
                  textTransform: "uppercase",
                }}
              >
                Tune PID Loop {inspectorView.index + 1}
              </h3>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "0.75rem",
              }}
            >
              <div className="input-group">
                <label className="input-label">PV Tag ID</label>
                <select
                  className="form-input"
                  value={config.pids[inspectorView.index].pv_tag_id}
                  onChange={(e) =>
                    handlePidChange(
                      inspectorView.index,
                      "pv_tag_id",
                      Number(e.target.value),
                    )
                  }
                >
                  {TAG_INDICES.map((i) => (
                    <option key={i} value={i}>
                      Tag {i}
                    </option>
                  ))}
                </select>
              </div>
              <div className="input-group">
                <label className="input-label">CV Tag ID</label>
                <select
                  className="form-input"
                  value={config.pids[inspectorView.index].cv_tag_id}
                  onChange={(e) =>
                    handlePidChange(
                      inspectorView.index,
                      "cv_tag_id",
                      Number(e.target.value),
                    )
                  }
                >
                  {TAG_INDICES.map((i) => (
                    <option key={i} value={i}>
                      Tag {i}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Setpoint Target</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                value={config.pids[inspectorView.index].setpoint}
                onChange={(e) =>
                  handlePidChange(
                    inspectorView.index,
                    "setpoint",
                    Number(e.target.value),
                  )
                }
              />
            </div>

            <div className="input-group">
              <label className="input-label">Proportional Gain (Kp)</label>
              <input
                type="number"
                step="0.01"
                className="form-input"
                value={config.pids[inspectorView.index].kp}
                onChange={(e) =>
                  handlePidChange(
                    inspectorView.index,
                    "kp",
                    Number(e.target.value),
                  )
                }
              />
            </div>

            <div className="input-group">
              <label className="input-label">Integral Gain (Ki)</label>
              <input
                type="number"
                step="0.01"
                className="form-input"
                value={config.pids[inspectorView.index].ki}
                onChange={(e) =>
                  handlePidChange(
                    inspectorView.index,
                    "ki",
                    Number(e.target.value),
                  )
                }
              />
            </div>

            <div className="input-group">
              <label className="input-label">Derivative Gain (Kd)</label>
              <input
                type="number"
                step="0.01"
                className="form-input"
                value={config.pids[inspectorView.index].kd}
                onChange={(e) =>
                  handlePidChange(
                    inspectorView.index,
                    "kd",
                    Number(e.target.value),
                  )
                }
              />
            </div>

            {/* Apply Panel Changes */}
            <button
              type="button"
              onClick={handleDeploy}
              disabled={deploying}
              className="btn btn-primary"
              style={{
                width: "100%",
                padding: "0.5rem",
                fontSize: "0.85rem",
                marginTop: "0.5rem",
              }}
            >
              {deploying ? "Deploying Configuration..." : "Commit PID Tuning"}
            </button>
          </div>
        )}

        {/* Routing Matrix Panel (in inspector) */}
        {inspectorView.type === "routing" && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "1rem",
            }}
          >
            <div>
              <h3
                style={{
                  fontSize: "1rem",
                  fontWeight: 700,
                  color: "var(--accent-cyan)",
                  textTransform: "uppercase",
                }}
              >
                Routing Matrix Patching
              </h3>
              <p
                style={{
                  fontSize: "0.8rem",
                  color: "var(--text-secondary)",
                  lineHeight: 1.4,
                  marginTop: "0.25rem",
                }}
              >
                Modify the crossbar mappings on the main screen matrix. Once
                matched correctly, click below to deploy the configuration down
                to the SAMD21 NVRAM.
              </p>
            </div>

            <div
              style={{
                padding: "0.75rem",
                background: "rgba(56, 189, 248, 0.08)",
                border: "1px solid var(--accent-cyan)",
                borderRadius: "4px",
              }}
            >
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-primary)",
                  fontWeight: 700,
                }}
              >
                Modbus Matrix Routing Info:
              </span>
              <p
                style={{
                  fontSize: "0.7rem",
                  color: "var(--text-secondary)",
                  marginTop: "0.2rem",
                }}
              >
                Input patching links physical thermocouple or analog cards
                (Channels 0-5) to broker tags. Output patching links broker tags
                to control variable registers (Channels 10-11) driving solenoid
                actuators.
              </p>
            </div>

            {/* Apply Panel Changes */}
            <button
              type="button"
              onClick={handleDeploy}
              disabled={deploying}
              className="btn btn-primary"
              style={{
                width: "100%",
                padding: "0.5rem",
                fontSize: "0.85rem",
                marginTop: "0.5rem",
              }}
            >
              {deploying
                ? "Deploying Configuration..."
                : "Commit Matrix Mapping"}
            </button>
          </div>
        )}

        {/* Settings inspector view */}
        {inspectorView.type === "settings" && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.6rem",
            }}
          >
            <span
              style={{
                fontSize: "0.75rem",
                color: "var(--text-primary)",
                fontWeight: 700,
                textTransform: "uppercase",
              }}
            >
              Visible tabs
            </span>
            {
              // ⚡ Bolt Optimization: Use the centralized TABS array to avoid allocating a new array of tuples on every render
              TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`tab-toggle ${visibleTabs[tab.id] ? "on" : ""}`}
                  onClick={() => handleTabVisibilityToggle(tab.id)}
                >
                  <span>{tab.settingsLabel}</span>
                  <span className="tab-toggle-switch" />
                </button>
              ))
            }
          </div>
        )}

        {/* Alicat Mass Flow Controller inspector */}
        {inspectorView.type === "alicat" &&
          (() => {
            const mfc = alicats.find(
              (m) => m.device_id === inspectorView.deviceId,
            );
            if (!mfc) {
              return (
                <div
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  Alicat MFC {inspectorView.deviceId} not found.
                </div>
              );
            }
            return (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "1rem",
                }}
              >
                <div>
                  <h3
                    style={{
                      fontSize: "1rem",
                      fontWeight: 700,
                      color: "var(--color-warning)",
                      textTransform: "uppercase",
                    }}
                  >
                    Inspect {mfc.name}
                  </h3>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-secondary)",
                      marginTop: "0.2rem",
                    }}
                  >
                    Device ID: {mfc.device_id} | State: {mfc.connection_state}
                  </div>
                </div>

                {/* Realtime Stats Block */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "0.5rem",
                    background: "var(--input-bg)",
                    padding: "0.75rem",
                    borderRadius: "4px",
                    border: "1px solid var(--panel-border)",
                  }}
                >
                  <div>
                    <div
                      style={{
                        fontSize: "0.65rem",
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                      }}
                    >
                      Mass Flow
                    </div>
                    <div
                      className="mono-text"
                      style={{
                        fontSize: "1.05rem",
                        fontWeight: 700,
                        color: "var(--accent-cyan)",
                      }}
                    >
                      {mfc.mass_flow.toFixed(2)}{" "}
                      <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>
                        SLPM
                      </span>
                    </div>
                  </div>
                  <div>
                    <div
                      style={{
                        fontSize: "0.65rem",
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                      }}
                    >
                      Vol. Flow
                    </div>
                    <div
                      className="mono-text"
                      style={{
                        fontSize: "1.05rem",
                        fontWeight: 700,
                        color: "var(--text-secondary)",
                      }}
                    >
                      {mfc.volumetric_flow.toFixed(2)}{" "}
                      <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>
                        LPM
                      </span>
                    </div>
                  </div>
                  <div style={{ marginTop: "0.35rem" }}>
                    <div
                      style={{
                        fontSize: "0.65rem",
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                      }}
                    >
                      Pressure
                    </div>
                    <div
                      className="mono-text"
                      style={{
                        fontSize: "1.05rem",
                        fontWeight: 700,
                        color: "var(--text-primary)",
                      }}
                    >
                      {mfc.pressure.toFixed(2)}{" "}
                      <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>
                        PSIA
                      </span>
                    </div>
                  </div>
                  <div style={{ marginTop: "0.35rem" }}>
                    <div
                      style={{
                        fontSize: "0.65rem",
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                      }}
                    >
                      Temperature
                    </div>
                    <div
                      className="mono-text"
                      style={{
                        fontSize: "1.05rem",
                        fontWeight: 700,
                        color: "var(--text-primary)",
                      }}
                    >
                      {mfc.temperature.toFixed(1)}{" "}
                      <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>
                        °C
                      </span>
                    </div>
                  </div>
                </div>

                {/* Form 1: Setpoint command */}
                <div
                  style={{
                    borderTop: "1px solid var(--panel-border)",
                    paddingTop: "0.75rem",
                  }}
                >
                  <label
                    className="input-label"
                    style={{
                      fontWeight: 700,
                      marginBottom: "0.4rem",
                      display: "block",
                    }}
                  >
                    Flow Setpoint Command (SLPM)
                  </label>
                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max={mfc.max_flow}
                      className="form-input"
                      style={{ flex: 1 }}
                      value={alicatSetpointVal}
                      onChange={(e) => setAlicatSetpointVal(e.target.value)}
                    />
                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{
                        padding: "0.45rem 1rem",
                        fontSize: "0.8rem",
                        whiteSpace: "nowrap",
                      }}
                      onClick={() => {
                        const parsed = parseFloat(alicatSetpointVal);
                        if (
                          !isNaN(parsed) &&
                          parsed >= 0 &&
                          parsed <= mfc.max_flow
                        ) {
                          handleAlicatSetpoint(mfc.device_id, parsed);
                        } else {
                          triggerNotification(
                            `Please enter a valid setpoint between 0 and ${mfc.max_flow}.`,
                            "error",
                          );
                        }
                      }}
                    >
                      Set
                    </button>
                  </div>
                  <div
                    style={{
                      fontSize: "0.65rem",
                      color: "var(--text-muted)",
                      marginTop: "0.25rem",
                    }}
                  >
                    Maximum flow limit: {mfc.max_flow} SLPM
                  </div>
                </div>

                {/* Form 2: Gas select dropdown */}
                <div
                  style={{
                    borderTop: "1px solid var(--panel-border)",
                    paddingTop: "0.75rem",
                  }}
                >
                  <label
                    className="input-label"
                    style={{
                      fontWeight: 700,
                      marginBottom: "0.4rem",
                      display: "block",
                    }}
                  >
                    Active Gas Calibration
                  </label>
                  <select
                    className="form-input"
                    value={mfc.gas}
                    onChange={(e) =>
                      handleAlicatGas(mfc.device_id, e.target.value)
                    }
                  >
                    {["O2", "N2", "CO2", "He", "H2", "Air"].map((species) => (
                      <option key={species} value={species}>
                        {species} (
                        {species === "O2"
                          ? "Oxygen"
                          : species === "N2"
                            ? "Nitrogen"
                            : species === "CO2"
                              ? "Carbon Dioxide"
                              : species === "He"
                                ? "Helium"
                                : species === "H2"
                                  ? "Hydrogen"
                                  : "Clean Air"}
                        )
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            );
          })()}
      </div>
    </aside>
  </>
);
