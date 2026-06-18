import React, { useState, useEffect } from "react";
import { RoutingMatrix } from "./components/RoutingMatrix";
import { TrendChart } from "./components/TrendChart";
import { AlarmsHeader } from "./components/AlarmsHeader";
import { EStopButton } from "./components/EStopButton";
import { InterlocksPanel } from "./components/InterlocksPanel";
import { EventLogView } from "./components/EventLogView";
import { ProjectImporter } from "./components/ProjectImporter";
import { LadderExplorer } from "./components/LadderExplorer";
import { PlantHierarchy } from "./components/PlantHierarchy";
import { PowerSupplyControl } from "./components/PowerSupplyControl";
import { TemperatureControl } from "./components/TemperatureControl";
import { TuningPanel } from "./components/TuningPanel";
import {
  InspectorDrawer,
  type InspectorState,
} from "./components/InspectorDrawer";
import type { LadderTagInfo } from "./api/schemas";
import { NotificationBanner } from "./components/NotificationBanner";
import { TabBar } from "./components/TabBar";
import { useTelemetryStream } from "./hooks/useTelemetryStream";
import { type TabId, defaultTabVisibility } from "./lib/tabs";
import { TAG_INDICES, tagName, parseTagId } from "./lib/tags";
import { fmtNumber } from "./lib/format";
import * as api from "./api/endpoints";
import { ApiError } from "./api/client";
import type {
  PIDConfig,
  InterlockConfig,
  RoutingConfig,
  NotificationState,
  NotificationType,
  EventLogEntry,
  TuningResult,
  MpcSimResult,
} from "./types";
import {
  Activity,
  Sliders,
  Shuffle,
  Sun,
  Moon,
  Info,
  Settings,
} from "lucide-react";

// Re-export domain types for back-compat with existing importers (AlarmsHeader,
// EventLogView, InterlocksPanel, RoutingMatrix, ControlDashboard).
export type {
  PIDConfig,
  InterlockConfig,
  RoutingConfig,
  AlicatMFCState,
  EventLogEntry,
  ActiveAlarm,
} from "./types";

const DEFAULT_CONFIG: RoutingConfig = {
  input_routing: [0, 1, 2, 3, 4, 5],
  output_routing: [10, 11],
  pids: Array.from({ length: 4 }).map(() => ({
    pv_tag_id: 0,
    cv_tag_id: 0,
    setpoint: 0.0,
    kp: 0.0,
    ki: 0.0,
    kd: 0.0,
  })),
  interlocks: TAG_INDICES.map(() => ({
    lolo_limit: 0.0,
    low_limit: 5.0,
    high_limit: 95.0,
    hihi_limit: 100.0,
  })),
};

export const App: React.FC = () => {
  const [config, setConfig] = useState<RoutingConfig>(DEFAULT_CONFIG);
  const [deploying, setDeploying] = useState<boolean>(false);
  const [notification, setNotification] = useState<NotificationState | null>(
    null,
  );

  // Show a notification banner (declared early so the telemetry hook can use it).
  const triggerNotification = (message: string, type: NotificationType) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  // Live telemetry (WS stream + derived state) is owned by a dedicated hook.
  const {
    tagValues,
    history,
    tagsDict,
    alicats,
    activeAlarms,
    eStopActive,
    powerSupplyStatus,
    temperatureStatus,
    isConnected,
    setAlicats,
    setActiveAlarms,
    setEStopActive,
  } = useTelemetryStream({
    onConnect: () =>
      triggerNotification("SCADA live stream connected.", "info"),
  });

  // Events history (polled separately from the live stream).
  const [eventsHistory, setEventsHistory] = useState<EventLogEntry[]>([]);

  // Tab Navigation and Visibility State
  const [activeTab, setActiveTab] = useState<TabId>("powerSupply");
  const [visibleTabs, setVisibleTabs] =
    useState<Record<TabId, boolean>>(defaultTabVisibility);

  // PID Tuning State
  const [selectedTuningLoop, setSelectedTuningLoop] = useState<number>(0);
  const [isTuningMode, setIsTuningMode] = useState<boolean>(false);
  const [tuningStepVal, setTuningStepVal] = useState<string>("50.0");
  const [tuningResults, setTuningResults] = useState<TuningResult | null>(null);

  // MPC Simulation State
  const [mpcParams, setMpcParams] = useState({
    prediction_horizon: 10,
    control_horizon: 3,
    setpoint: 50.0,
    rho: 0.1,
    process_gain: 1.2,
    process_tau: 5.0,
    process_delay: 1.0,
  });
  const [mpcSimData, setMpcSimData] = useState<MpcSimResult | null>(null);

  // Theme & Navigation Sidebar State
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (localStorage.getItem("theme") as "dark" | "light") || "dark";
  });
  const [inspectorView, setInspectorView] = useState<InspectorState>({
    type: "none",
  });

  // Manual Override Form State inside inspector
  const [overrideVal, setOverrideVal] = useState<string>("0.0");
  const [showOverrideConfirm, setShowOverrideConfirm] =
    useState<boolean>(false);

  // Alicat MFC form states
  const [alicatSetpointVal, setAlicatSetpointVal] = useState<string>("");

  // Large-scale plant tags state (ladder registry, for the inspector).
  const [allTags, setAllTags] = useState<LadderTagInfo[]>([]);

  const fetchAllTags = async () => {
    try {
      setAllTags(await api.getLadderExplorer());
    } catch (e) {
      console.error("Failed to fetch all tags", e);
    }
  };

  const handleSelectTag = (name: string) => {
    const id = parseTagId(name);
    if (id !== null) {
      setInspectorView({ type: "tag", tagId: id });
      setOverrideVal(fmtNumber(tagValues[id] ?? 0.0, 2, "0.00"));
      setShowOverrideConfirm(false);
      return;
    }
    setInspectorView({ type: "custom_tag", tagName: name });
    setOverrideVal(fmtNumber(tagsDict[name] ?? 0.0, 2, "0.00"));
    setShowOverrideConfirm(false);
  };

  useEffect(() => {
    if (inspectorView.type === "alicat") {
      const mfc = alicats.find((m) => m.device_id === inspectorView.deviceId);
      if (mfc) {
        setAlicatSetpointVal(mfc.setpoint.toString());
      }
    }
  }, [inspectorView, alicats]);

  // Synchronize CSS custom property set on HTML element
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Fetch current PLC configuration on mount. The routing payload is a dynamic
  // TAG_x <-> index normalization, so it is handled with explicit shape guards
  // rather than a fixed schema.
  const fetchConfig = async () => {
    try {
      const data = (await api.getRouting()) as {
        input_routing?: unknown;
        output_routing?: unknown;
        pids?: unknown;
        interlocks?: Record<string, Partial<InterlockConfig>>;
      };

      // Helper to convert TAG_x to x (number)
      const toInt = (t: unknown): number => {
        if (typeof t === "number") return t;
        if (typeof t === "string") {
          const parsed = parseTagId(t);
          return parsed ?? 0;
        }
        return 0;
      };

      const mappedConfig: RoutingConfig = {
        input_routing: Array.isArray(data.input_routing)
          ? data.input_routing.map(toInt)
          : [],
        output_routing: Array.isArray(data.output_routing)
          ? data.output_routing.map(toInt)
          : [],
        pids: Array.isArray(data.pids)
          ? data.pids.map((p: Record<string, unknown>) => ({
              pv_tag_id: toInt(p.pv_tag),
              cv_tag_id: toInt(p.cv_tag),
              setpoint: typeof p.setpoint === "number" ? p.setpoint : 0,
              kp: typeof p.kp === "number" ? p.kp : 0,
              ki: typeof p.ki === "number" ? p.ki : 0,
              kd: typeof p.kd === "number" ? p.kd : 0,
            }))
          : [],
        interlocks: (() => {
          const mappedInts: InterlockConfig[] = TAG_INDICES.map(() => ({
            lolo_limit: 0.0,
            low_limit: 5.0,
            high_limit: 95.0,
            hihi_limit: 100.0,
          }));
          if (data.interlocks && typeof data.interlocks === "object") {
            TAG_INDICES.forEach((i) => {
              const entry = data.interlocks?.[tagName(i)];
              if (entry) {
                mappedInts[i] = {
                  lolo_limit: entry.lolo_limit ?? 0.0,
                  low_limit: entry.low_limit ?? 5.0,
                  high_limit: entry.high_limit ?? 95.0,
                  hihi_limit: entry.hihi_limit ?? 100.0,
                };
              }
            });
          }
          return mappedInts;
        })(),
      };

      setConfig(mappedConfig);
      triggerNotification("Loaded active PLC configuration.", "success");
    } catch {
      triggerNotification("PLC offline. Unable to read registers.", "error");
    }
  };

  // Deploy configuration & write to NVRAM
  const handleDeploy = async () => {
    setDeploying(true);
    try {
      const payload = {
        input_routing: config.input_routing.map(tagName),
        output_routing: config.output_routing.map(tagName),
        pids: config.pids.map((p) => ({
          pv_tag: tagName(p.pv_tag_id),
          cv_tag: tagName(p.cv_tag_id),
          setpoint: p.setpoint,
          kp: p.kp,
          ki: p.ki,
          kd: p.kd,
        })),
        interlocks: (() => {
          const dict: Record<string, InterlockConfig> = {};
          config.interlocks.forEach((intVal, i) => {
            dict[tagName(i)] = intVal;
          });
          return dict;
        })(),
      };

      await api.deployRouting(payload);
      triggerNotification(
        "Configuration deployed & written to NVRAM successfully.",
        "success",
      );
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : "connection error";
      triggerNotification(`Deployment failed: ${detail}`, "error");
    } finally {
      setDeploying(false);
    }
  };

  // Trigger global emergency stop (E-stop). Fail toward the stopped state: on a
  // successful POST we optimistically latch the button (the WebSocket frame
  // corrects it), so the operator gets confirmation even if the stream is down.
  const handleEStop = async () => {
    try {
      await api.triggerEStop();
      setEStopActive(true);
      triggerNotification("EMERGENCY SHUTDOWN COMMAND ISSUED!", "error");
    } catch {
      triggerNotification(
        "E-STOP NOT CONFIRMED by server — verify the PLC output is dead!",
        "error",
      );
    }
  };

  // Clear a latched E-stop. Never optimistically mark the system safe — only the
  // server's e_stop_active (via the WebSocket) lowers the latch. On failure the
  // button stays in the "latched" state so the UI never lies about safety.
  const handleClearEStop = async () => {
    if (
      !window.confirm(
        "Clear the emergency stop? The supply stays at zero until you " +
          "re-enable permissive and re-enter a setpoint. Continue?",
      )
    )
      return;
    try {
      await api.clearEStop();
      triggerNotification("E-stop cleared. Re-arm to resume.", "info");
    } catch {
      triggerNotification(
        "E-stop clear was rejected — system remains latched.",
        "error",
      );
    }
  };

  // Execute direct tag value force override
  const executeOverride = async (tagId: number | string) => {
    const parsed = parseFloat(overrideVal);
    if (isNaN(parsed)) {
      triggerNotification("Invalid numeric value.", "error");
      return;
    }
    try {
      await api.forceTag(tagId, parsed);
      triggerNotification(
        `Successfully forced Tag ${tagId} to ${parsed}`,
        "success",
      );
      setShowOverrideConfirm(false);
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : "PLC write failed";
      triggerNotification(`Override failed: ${detail}`, "error");
    }
  };

  // Helper to update active PID config
  const handlePidChange = (
    index: number,
    field: keyof PIDConfig,
    value: number,
  ) => {
    const updatedPids = config.pids.map((pid, idx) => {
      if (idx === index) {
        return { ...pid, [field]: value };
      }
      return pid;
    });
    setConfig({ ...config, pids: updatedPids });
  };

  // Helper to update active Interlock config
  const handleInterlockChange = (
    tagId: number,
    field: keyof InterlockConfig,
    value: number,
  ) => {
    const updatedInterlocks = config.interlocks.map((interlock, idx) => {
      if (idx === tagId) {
        return { ...interlock, [field]: value };
      }
      return interlock;
    });
    setConfig({ ...config, interlocks: updatedInterlocks });
  };

  // Fetch current Alicats configurations
  const fetchAlicats = async () => {
    try {
      setAlicats(await api.getAlicats());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchAlarmsAndEvents = async () => {
    try {
      const [alarms, events] = await Promise.all([
        api.getActiveAlarms(),
        api.getEvents(50),
      ]);
      setActiveAlarms(alarms);
      setEventsHistory(events);
    } catch (err) {
      console.error("Failed to fetch alarms and events", err);
    }
  };

  const handleAcknowledgeAlarm = async (tagId: string) => {
    try {
      await api.acknowledgeAlarm(tagId);
      triggerNotification(`Alarm on Tag ${tagId} acknowledged`, "success");
      fetchAlarmsAndEvents();
    } catch {
      triggerNotification("Failed to acknowledge alarm", "error");
    }
  };

  const handleAlicatSetpoint = async (deviceId: string, val: number) => {
    try {
      await api.setAlicatSetpoint(deviceId, val);
      setAlicats((prev) =>
        prev.map((m) =>
          m.device_id === deviceId ? { ...m, setpoint: val } : m,
        ),
      );
      triggerNotification(
        `Setpoint for MFC ${deviceId} set to ${val} SLPM.`,
        "success",
      );
    } catch {
      triggerNotification("Failed to update Alicat setpoint.", "error");
    }
  };

  const handleAlicatGas = async (deviceId: string, gas: string) => {
    try {
      await api.setAlicatGas(deviceId, gas);
      setAlicats((prev) =>
        prev.map((m) => (m.device_id === deviceId ? { ...m, gas } : m)),
      );
      triggerNotification(
        `Alicat MFC ${deviceId} gas calibration set to ${gas}.`,
        "success",
      );
    } catch {
      triggerNotification("Failed to update Alicat gas calibration.", "error");
    }
  };

  // PID Tuning & MPC Helper Functions
  const startTuning = async (index: number) => {
    try {
      await api.startTuning(index);
      setIsTuningMode(true);
      setTuningResults(null);
      triggerNotification(
        `Tuning session started for PID Loop ${index + 1}. Automatic control decoupled.`,
        "success",
      );
    } catch {
      triggerNotification("Failed to start tuning session.", "error");
    }
  };

  const stepTuning = async (index: number, val: number) => {
    try {
      await api.stepTuning(index, val);
      triggerNotification(`Tuning step change applied: CV = ${val}`, "success");
    } catch {
      triggerNotification("Failed to apply tuning step change.", "error");
    }
  };

  const stopTuning = async (index: number) => {
    try {
      const data = await api.stopTuning(index);
      setTuningResults(data);
      setIsTuningMode(false);
      if (data.status === "success") {
        triggerNotification(
          "Tuning session stopped. Process parameters identified.",
          "success",
        );
      } else {
        triggerNotification(data.message || "Tuning session stopped.", "info");
      }
    } catch {
      triggerNotification("Failed to stop tuning session.", "error");
    }
  };

  const applyRecommendedGains = (index: number) => {
    if (!tuningResults || !tuningResults.recommended_pid) return;
    const { kp, ki, kd } = tuningResults.recommended_pid;
    const updatedPids = config.pids.map((pid, idx) => {
      if (idx === index) {
        return { ...pid, kp, ki, kd };
      }
      return pid;
    });
    setConfig({ ...config, pids: updatedPids });
    triggerNotification(
      `Applied recommended gains (Kp: ${kp}, Ki: ${ki}, Kd: ${kd}) to Loop ${index + 1}.`,
      "info",
    );
  };

  const runMpcSimulation = async () => {
    try {
      const data = await api.simulateMpc(mpcParams);
      setMpcSimData(data);
      triggerNotification("MPC vs PID simulation complete.", "success");
    } catch {
      triggerNotification("Failed to execute MPC simulation.", "error");
    }
  };

  const handleTabVisibilityToggle = (tab: TabId) => {
    const nextVisible = { ...visibleTabs, [tab]: !visibleTabs[tab] };
    const anyVisible = Object.values(nextVisible).some(Boolean);
    if (!anyVisible) {
      triggerNotification("At least one tab must remain visible.", "error");
      return;
    }
    setVisibleTabs(nextVisible);
    if (activeTab === tab && !nextVisible[tab]) {
      const firstVisible = (Object.keys(nextVisible) as TabId[]).find(
        (k) => nextVisible[k],
      );
      setActiveTab(firstVisible ?? "trends");
    }
  };

  // Initial REST fetches on mount. The live WebSocket stream is owned by the
  // useTelemetryStream hook above; this only pulls the one-shot config/registry.
  useEffect(() => {
    fetchConfig();
    fetchAlicats();
    fetchAllTags();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="dashboard-container">
      {/* Top Banner Notification */}
      <NotificationBanner notification={notification} />

      {/* Sticky header — always visible so the E-stop is always one click away */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "sticky",
          top: 0,
          // Above the inspector drawer (200) and its backdrop (199) so the
          // E-stop in this header is ALWAYS clickable, even with a drawer open.
          zIndex: 300,
          background: "var(--bg-color)",
          marginBottom: "1rem",
          paddingTop: "0.85rem",
          paddingBottom: "0.85rem",
          borderBottom: "1px solid var(--panel-border)",
          boxShadow: "0 6px 14px -10px rgba(0,0,0,0.5)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              background: "var(--accent-cyan)",
              width: "36px",
              height: "36px",
              borderRadius: "4px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Activity color="#0f172a" size={20} />
          </div>
          <div>
            <h1
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "1.25rem",
                fontWeight: 800,
                letterSpacing: "0.5px",
                textTransform: "uppercase",
              }}
            >
              P1AM SCADA System
            </h1>
            <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
              Trend Monitoring & Control Interface
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          {/* Connection status indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span
              className={`status-indicator ${
                isConnected ? "status-connected" : "status-disconnected"
              }`}
            />
            <span
              style={{
                fontSize: "0.8rem",
                fontWeight: 700,
                color: "var(--text-secondary)",
              }}
            >
              {isConnected ? "CONNECTED" : "OFFLINE"}
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="btn"
            style={{ padding: "0.5rem" }}
            title="Toggle light/dark theme"
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
          </button>

          {/* Settings Gear */}
          <button
            type="button"
            onClick={() =>
              setInspectorView(
                inspectorView.type === "settings"
                  ? { type: "none" }
                  : { type: "settings" },
              )
            }
            className="btn"
            style={{
              padding: "0.5rem",
              color:
                inspectorView.type === "settings"
                  ? "var(--accent-cyan)"
                  : "inherit",
            }}
            title="Configure visible tabs and view help docs"
          >
            <Settings size={14} />
          </button>

          <EStopButton
            eStopActive={eStopActive}
            onTriggerEStop={handleEStop}
            onClearEStop={handleClearEStop}
          />
        </div>
      </header>

      <AlarmsHeader
        activeAlarms={activeAlarms}
        onAcknowledgeAll={async () => {
          for (const a of activeAlarms) {
            if (!a.acknowledged) {
              await handleAcknowledgeAlarm(a.tag_id);
            }
          }
        }}
      />

      {/* Main content — the inspector is now a slide-in drawer (below), not a column */}
      <div className="main-layout-grid" style={{ gridTemplateColumns: "1fr" }}>
        {/* Left Column (Master Dashboard elements) */}
        <div
          style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}
        >
          {/* Tabbed Navigation Bar (centralized TABS array, #3546) */}
          <TabBar
            activeTab={activeTab}
            visibleTabs={visibleTabs}
            onSelect={setActiveTab}
          />

          {activeTab === "powerSupply" && visibleTabs.powerSupply && (
            <PowerSupplyControl
              liveStatus={powerSupplyStatus}
              onExport={() => setInspectorView({ type: "export" })}
            />
          )}

          {activeTab === "temperature" && visibleTabs.temperature && (
            <TemperatureControl liveStatus={temperatureStatus} />
          )}

          {/* Render Tab Contents */}
          {activeTab === "trends" && visibleTabs.trends && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1.25rem",
              }}
            >
              {/* Live Customizable Graph */}
              <TrendChart history={history} tagValues={tagValues} />

              {/* 32 Tag Broker Monitor Grid */}
              <section className="glass-panel">
                <div className="panel-header">
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <Settings size={16} color="var(--accent-magenta)" />
                    <span>Signal Monitors</span>
                  </div>
                  <span className="tooltip-container">
                    <Info size={14} color="var(--text-muted)" />
                    <span className="tooltip-text">
                      Click any tag cell to inspect safety limits or write a
                      manual override value in the inspector sidebar.
                    </span>
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(auto-fill, minmax(130px, 1fr))",
                    gap: "0.6rem",
                  }}
                >
                  {TAG_INDICES.map((i) => {
                    const val = tagValues[i] ?? 0.0;
                    const interlock = config.interlocks[i];
                    const isTripped =
                      interlock &&
                      (val > interlock.high_limit || val < interlock.low_limit);

                    return (
                      <div
                        key={i}
                        className="tag-monitor-card"
                        style={{
                          borderColor: isTripped
                            ? "var(--color-error)"
                            : "var(--tag-card-border)",
                        }}
                        onClick={() => {
                          setInspectorView({ type: "tag", tagId: i });
                          setOverrideVal(val.toFixed(2));
                        }}
                      >
                        <div
                          style={{
                            fontSize: "0.7rem",
                            color: "var(--text-muted)",
                            textTransform: "uppercase",
                            fontWeight: 700,
                            marginBottom: "0.15rem",
                          }}
                        >
                          Tag {i}
                        </div>
                        <div
                          className="mono-text"
                          style={{
                            fontSize: "1.05rem",
                            fontWeight: 700,
                            color: isTripped
                              ? "var(--color-error)"
                              : "var(--accent-cyan)",
                          }}
                        >
                          {val.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            </div>
          )}

          {activeTab === "controllers" && visibleTabs.controllers && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1.25rem",
              }}
            >
              {/* PID Loop Cards Row */}
              <div className="glass-panel">
                <div className="panel-header">
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <Sliders size={16} color="var(--accent-purple)" />
                    <span>PID Controllers</span>
                  </div>
                  <span className="tooltip-container">
                    <Info size={14} color="var(--text-muted)" />
                    <span className="tooltip-text">
                      Click any controller card here to tune parameters, assign
                      process variables, and configure output registers in the
                      inspector sidebar.
                    </span>
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: "1rem",
                  }}
                >
                  {config.pids.map((pid, idx) => {
                    const pvVal = tagValues[pid.pv_tag_id] ?? 0.0;
                    const cvVal = tagValues[pid.cv_tag_id] ?? 0.0;
                    return (
                      <div
                        key={idx}
                        className="tag-monitor-card"
                        style={{
                          borderLeft: "3px solid var(--accent-cyan)",
                          textAlign: "left",
                          padding: "0.75rem",
                        }}
                        onClick={() => {
                          setInspectorView({ type: "pid", index: idx });
                        }}
                      >
                        <div
                          style={{
                            fontWeight: 800,
                            fontSize: "0.8rem",
                            textTransform: "uppercase",
                            color: "var(--text-primary)",
                            marginBottom: "0.35rem",
                          }}
                        >
                          PID Loop {idx + 1}
                        </div>
                        <div
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--text-secondary)",
                            display: "flex",
                            justifyContent: "space-between",
                          }}
                        >
                          <span>SP: {pid.setpoint.toFixed(1)}</span>
                          <span>
                            PV: {pvVal.toFixed(1)} (Tag {pid.pv_tag_id})
                          </span>
                        </div>
                        <div
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--text-secondary)",
                            display: "flex",
                            justifyContent: "space-between",
                            marginTop: "0.2rem",
                          }}
                        >
                          <span>
                            CV: {cvVal.toFixed(1)} (Tag {pid.cv_tag_id})
                          </span>
                          <span>Kp: {pid.kp.toFixed(2)}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Alicat Mass Flow Controllers */}
              <div className="glass-panel">
                <div className="panel-header">
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <Sliders size={16} color="var(--color-warning)" />
                    <span>Alicat Mass Flow Controllers</span>
                  </div>
                  <span className="tooltip-container">
                    <Info size={14} color="var(--text-muted)" />
                    <span className="tooltip-text">
                      Monitor and adjust R&D facility Mass Flow Controllers.
                      Click a card to control setpoints and select active gas
                      calibrations in the inspector sidebar.
                    </span>
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: "1rem",
                  }}
                >
                  {alicats.map((mfc) => (
                    <div
                      key={mfc.device_id}
                      className="tag-monitor-card"
                      style={{
                        borderLeft: "3px solid var(--color-warning)",
                        textAlign: "left",
                        padding: "0.75rem",
                      }}
                      onClick={() => {
                        setInspectorView({
                          type: "alicat",
                          deviceId: mfc.device_id,
                        });
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: "0.35rem",
                        }}
                      >
                        <span
                          style={{
                            fontWeight: 800,
                            fontSize: "0.8rem",
                            textTransform: "uppercase",
                            color: "var(--text-primary)",
                          }}
                        >
                          {mfc.name} ({mfc.device_id})
                        </span>
                        <span
                          style={{
                            fontSize: "0.65rem",
                            padding: "0.1rem 0.35rem",
                            borderRadius: "3px",
                            backgroundColor:
                              mfc.connection_state === "simulated" ||
                              mfc.connection_state === "connected"
                                ? "rgba(16, 185, 129, 0.15)"
                                : "rgba(239, 68, 68, 0.15)",
                            color:
                              mfc.connection_state === "simulated" ||
                              mfc.connection_state === "connected"
                                ? "var(--color-success)"
                                : "var(--color-error)",
                            fontWeight: 700,
                            textTransform: "uppercase",
                          }}
                        >
                          {mfc.connection_state}
                        </span>
                      </div>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr",
                          gap: "0.5rem",
                        }}
                      >
                        <div>
                          <div
                            style={{
                              fontSize: "0.65rem",
                              color: "var(--text-muted)",
                            }}
                          >
                            Mass Flow
                          </div>
                          <div
                            className="mono-text"
                            style={{
                              fontSize: "1.1rem",
                              fontWeight: 700,
                              color: "var(--accent-cyan)",
                            }}
                          >
                            {mfc.mass_flow.toFixed(2)}{" "}
                            <span
                              style={{ fontSize: "0.7rem", fontWeight: 500 }}
                            >
                              SLPM
                            </span>
                          </div>
                        </div>
                        <div>
                          <div
                            style={{
                              fontSize: "0.65rem",
                              color: "var(--text-muted)",
                            }}
                          >
                            Setpoint
                          </div>
                          <div
                            className="mono-text"
                            style={{
                              fontSize: "1.1rem",
                              fontWeight: 700,
                              color: "var(--text-primary)",
                            }}
                          >
                            {mfc.setpoint.toFixed(2)}{" "}
                            <span
                              style={{ fontSize: "0.7rem", fontWeight: 500 }}
                            >
                              SLPM
                            </span>
                          </div>
                        </div>
                      </div>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1.2fr 1fr 1fr",
                          gap: "0.25rem",
                          marginTop: "0.5rem",
                          borderTop: "1px solid var(--panel-border)",
                          paddingTop: "0.5rem",
                          fontSize: "0.7rem",
                          color: "var(--text-secondary)",
                        }}
                      >
                        <div>
                          Gas:{" "}
                          <span
                            style={{
                              fontWeight: 700,
                              color: "var(--text-primary)",
                            }}
                          >
                            {mfc.gas}
                          </span>
                        </div>
                        <div>
                          P:{" "}
                          <span style={{ fontWeight: 700 }}>
                            {mfc.pressure.toFixed(1)}{" "}
                            <span style={{ fontSize: "0.6rem" }}>PSIA</span>
                          </span>
                        </div>
                        <div>
                          T:{" "}
                          <span style={{ fontWeight: 700 }}>
                            {mfc.temperature.toFixed(1)}{" "}
                            <span style={{ fontSize: "0.6rem" }}>°C</span>
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "routing" && visibleTabs.routing && (
            <div
              style={{ display: "flex", gap: "1rem", flexDirection: "column" }}
            >
              <div
                className="glass-panel"
                onClick={() => setInspectorView({ type: "routing" })}
                title="Click to open Routing Matrix Editor"
              >
                <div
                  className="panel-header"
                  style={{
                    borderBottom: "none",
                    marginBottom: 0,
                    paddingBottom: 0,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <Shuffle size={16} color="var(--accent-cyan)" />
                    <span>Signal Routing Matrix</span>
                  </div>
                </div>
                <div style={{ marginTop: "0.75rem" }}>
                  <RoutingMatrix
                    config={config}
                    onUpdate={setConfig}
                    tagValues={tagValues}
                  />
                </div>
              </div>

              <div className="glass-panel h-96">
                <InterlocksPanel
                  interlocks={config.interlocks}
                  onChange={handleInterlockChange}
                  onDeploy={handleDeploy}
                  deploying={deploying}
                />
              </div>
            </div>
          )}

          {activeTab === "events" && visibleTabs.events && (
            <div className="glass-panel h-96">
              <EventLogView events={eventsHistory} />
            </div>
          )}

          {activeTab === "tuning" && visibleTabs.tuning && (
            <TuningPanel
              pids={config.pids}
              selectedTuningLoop={selectedTuningLoop}
              isTuningMode={isTuningMode}
              tuningStepVal={tuningStepVal}
              tuningResults={tuningResults}
              mpcParams={mpcParams}
              mpcSimData={mpcSimData}
              onSelectedTuningLoopChange={setSelectedTuningLoop}
              onTuningStepValChange={setTuningStepVal}
              onMpcParamsChange={setMpcParams}
              onStartTuning={startTuning}
              onStepTuning={stepTuning}
              onStopTuning={stopTuning}
              onApplyRecommendedGains={applyRecommendedGains}
              onRunMpcSimulation={runMpcSimulation}
            />
          )}

          {activeTab === "ladder" && visibleTabs.ladder && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1.25rem",
              }}
            >
              <ProjectImporter
                onImportSuccess={() => {
                  fetchAllTags();
                  fetchConfig();
                }}
                triggerNotification={triggerNotification}
              />
              <LadderExplorer
                onSelectTag={handleSelectTag}
                triggerNotification={triggerNotification}
              />
            </div>
          )}

          {activeTab === "hierarchy" && visibleTabs.hierarchy && (
            <PlantHierarchy
              onSelectTag={handleSelectTag}
              triggerNotification={triggerNotification}
            />
          )}
        </div>

        <InspectorDrawer
          inspectorView={inspectorView}
          setInspectorView={setInspectorView}
          config={config}
          deploying={deploying}
          allTags={allTags}
          tagsDict={tagsDict}
          tagValues={tagValues}
          overrideVal={overrideVal}
          showOverrideConfirm={showOverrideConfirm}
          alicats={alicats}
          alicatSetpointVal={alicatSetpointVal}
          visibleTabs={visibleTabs}
          setOverrideVal={setOverrideVal}
          setShowOverrideConfirm={setShowOverrideConfirm}
          setAlicatSetpointVal={setAlicatSetpointVal}
          executeOverride={executeOverride}
          handleInterlockChange={handleInterlockChange}
          handlePidChange={handlePidChange}
          handleDeploy={handleDeploy}
          handleTabVisibilityToggle={handleTabVisibilityToggle}
          handleAlicatSetpoint={handleAlicatSetpoint}
          handleAlicatGas={handleAlicatGas}
          triggerNotification={triggerNotification}
        />
      </div>
    </div>
  );
};

export default App;
