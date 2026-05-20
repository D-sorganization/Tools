import React, { useState, useEffect, useRef } from "react";
import { RoutingMatrix } from "./components/RoutingMatrix";
import { TrendChart } from "./components/TrendChart";
import {
  AlertOctagon,
  Activity,
  Sliders,
  Shuffle,
  ShieldAlert,
  Download,
  Sun,
  Moon,
  Info,
  BookOpen,
  Settings,
  X,
  FileText,
} from "lucide-react";

export interface PIDConfig {
  pv_tag_id: number;
  cv_tag_id: number;
  setpoint: number;
  kp: number;
  ki: number;
  kd: number;
}

export interface InterlockConfig {
  high_limit: number;
  low_limit: number;
}

export interface RoutingConfig {
  input_routing: number[];
  output_routing: number[];
  pids: PIDConfig[];
  interlocks: InterlockConfig[];
}

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
  interlocks: Array.from({ length: 32 }).map(() => ({
    high_limit: 100.0,
    low_limit: 0.0,
  })),
};

export interface AlicatMFCState {
  device_id: string;
  name: string;
  gas: string;
  setpoint: number;
  mass_flow: number;
  volumetric_flow: number;
  pressure: number;
  temperature: number;
  max_flow: number;
  connection_type: string;
  port_or_ip: string | null;
  connection_state: string;
}

type InspectorState =
  | { type: "none" }
  | { type: "tag"; tagId: number }
  | { type: "pid"; index: number }
  | { type: "routing" }
  | { type: "alicat"; deviceId: string }
  | { type: "settings" };

export const App: React.FC = () => {
  const [config, setConfig] = useState<RoutingConfig>(DEFAULT_CONFIG);
  const [tagValues, setTagValues] = useState<number[]>(Array(32).fill(0.0));
  const [history, setHistory] = useState<number[][]>([]);
  const [alicats, setAlicats] = useState<AlicatMFCState[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [deploying, setDeploying] = useState<boolean>(false);
  const [notification, setNotification] = useState<{
    message: string;
    type: "success" | "error" | "info";
  } | null>(null);

  // Tab Navigation and Visibility State
  const [activeTab, setActiveTab] = useState<string>("trends");
  const [visibleTabs, setVisibleTabs] = useState<{
    trends: boolean;
    controllers: boolean;
    routing: boolean;
    tuning: boolean;
  }>({
    trends: true,
    controllers: true,
    routing: true,
    tuning: true,
  });

  // PID Tuning State
  const [selectedTuningLoop, setSelectedTuningLoop] = useState<number>(0);
  const [isTuningMode, setIsTuningMode] = useState<boolean>(false);
  const [tuningStepVal, setTuningStepVal] = useState<string>("50.0");
  const [tuningResults, setTuningResults] = useState<{
    status: string;
    message: string;
    parameters: { kp: number; tau: number; theta: number };
    recommended_pid: { kp: number; ki: number; kd: number };
  } | null>(null);

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
  const [mpcSimData, setMpcSimData] = useState<{
    status: string;
    time: number[];
    pid: { pv: number[]; cv: number[] };
    mpc: { pv: number[]; cv: number[] };
  } | null>(null);

  // Theme & Navigation Sidebar State
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (localStorage.getItem("theme") as "dark" | "light") || "dark";
  });
  const [inspectorView, setInspectorView] = useState<InspectorState>({ type: "none" });

  // Manual Override Form State inside inspector
  const [overrideVal, setOverrideVal] = useState<string>("0.0");
  const [showOverrideConfirm, setShowOverrideConfirm] = useState<boolean>(false);

  // Alicat MFC form states
  const [alicatSetpointVal, setAlicatSetpointVal] = useState<string>("");

  useEffect(() => {
    if (inspectorView.type === "alicat") {
      const mfc = alicats.find((m) => m.device_id === inspectorView.deviceId);
      if (mfc) {
        setAlicatSetpointVal(mfc.setpoint.toString());
      }
    }
  }, [inspectorView]);

  // Historical Telemetry Downloader State (Inside Default Sidebar view)
  const [exportTags, setExportTags] = useState<string>("0, 1, 10");
  const [exportStart, setExportStart] = useState<string>(() => {
    const d = new Date();
    d.setMinutes(d.getMinutes() - 15);
    const tzOffset = d.getTimezoneOffset() * 60000;
    const localISOTime = new Date(d.getTime() - tzOffset).toISOString().slice(0, 16);
    return localISOTime;
  });
  const [exportEnd, setExportEnd] = useState<string>(() => {
    const d = new Date();
    const tzOffset = d.getTimezoneOffset() * 60000;
    const localISOTime = new Date(d.getTime() - tzOffset).toISOString().slice(0, 16);
    return localISOTime;
  });

  const wsRef = useRef<WebSocket | null>(null);

  // Synchronize CSS custom property set on HTML element
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Show a notification banner
  const triggerNotification = (
    message: string,
    type: "success" | "error" | "info"
  ) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  // Fetch current PLC configuration on mount
  const fetchConfig = async () => {
    try {
      const res = await fetch("/api/routing");
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
        triggerNotification("Loaded active PLC configuration.", "success");
      } else {
        triggerNotification("Failed to fetch routing configuration from PLC.", "error");
      }
    } catch (err) {
      triggerNotification("PLC offline. Unable to read registers.", "error");
    }
  };

  // Deploy configuration & write to NVRAM
  const handleDeploy = async () => {
    setDeploying(true);
    try {
      const res = await fetch("/api/routing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (res.ok) {
        triggerNotification(
          "Configuration deployed & written to NVRAM successfully.",
          "success"
        );
      } else {
        const errData = await res.json();
        triggerNotification(
          `Deployment failed: ${errData.detail || "Modbus error"}`,
          "error"
        );
      }
    } catch (err) {
      triggerNotification("Deployment failed: connection error.", "error");
    } finally {
      setDeploying(false);
    }
  };

  // Trigger global emergency stop (E-stop)
  const handleEStop = async () => {
    try {
      const res = await fetch("/api/estop", { method: "POST" });
      if (res.ok) {
        triggerNotification("EMERGENCY SHUTDOWN COMMAND ISSUED!", "error");
      } else {
        triggerNotification("Failed to issue E-stop command.", "error");
      }
    } catch (err) {
      triggerNotification("Error connecting to SCADA server for E-stop.", "error");
    }
  };

  // Execute direct tag value force override
  const executeOverride = async (tagId: number) => {
    const parsed = parseFloat(overrideVal);
    if (isNaN(parsed)) {
      triggerNotification("Invalid numeric value.", "error");
      return;
    }
    try {
      const res = await fetch(`/api/tags/${tagId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: parsed }),
      });
      if (res.ok) {
        triggerNotification(
          `Successfully forced Tag ${tagId} to ${parsed}`,
          "success"
        );
        setShowOverrideConfirm(false);
      } else {
        const err = await res.json();
        triggerNotification(
          `Override failed: ${err.detail || "PLC write failed"}`,
          "error"
        );
      }
    } catch (err) {
      triggerNotification("Failed to connect to PLC backend.", "error");
    }
  };

  // Download CSV logs
  const handleDownloadCSV = () => {
    const startISO = new Date(exportStart).toISOString();
    const endISO = new Date(exportEnd).toISOString();
    const cleanedTags = exportTags
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .join(",");

    if (!cleanedTags) {
      triggerNotification("Please enter at least one Tag ID.", "error");
      return;
    }

    const url = `/api/export?tag_ids=${encodeURIComponent(
      cleanedTags
    )}&start_time=${encodeURIComponent(startISO)}&end_time=${encodeURIComponent(
      endISO
    )}`;
    window.open(url, "_blank");
  };

  // Helper to update active PID config
  const handlePidChange = (index: number, field: keyof PIDConfig, value: number) => {
    const updatedPids = config.pids.map((pid, idx) => {
      if (idx === index) {
        return { ...pid, [field]: value };
      }
      return pid;
    });
    setConfig({ ...config, pids: updatedPids });
  };

  // Helper to update active Interlock config
  const handleInterlockChange = (tagId: number, field: keyof InterlockConfig, value: number) => {
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
      const res = await fetch("/api/alicats");
      if (res.ok) {
        const data = await res.json();
        setAlicats(data);
      }
    } catch (err) {
      console.error("Error fetching Alicat controllers state:", err);
    }
  };

  const handleAlicatSetpoint = async (deviceId: string, val: number) => {
    try {
      const res = await fetch(`/api/alicats/${deviceId}/setpoint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ setpoint: val }),
      });
      if (res.ok) {
        setAlicats((prev) =>
          prev.map((m) => (m.device_id === deviceId ? { ...m, setpoint: val } : m))
        );
        triggerNotification(`Setpoint for MFC ${deviceId} set to ${val} SLPM.`, "success");
      } else {
        triggerNotification("Failed to update Alicat setpoint.", "error");
      }
    } catch (err) {
      triggerNotification("Connection error updating setpoint.", "error");
    }
  };

  const handleAlicatGas = async (deviceId: string, gas: string) => {
    try {
      const res = await fetch(`/api/alicats/${deviceId}/gas`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gas }),
      });
      if (res.ok) {
        setAlicats((prev) =>
          prev.map((m) => (m.device_id === deviceId ? { ...m, gas } : m))
        );
        triggerNotification(`Alicat MFC ${deviceId} gas calibration set to ${gas}.`, "success");
      } else {
        triggerNotification("Failed to update Alicat gas calibration.", "error");
      }
    } catch (err) {
      triggerNotification("Connection error updating gas type.", "error");
    }
  };

  // PID Tuning & MPC Helper Functions
  const startTuning = async (index: number) => {
    try {
      const res = await fetch(`/api/pid/${index}/tuning/start`, { method: "POST" });
      if (res.ok) {
        setIsTuningMode(true);
        setTuningResults(null);
        triggerNotification(`Tuning session started for PID Loop ${index + 1}. Automatic control decoupled.`, "success");
      } else {
        triggerNotification("Failed to start tuning session.", "error");
      }
    } catch (err) {
      triggerNotification("Connection error starting tuning.", "error");
    }
  };

  const stepTuning = async (index: number, val: number) => {
    try {
      const res = await fetch(`/api/pid/${index}/tuning/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step_value: val }),
      });
      if (res.ok) {
        triggerNotification(`Tuning step change applied: CV = ${val}`, "success");
      } else {
        triggerNotification("Failed to apply tuning step change.", "error");
      }
    } catch (err) {
      triggerNotification("Connection error applying step change.", "error");
    }
  };

  const stopTuning = async (index: number) => {
    try {
      const res = await fetch(`/api/pid/${index}/tuning/stop`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setTuningResults(data);
        setIsTuningMode(false);
        if (data.status === "success") {
          triggerNotification("Tuning session stopped. Process parameters identified.", "success");
        } else {
          triggerNotification(data.message || "Tuning session stopped.", "info");
        }
      } else {
        triggerNotification("Failed to stop tuning session.", "error");
      }
    } catch (err) {
      triggerNotification("Connection error stopping tuning.", "error");
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
    triggerNotification(`Applied recommended gains (Kp: ${kp}, Ki: ${ki}, Kd: ${kd}) to Loop ${index + 1}.`, "info");
  };

  const runMpcSimulation = async () => {
    try {
      const res = await fetch(`/api/mpc/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mpcParams),
      });
      if (res.ok) {
        const data = await res.json();
        setMpcSimData(data);
        triggerNotification("MPC vs PID simulation complete.", "success");
      } else {
        triggerNotification("Failed to execute MPC simulation.", "error");
      }
    } catch (err) {
      triggerNotification("Connection error running MPC simulation.", "error");
    }
  };

  const handleTabVisibilityToggle = (tabName: keyof typeof visibleTabs) => {
    const nextVisible = { ...visibleTabs, [tabName]: !visibleTabs[tabName] };
    const anyVisible = Object.values(nextVisible).some(Boolean);
    if (!anyVisible) {
      triggerNotification("At least one tab must remain visible.", "error");
      return;
    }
    setVisibleTabs(nextVisible);
    if (activeTab === tabName && !nextVisible[tabName]) {
      const firstVisible = Object.keys(nextVisible).find((k) => nextVisible[k as keyof typeof visibleTabs]) || "trends";
      setActiveTab(firstVisible);
    }
  };

  // Set up WebSocket connections for live metrics stream
  useEffect(() => {
    fetchConfig();
    fetchAlicats();

    const connectWebSocket = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/api/stream`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        triggerNotification("SCADA live stream connected.", "info");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data && typeof data === "object" && "tags" in data) {
            const rawValues = data.tags;
            if (Array.isArray(rawValues) && rawValues.length === 32) {
              setTagValues(rawValues);
              setHistory((prev) => {
                const updated = [...prev, rawValues];
                if (updated.length > 1200) {
                  updated.shift();
                }
                return updated;
              });
            }
            if (Array.isArray(data.alicats)) {
              setAlicats(data.alicats);
            }
          } else {
            // Fallback for simple legacy tag arrays
            const rawValues = data;
            if (Array.isArray(rawValues) && rawValues.length === 32) {
              setTagValues(rawValues);
              setHistory((prev) => {
                const updated = [...prev, rawValues];
                if (updated.length > 1200) {
                  updated.shift();
                }
                return updated;
              });
            }
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="dashboard-container">
      {/* Top Banner Notification */}
      {notification && (
        <div
          style={{
            position: "fixed",
            top: "1.25rem",
            right: "1.25rem",
            zIndex: 9999,
            padding: "0.75rem 1.25rem",
            borderRadius: "4px",
            fontSize: "0.85rem",
            fontWeight: 500,
            borderLeft: "4px solid",
            color: "#ffffff",
            backgroundColor:
              notification.type === "success"
                ? "rgba(16, 185, 129, 0.2)"
                : notification.type === "error"
                ? "rgba(239, 68, 68, 0.25)"
                : "rgba(56, 189, 248, 0.15)",
            borderColor:
              notification.type === "success"
                ? "var(--color-success)"
                : notification.type === "error"
                ? "var(--color-error)"
                : "var(--accent-cyan)",
          }}
        >
          {notification.message}
        </div>
      )}

      {/* Cybernetic Flat Header */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "1.25rem",
          paddingBottom: "0.85rem",
          borderBottom: "1px solid var(--panel-border)",
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
            onClick={() => setInspectorView(inspectorView.type === "settings" ? { type: "none" } : { type: "settings" })}
            className="btn"
            style={{ padding: "0.5rem", color: inspectorView.type === "settings" ? "var(--accent-cyan)" : "inherit" }}
            title="Configure visible tabs and view help docs"
          >
            <Settings size={14} />
          </button>

          {/* E-Stop Button - Solid, flat, no shadows or blinking animations */}
          <button
            type="button"
            onClick={handleEStop}
            className="btn btn-estop"
            style={{
              padding: "0.5rem 1.25rem",
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
            }}
          >
            <AlertOctagon size={16} />
            EMERGENCY ESTOP
          </button>
        </div>
      </header>

      {/* Main Two-Column Master-Detail Layout */}
      <div className="main-layout-grid">
        {/* Left Column (Master Dashboard elements) */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Tabbed Navigation Bar */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--panel-border)", gap: "0.25rem", paddingBottom: "0.25rem", marginBottom: "0.5rem" }}>
            {visibleTabs.trends && (
              <button
                type="button"
                className={`tab-btn ${activeTab === "trends" ? "active" : ""}`}
                onClick={() => setActiveTab("trends")}
                style={{
                  background: "none",
                  border: "none",
                  color: activeTab === "trends" ? "var(--accent-cyan)" : "var(--text-secondary)",
                  padding: "0.5rem 1rem",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  borderBottom: activeTab === "trends" ? "2px solid var(--accent-cyan)" : "2px solid transparent",
                  transition: "all var(--transition-fast)",
                }}
              >
                Trends & Monitors
              </button>
            )}
            {visibleTabs.controllers && (
              <button
                type="button"
                className={`tab-btn ${activeTab === "controllers" ? "active" : ""}`}
                onClick={() => setActiveTab("controllers")}
                style={{
                  background: "none",
                  border: "none",
                  color: activeTab === "controllers" ? "var(--accent-purple)" : "var(--text-secondary)",
                  padding: "0.5rem 1rem",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  borderBottom: activeTab === "controllers" ? "2px solid var(--accent-purple)" : "2px solid transparent",
                  transition: "all var(--transition-fast)",
                }}
              >
                PID & Mass Flow
              </button>
            )}
            {visibleTabs.routing && (
              <button
                type="button"
                className={`tab-btn ${activeTab === "routing" ? "active" : ""}`}
                onClick={() => setActiveTab("routing")}
                style={{
                  background: "none",
                  border: "none",
                  color: activeTab === "routing" ? "var(--accent-magenta)" : "var(--text-secondary)",
                  padding: "0.5rem 1rem",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  borderBottom: activeTab === "routing" ? "2px solid var(--accent-magenta)" : "2px solid transparent",
                  transition: "all var(--transition-fast)",
                }}
              >
                Signal Routing
              </button>
            )}
            {visibleTabs.tuning && (
              <button
                type="button"
                className={`tab-btn ${activeTab === "tuning" ? "active" : ""}`}
                onClick={() => setActiveTab("tuning")}
                style={{
                  background: "none",
                  border: "none",
                  color: activeTab === "tuning" ? "var(--accent-cyan)" : "var(--text-secondary)",
                  padding: "0.5rem 1rem",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  borderBottom: activeTab === "tuning" ? "2px solid var(--accent-cyan)" : "2px solid transparent",
                  transition: "all var(--transition-fast)",
                }}
              >
                Tuning & MPC
              </button>
            )}
          </div>

          {/* Render Tab Contents */}
          {activeTab === "trends" && visibleTabs.trends && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
              {/* Live Customizable Graph */}
              <TrendChart history={history} tagValues={tagValues} />

              {/* 32 Tag Broker Monitor Grid */}
              <section className="glass-panel">
                <div className="panel-header">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Settings size={16} color="var(--accent-magenta)" />
                    <span>Signal Monitors</span>
                  </div>
                  <span className="tooltip-container">
                    <Info size={14} color="var(--text-muted)" />
                    <span className="tooltip-text">Click any tag cell to inspect safety limits or write a manual override value in the inspector sidebar.</span>
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
                    gap: "0.6rem",
                  }}
                >
                  {Array.from({ length: 32 }).map((_, i) => {
                    const val = tagValues[i] ?? 0.0;
                    const interlock = config.interlocks[i];
                    const isTripped =
                      interlock && (val > interlock.high_limit || val < interlock.low_limit);

                    return (
                      <div
                        key={i}
                        className="tag-monitor-card"
                        style={{
                          borderColor: isTripped ? "var(--color-error)" : "var(--tag-card-border)",
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
                            color: isTripped ? "var(--color-error)" : "var(--accent-cyan)",
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
            <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
              {/* PID Loop Cards Row */}
              <div className="glass-panel">
                <div className="panel-header">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Sliders size={16} color="var(--accent-purple)" />
                    <span>PID Controllers</span>
                  </div>
                  <span className="tooltip-container">
                    <Info size={14} color="var(--text-muted)" />
                    <span className="tooltip-text">Click any controller card here to tune parameters, assign process variables, and configure output registers in the inspector sidebar.</span>
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
                        <div style={{ fontWeight: 800, fontSize: "0.8rem", textTransform: "uppercase", color: "var(--text-primary)", marginBottom: "0.35rem" }}>
                          PID Loop {idx + 1}
                        </div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", justifyContent: "space-between" }}>
                          <span>SP: {pid.setpoint.toFixed(1)}</span>
                          <span>PV: {pvVal.toFixed(1)} (Tag {pid.pv_tag_id})</span>
                        </div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", justifyContent: "space-between", marginTop: "0.2rem" }}>
                          <span>CV: {cvVal.toFixed(1)} (Tag {pid.cv_tag_id})</span>
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
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Sliders size={16} color="var(--color-warning)" />
                    <span>Alicat Mass Flow Controllers</span>
                  </div>
                  <span className="tooltip-container">
                    <Info size={14} color="var(--text-muted)" />
                    <span className="tooltip-text">Monitor and adjust R&D facility Mass Flow Controllers. Click a card to control setpoints and select active gas calibrations in the inspector sidebar.</span>
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
                        setInspectorView({ type: "alicat", deviceId: mfc.device_id });
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.35rem" }}>
                        <span style={{ fontWeight: 800, fontSize: "0.8rem", textTransform: "uppercase", color: "var(--text-primary)" }}>
                          {mfc.name} ({mfc.device_id})
                        </span>
                        <span
                          style={{
                            fontSize: "0.65rem",
                            padding: "0.1rem 0.35rem",
                            borderRadius: "3px",
                            backgroundColor: mfc.connection_state === "simulated" || mfc.connection_state === "connected" ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
                            color: mfc.connection_state === "simulated" || mfc.connection_state === "connected" ? "var(--color-success)" : "var(--color-error)",
                            fontWeight: 700,
                            textTransform: "uppercase",
                          }}
                        >
                          {mfc.connection_state}
                        </span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                        <div>
                          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Mass Flow</div>
                          <div className="mono-text" style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                            {mfc.mass_flow.toFixed(2)} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>SLPM</span>
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Setpoint</div>
                          <div className="mono-text" style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)" }}>
                            {mfc.setpoint.toFixed(2)} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>SLPM</span>
                          </div>
                        </div>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr 1fr", gap: "0.25rem", marginTop: "0.5rem", borderTop: "1px solid var(--panel-border)", paddingTop: "0.5rem", fontSize: "0.7rem", color: "var(--text-secondary)" }}>
                        <div>Gas: <span style={{ fontWeight: 700, color: "var(--text-primary)" }}>{mfc.gas}</span></div>
                        <div>P: <span style={{ fontWeight: 700 }}>{mfc.pressure.toFixed(1)} <span style={{ fontSize: "0.6rem" }}>PSIA</span></span></div>
                        <div>T: <span style={{ fontWeight: 700 }}>{mfc.temperature.toFixed(1)} <span style={{ fontSize: "0.6rem" }}>°C</span></span></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "routing" && visibleTabs.routing && (
            <div
              className="glass-panel"
              onClick={() => setInspectorView({ type: "routing" })}
              title="Click to open Routing Matrix Editor"
            >
              <div className="panel-header" style={{ borderBottom: "none", marginBottom: 0, paddingBottom: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Shuffle size={16} color="var(--accent-cyan)" />
                  <span>Signal Routing Matrix</span>
                </div>
                <span className="tooltip-container">
                  <Info size={14} color="var(--text-muted)" />
                  <span className="tooltip-text">Configure routing patches linking hardware channels to registers. Click down on any patch row.</span>
                </span>
              </div>
              <div style={{ marginTop: "0.75rem" }}>
                <RoutingMatrix config={config} onUpdate={setConfig} tagValues={tagValues} />
              </div>
            </div>
          )}

          {activeTab === "tuning" && visibleTabs.tuning && (
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
          )}
        </div>

        {/* Right Column (Sticky Inspector Sidebar Panel) */}
        <aside
          style={{
            position: "sticky",
            top: "1.25rem",
            maxHeight: "calc(100vh - 2.5rem)",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "1.25rem",
          }}
        >
          {/* Main inspector panel */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="panel-header" style={{ borderBottom: "1px solid var(--panel-border)", paddingBottom: "0.5rem" }}>
              <span>Inspector Panel</span>
              {inspectorView.type !== "none" && (
                <button
                  type="button"
                  onClick={() => setInspectorView({ type: "none" })}
                  style={{ background: "none", border: "none", color: "var(--text-secondary)", cursor: "pointer" }}
                >
                  <X size={16} />
                </button>
              )}
            </div>

            {/* Render details depending on selection */}
            {inspectorView.type === "none" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                <div>
                  <h3 style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "0.35rem", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                    <Info size={14} color="var(--accent-cyan)" />
                    <span>Inspector Guide</span>
                  </h3>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Click on any Tag, PID Loop, or Routing matrix on the main screen to inspect safety details, tune controllers, or issue manual overrides.
                  </p>
                </div>

                {/* CSV Log Exporter inside Default Sidebar view */}
                <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "1rem" }}>
                  <h3 style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "0.75rem", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                    <FileText size={14} color="var(--accent-purple)" />
                    <span>CSV Data Exporter</span>
                  </h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    <div className="input-group">
                      <label className="input-label">Tags (comma-separated)</label>
                      <input
                        type="text"
                        className="form-input"
                        value={exportTags}
                        onChange={(e) => setExportTags(e.target.value)}
                        placeholder="e.g. 0,1,10"
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Start Time</label>
                      <input
                        type="datetime-local"
                        className="form-input"
                        value={exportStart}
                        onChange={(e) => setExportStart(e.target.value)}
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">End Time</label>
                      <input
                        type="datetime-local"
                        className="form-input"
                        value={exportEnd}
                        onChange={(e) => setExportEnd(e.target.value)}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={handleDownloadCSV}
                      className="btn"
                      style={{ width: "100%", padding: "0.45rem", fontSize: "0.8rem" }}
                    >
                      <Download size={14} />
                      Export Log Data
                    </button>
                  </div>
                </div>

                {/* Hardware Reference Docs Drawer */}
                <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "1rem" }}>
                  <h3 style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "0.6rem", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                    <BookOpen size={14} color="var(--accent-magenta)" />
                    <span>P1AM Hardware Reference Docs</span>
                  </h3>
                  <ul style={{ fontSize: "0.8rem", listStyleType: "none", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                    <li>
                      <a href="https://www.automationdirect.com/pn/P1AM-100" target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-cyan)", textDecoration: "none" }}>
                        P1AM-100 CPU module Manual
                      </a>
                    </li>
                    <li>
                      <a href="https://www.automationdirect.com/pn/P1AM-ETH" target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-cyan)", textDecoration: "none" }}>
                        P1AM-ETH Ethernet module
                      </a>
                    </li>
                    <li>
                      <a href="https://www.automationdirect.com/pn/P1-04THM" target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-cyan)", textDecoration: "none" }}>
                        P1-04THM Thermocouple card
                      </a>
                    </li>
                    <li>
                      <a href="https://www.automationdirect.com/pn/P1-4ADL2DAL-1" target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-cyan)", textDecoration: "none" }}>
                        P1-4ADL2DAL-1 Analog I/O
                      </a>
                    </li>
                    <li>
                      <a href="https://www.automationdirect.com/pn/P1AM-GPIO" target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-cyan)", textDecoration: "none" }}>
                        P1AM-GPIO interface shield
                      </a>
                    </li>
                  </ul>
                </div>
              </div>
            )}

            {/* Tag register editor */}
            {inspectorView.type === "tag" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase" }}>
                    Tag Register #{inspectorView.tagId}
                  </h3>
                  <div style={{ display: "flex", justifyContent: "space-between", background: "var(--input-bg)", padding: "0.5rem", borderRadius: "4px", marginTop: "0.5rem", border: "1px solid var(--panel-border)" }}>
                    <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Current Value:</span>
                    <span className="mono-text" style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                      {tagValues[inspectorView.tagId]?.toFixed(2) ?? "0.00"}
                    </span>
                  </div>
                </div>

                {/* Section 1: Manual Value Override (Write Force) */}
                <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", marginBottom: "0.5rem" }}>
                    <ShieldAlert size={14} color="var(--color-warning)" />
                    <span style={{ fontSize: "0.8rem", fontWeight: 700, textTransform: "uppercase" }}>Manual Force Override</span>
                  </div>
                  <div className="input-group">
                    <label className="input-label">Override Force Value</label>
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
                    style={{ width: "100%", color: "var(--color-warning)", borderColor: "var(--color-warning)", padding: "0.45rem", fontSize: "0.8rem" }}
                  >
                    Force Register Write
                  </button>

                  {showOverrideConfirm && (
                    <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid var(--color-error)", borderRadius: "4px", padding: "0.6rem", marginTop: "0.5rem" }}>
                      <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--color-error)", marginBottom: "0.25rem" }}>Confirm Direct Write</div>
                      <p style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginBottom: "0.6rem" }}>
                        Forcing Tag {inspectorView.tagId} to {overrideVal} will overwrite normal logic. Continue?
                      </p>
                      <div style={{ display: "flex", gap: "0.4rem" }}>
                        <button
                          type="button"
                          onClick={() => executeOverride(inspectorView.tagId)}
                          className="btn"
                          style={{ background: "var(--color-error)", border: "none", color: "#ffffff", padding: "0.2rem 0.5rem", fontSize: "0.75rem" }}
                        >
                          Confirm
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowOverrideConfirm(false)}
                          className="btn"
                          style={{ padding: "0.2rem 0.5rem", fontSize: "0.75rem" }}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Section 2: Safety Interlocks Boundaries */}
                <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", marginBottom: "0.5rem" }}>
                    <Sliders size={14} color="var(--accent-purple)" />
                    <span style={{ fontSize: "0.8rem", fontWeight: 700, textTransform: "uppercase" }}>Safety Limits Interlocks</span>
                  </div>
                  <div className="input-group">
                    <label className="input-label">High Trip Limit (Alarm High)</label>
                    <input
                      type="number"
                      step="0.5"
                      className="form-input"
                      value={config.interlocks[inspectorView.tagId]?.high_limit ?? 100.0}
                      onChange={(e) => handleInterlockChange(inspectorView.tagId, "high_limit", Number(e.target.value))}
                    />
                  </div>
                  <div className="input-group">
                    <label className="input-label">Low Trip Limit (Alarm Low)</label>
                    <input
                      type="number"
                      step="0.5"
                      className="form-input"
                      value={config.interlocks[inspectorView.tagId]?.low_limit ?? 0.0}
                      onChange={(e) => handleInterlockChange(inspectorView.tagId, "low_limit", Number(e.target.value))}
                    />
                  </div>
                </div>

                {/* Apply Panel Changes */}
                <button
                  type="button"
                  onClick={handleDeploy}
                  disabled={deploying}
                  className="btn btn-primary"
                  style={{ width: "100%", padding: "0.5rem", fontSize: "0.85rem", marginTop: "0.5rem" }}
                >
                  {deploying ? "Deploying Configuration..." : "Commit Safety Limits"}
                </button>
              </div>
            )}

            {/* PID Loop Tune editor */}
            {inspectorView.type === "pid" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-purple)", textTransform: "uppercase" }}>
                    Tune PID Loop {inspectorView.index + 1}
                  </h3>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                  <div className="input-group">
                    <label className="input-label">PV Tag ID</label>
                    <select
                      className="form-input"
                      value={config.pids[inspectorView.index].pv_tag_id}
                      onChange={(e) => handlePidChange(inspectorView.index, "pv_tag_id", Number(e.target.value))}
                    >
                      {Array.from({ length: 32 }).map((_, i) => (
                        <option key={i} value={i}>Tag {i}</option>
                      ))}
                    </select>
                  </div>
                  <div className="input-group">
                    <label className="input-label">CV Tag ID</label>
                    <select
                      className="form-input"
                      value={config.pids[inspectorView.index].cv_tag_id}
                      onChange={(e) => handlePidChange(inspectorView.index, "cv_tag_id", Number(e.target.value))}
                    >
                      {Array.from({ length: 32 }).map((_, i) => (
                        <option key={i} value={i}>Tag {i}</option>
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
                    onChange={(e) => handlePidChange(inspectorView.index, "setpoint", Number(e.target.value))}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Proportional Gain (Kp)</label>
                  <input
                    type="number"
                    step="0.01"
                    className="form-input"
                    value={config.pids[inspectorView.index].kp}
                    onChange={(e) => handlePidChange(inspectorView.index, "kp", Number(e.target.value))}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Integral Gain (Ki)</label>
                  <input
                    type="number"
                    step="0.01"
                    className="form-input"
                    value={config.pids[inspectorView.index].ki}
                    onChange={(e) => handlePidChange(inspectorView.index, "ki", Number(e.target.value))}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Derivative Gain (Kd)</label>
                  <input
                    type="number"
                    step="0.01"
                    className="form-input"
                    value={config.pids[inspectorView.index].kd}
                    onChange={(e) => handlePidChange(inspectorView.index, "kd", Number(e.target.value))}
                  />
                </div>

                {/* Apply Panel Changes */}
                <button
                  type="button"
                  onClick={handleDeploy}
                  disabled={deploying}
                  className="btn btn-primary"
                  style={{ width: "100%", padding: "0.5rem", fontSize: "0.85rem", marginTop: "0.5rem" }}
                >
                  {deploying ? "Deploying Configuration..." : "Commit PID Tuning"}
                </button>
              </div>
            )}

            {/* Routing Matrix Panel (in inspector) */}
            {inspectorView.type === "routing" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase" }}>
                    Routing Matrix Patching
                  </h3>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.4, marginTop: "0.25rem" }}>
                    Modify the crossbar mappings on the main screen matrix. Once matched correctly, click below to deploy the configuration down to the SAMD21 NVRAM.
                  </p>
                </div>

                <div style={{ padding: "0.75rem", background: "rgba(56, 189, 248, 0.08)", border: "1px solid var(--accent-cyan)", borderRadius: "4px" }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-primary)", fontWeight: 700 }}>Modbus Matrix Routing Info:</span>
                  <p style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
                    Input patching links physical thermocouple or analog cards (Channels 0-5) to broker tags. Output patching links broker tags to control variable registers (Channels 10-11) driving solenoid actuators.
                  </p>
                </div>

                {/* Apply Panel Changes */}
                <button
                  type="button"
                  onClick={handleDeploy}
                  disabled={deploying}
                  className="btn btn-primary"
                  style={{ width: "100%", padding: "0.5rem", fontSize: "0.85rem", marginTop: "0.5rem" }}
                >
                  {deploying ? "Deploying Configuration..." : "Commit Matrix Mapping"}
                </button>
              </div>
            )}

            {/* Settings inspector view */}
            {inspectorView.type === "settings" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase" }}>
                    Dashboard Settings
                  </h3>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.4, marginTop: "0.25rem" }}>
                    Configure the SCADA dashboard tab visibility. Toggle tabs to customize your workspace layout.
                  </p>
                </div>

                <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem", display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-primary)", fontWeight: 700, textTransform: "uppercase" }}>
                    Visible Tabs
                  </span>
                  
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={visibleTabs.trends}
                      onChange={() => handleTabVisibilityToggle("trends")}
                    />
                    <span>Live Trends & Monitors</span>
                  </label>

                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={visibleTabs.controllers}
                      onChange={() => handleTabVisibilityToggle("controllers")}
                    />
                    <span>PID Loops & MFCs</span>
                  </label>

                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={visibleTabs.routing}
                      onChange={() => handleTabVisibilityToggle("routing")}
                    />
                    <span>Signal Routing Matrix</span>
                  </label>

                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={visibleTabs.tuning}
                      onChange={() => handleTabVisibilityToggle("tuning")}
                    />
                    <span>Tuning & MPC Groundwork</span>
                  </label>
                </div>

                <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem" }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-primary)", fontWeight: 700, textTransform: "uppercase", display: "block", marginBottom: "0.35rem" }}>
                    Operator Guidelines
                  </span>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                    Maintain safety limit boundary configurations. All parameter deployments are saved to non-volatile memory (NVRAM) and persist across power cycles.
                  </p>
                </div>
              </div>
            )}

            {/* Alicat Mass Flow Controller inspector */}
            {inspectorView.type === "alicat" && (() => {
              const mfc = alicats.find((m) => m.device_id === inspectorView.deviceId);
              if (!mfc) {
                return (
                  <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                    Alicat MFC {inspectorView.deviceId} not found.
                  </div>
                );
              }
              return (
                <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                  <div>
                    <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--color-warning)", textTransform: "uppercase" }}>
                      Inspect {mfc.name}
                    </h3>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
                      Device ID: {mfc.device_id} | State: {mfc.connection_state}
                    </div>
                  </div>

                  {/* Realtime Stats Block */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", background: "var(--input-bg)", padding: "0.75rem", borderRadius: "4px", border: "1px solid var(--panel-border)" }}>
                    <div>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Mass Flow</div>
                      <div className="mono-text" style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                        {mfc.mass_flow.toFixed(2)} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>SLPM</span>
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Vol. Flow</div>
                      <div className="mono-text" style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                        {mfc.volumetric_flow.toFixed(2)} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>LPM</span>
                      </div>
                    </div>
                    <div style={{ marginTop: "0.35rem" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Pressure</div>
                      <div className="mono-text" style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)" }}>
                        {mfc.pressure.toFixed(2)} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>PSIA</span>
                      </div>
                    </div>
                    <div style={{ marginTop: "0.35rem" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Temperature</div>
                      <div className="mono-text" style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)" }}>
                        {mfc.temperature.toFixed(1)} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>°C</span>
                      </div>
                    </div>
                  </div>

                  {/* Form 1: Setpoint command */}
                  <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem" }}>
                    <label className="input-label" style={{ fontWeight: 700, marginBottom: "0.4rem", display: "block" }}>
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
                        style={{ padding: "0.45rem 1rem", fontSize: "0.8rem", whiteSpace: "nowrap" }}
                        onClick={() => {
                          const parsed = parseFloat(alicatSetpointVal);
                          if (!isNaN(parsed) && parsed >= 0 && parsed <= mfc.max_flow) {
                            handleAlicatSetpoint(mfc.device_id, parsed);
                          } else {
                            triggerNotification(`Please enter a valid setpoint between 0 and ${mfc.max_flow}.`, "error");
                          }
                        }}
                      >
                        Set
                      </button>
                    </div>
                    <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                      Maximum flow limit: {mfc.max_flow} SLPM
                    </div>
                  </div>

                  {/* Form 2: Gas select dropdown */}
                  <div style={{ borderTop: "1px solid var(--panel-border)", paddingTop: "0.75rem" }}>
                    <label className="input-label" style={{ fontWeight: 700, marginBottom: "0.4rem", display: "block" }}>
                      Active Gas Calibration
                    </label>
                    <select
                      className="form-input"
                      value={mfc.gas}
                      onChange={(e) => handleAlicatGas(mfc.device_id, e.target.value)}
                    >
                      {["O2", "N2", "CO2", "He", "H2", "Air"].map((species) => (
                        <option key={species} value={species}>
                          {species} ({species === "O2" ? "Oxygen" : species === "N2" ? "Nitrogen" : species === "CO2" ? "Carbon Dioxide" : species === "He" ? "Helium" : species === "H2" ? "Hydrogen" : "Clean Air"})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              );
            })()}
          </div>
        </aside>
      </div>
    </div>
  );
};

export default App;
