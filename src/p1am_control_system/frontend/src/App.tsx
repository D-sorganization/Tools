import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  lazy,
  Suspense,
} from "react";
import { RoutingMatrix } from "./components/RoutingMatrix";
import { TrendChart } from "./components/TrendChart";
import { PanelStack } from "./components/PanelStack";
import { AlarmsHeader } from "./components/AlarmsHeader";
import { EStopButton } from "./components/EStopButton";
import { DataCapturePanel } from "./components/DataCapturePanel";
import { InterlocksPanel } from "./components/InterlocksPanel";
import { EventLogView } from "./components/EventLogView";
import { PowerSupplyControl } from "./components/PowerSupplyControl";
import { TemperatureControl } from "./components/TemperatureControl";
import { SignalDiagnostics } from "./components/SignalDiagnostics";
import { AlicatInspector } from "./components/AlicatInspector";
import { TagInspector } from "./components/TagInspector";
import type { LadderTagInfo } from "./api/schemas";
import { NotificationBanner } from "./components/NotificationBanner";
import { TabBar } from "./components/TabBar";
import { HelpModal } from "./components/HelpModal";
import { CsvExporter } from "./components/CsvExporter";
import { CommsQualityBadge } from "./components/CommsQualityBadge";
import { ProfessionalAlarmPanel } from "./components/ProfessionalAlarmPanel";
import { ConfigurationWorkflowPanel } from "./components/ConfigurationWorkflowPanel";
import { SystemHealthPanel } from "./components/SystemHealthPanel";
import { useTelemetryStream } from "./hooks/useTelemetryStream";
import {
  TABS,
  type TabId,
  loadTabOrder,
  saveTabOrder,
  loadTabVisibility,
  saveTabVisibility,
} from "./lib/tabs";
import { TAG_INDICES, tagName, parseTagId } from "./lib/tags";
import type { InspectorState } from "./lib/inspector";
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
  BookOpen,
  Settings,
  X,
} from "lucide-react";

// Code-split the heavy, occasionally-used reference/config tabs so the Pi's
// browser doesn't parse them on cold load. The Data Explorer in particular pulls
// in the whole plots subsystem; it and the project/ladder/hierarchy tabs load
// on demand (wrapped in a <Suspense> boundary below).
const DataExplorer = lazy(() =>
  import("./components/data_explorer/DataExplorer").then((m) => ({
    default: m.DataExplorer,
  })),
);
const ProjectImporter = lazy(() =>
  import("./components/ProjectImporter").then((m) => ({
    default: m.ProjectImporter,
  })),
);
const LadderExplorer = lazy(() =>
  import("./components/LadderExplorer").then((m) => ({
    default: m.LadderExplorer,
  })),
);
const PlantHierarchy = lazy(() =>
  import("./components/PlantHierarchy").then((m) => ({
    default: m.PlantHierarchy,
  })),
);

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
  const [notification, setNotification] = useState<NotificationState | null>(null);

  // Show a notification banner (declared early so the telemetry hook can use it).
  // Stable identity (empty deps — only calls stable setState) so memoized
  // children that take `triggerNotification` don't re-render every frame.
  const triggerNotification = useCallback(
    (message: string, type: NotificationType) => {
      setNotification({ message, type });
      setTimeout(() => setNotification(null), 5000);
    },
    [],
  );

  // Live telemetry (WS stream + derived state) is owned by a dedicated hook.
  const {
    tagValues,
    history,
    historyTimes,
    tagsDict,
    alicats,
    activeAlarms,
    eStopActive,
    powerSupplyStatus,
    temperatureStatus,
    commsHealth,
    isConnected,
    setAlicats,
    setActiveAlarms,
    setEStopActive,
  } = useTelemetryStream({
    onConnect: () => triggerNotification("SCADA live stream connected.", "info"),
  });

  // Events history (polled separately from the live stream).
  const [eventsHistory, setEventsHistory] = useState<EventLogEntry[]>([]);

  // Tab Navigation, Order, and Visibility State (order + visibility persisted
  // to localStorage so an operator's layout survives reloads).
  const [activeTab, setActiveTab] = useState<TabId>("powerSupply");
  // Per-tab help modal (opened by the Help button; shows HELP[activeTab]).
  const [helpOpen, setHelpOpen] = useState(false);
  const [visibleTabs, setVisibleTabs] = useState<Record<TabId, boolean>>(
    loadTabVisibility,
  );
  const [tabOrder, setTabOrder] = useState<TabId[]>(loadTabOrder);

  // Measure the sticky header so the tab bar can freeze just below it on scroll.
  const headerRef = useRef<HTMLElement>(null);
  const [headerHeight, setHeaderHeight] = useState(0);
  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    const measure = () => setHeaderHeight(el.offsetHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Global performance mode. `perfMode` is the operator's preference for when
  // the HMI tab is visible; lightweight by default. Whenever the tab is hidden
  // we force lightweight (no point polling fast when nobody is looking), then
  // restore the preference when the tab is shown again.
  const [perfMode, setPerfMode] = useState<"performance" | "lightweight">(
    "lightweight",
  );
  useEffect(() => {
    const apply = () => {
      const effective = document.hidden ? "lightweight" : perfMode;
      api.setPerformanceMode(effective).catch(() => {});
    };
    apply();
    document.addEventListener("visibilitychange", apply);
    return () => document.removeEventListener("visibilitychange", apply);
  }, [perfMode]);
  const togglePerfMode = () =>
    setPerfMode((m) => (m === "performance" ? "lightweight" : "performance"));

  useEffect(() => {
    saveTabOrder(tabOrder);
  }, [tabOrder]);
  useEffect(() => {
    saveTabVisibility(visibleTabs);
  }, [visibleTabs]);

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
  const [inspectorView, setInspectorView] = useState<InspectorState>({ type: "none" });

  // Manual Override Form State inside inspector
  const [overrideVal, setOverrideVal] = useState<string>("0.0");
  const [showOverrideConfirm, setShowOverrideConfirm] = useState<boolean>(false);

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

  // Latest live values read by the (stable) tag selector below. Kept in refs so
  // `handleSelectTag` keeps a stable identity (for memoized consumers) while
  // still reading the current frame's values when actually invoked.
  const tagValuesRef = useRef(tagValues);
  tagValuesRef.current = tagValues;
  const tagsDictRef = useRef(tagsDict);
  tagsDictRef.current = tagsDict;

  const handleSelectTag = useCallback((name: string) => {
    const id = parseTagId(name);
    if (id !== null) {
      setInspectorView({ type: "tag", tagId: id });
      setOverrideVal(fmtNumber(tagValuesRef.current[id] ?? 0.0, 2, "0.00"));
      setShowOverrideConfirm(false);
      return;
    }
    setInspectorView({ type: "custom_tag", tagName: name });
    setOverrideVal(fmtNumber(tagsDictRef.current[name] ?? 0.0, 2, "0.00"));
    setShowOverrideConfirm(false);
  }, []);

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

  // Create a protected draft. Validation, review, approval, and activation are
  // intentionally separate operator actions in the workflow panel.
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

      const revision = await api.createConfigurationDraft(
        payload,
        "HMI protected configuration draft",
      );
      triggerNotification(
        `Draft ${revision.revision_id} created; review it in the protected workflow.`,
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
      triggerNotification(`Successfully forced Tag ${tagId} to ${parsed}`, "success");
      setShowOverrideConfirm(false);
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : "PLC write failed";
      triggerNotification(`Override failed: ${detail}`, "error");
    }
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
      setAlicats(await api.getAlicats());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchAlarmsAndEvents = useCallback(async () => {
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
  }, [setActiveAlarms]);

  const handleAcknowledgeAlarm = useCallback(
    async (tagId: string) => {
      try {
        await api.acknowledgeAlarm(tagId);
        triggerNotification(`Alarm on Tag ${tagId} acknowledged`, "success");
        fetchAlarmsAndEvents();
      } catch {
        triggerNotification("Failed to acknowledge alarm", "error");
      }
    },
    [triggerNotification, fetchAlarmsAndEvents],
  );

  // Stable ack-all handler for the memoized AlarmsHeader. `activeAlarms` is read
  // from a ref so this identity survives every telemetry frame.
  const activeAlarmsRef = useRef(activeAlarms);
  activeAlarmsRef.current = activeAlarms;
  const handleAcknowledgeAll = useCallback(async () => {
    for (const a of activeAlarmsRef.current) {
      if (!a.acknowledged) {
        await handleAcknowledgeAlarm(a.tag_id);
      }
    }
  }, [handleAcknowledgeAlarm]);

  const handleAlicatSetpoint = async (deviceId: string, val: number) => {
    try {
      await api.setAlicatSetpoint(deviceId, val);
      setAlicats((prev) =>
        prev.map((m) => (m.device_id === deviceId ? { ...m, setpoint: val } : m)),
      );
      triggerNotification(`Setpoint for MFC ${deviceId} set to ${val} SLPM.`, "success");
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
      triggerNotification(`Alicat MFC ${deviceId} gas calibration set to ${gas}.`, "success");
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
      triggerNotification(`Tuning session started for PID Loop ${index + 1}. Automatic control decoupled.`, "success");
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
        triggerNotification("Tuning session stopped. Process parameters identified.", "success");
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
    triggerNotification(`Applied recommended gains (Kp: ${kp}, Ki: ${ki}, Kd: ${kd}) to Loop ${index + 1}.`, "info");
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

  const handleTabVisibilityToggle = useCallback(
    (tab: TabId) => {
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
    },
    [visibleTabs, activeTab, triggerNotification],
  );

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
      <NotificationBanner notification={notification} />

      <header
        ref={headerRef}
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
          <CommsQualityBadge
            transportConnected={isConnected}
            health={commsHealth}
          />

          <button
            type="button"
            onClick={togglePerfMode}
            className="btn"
            style={{
              padding: "0.4rem 0.6rem",
              fontSize: "0.72rem",
              fontWeight: 700,
              whiteSpace: "nowrap",
              color:
                perfMode === "lightweight"
                  ? "var(--color-warning)"
                  : "var(--accent-cyan)",
            }}
            title={
              perfMode === "lightweight"
                ? "Lightweight mode — PLC polled slowly to conserve CPU / browser load. Click for Performance. (Auto-engages whenever this tab is hidden.)"
                : "Performance mode — fast PLC polling + live updates while this tab is visible. Click for Lightweight (saves CPU). Auto-drops to Lightweight when the tab is hidden."
            }
          >
            {perfMode === "lightweight" ? "◐ Lightweight" : "● Performance"}
          </button>

          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="btn"
            style={{ padding: "0.5rem" }}
            title="Toggle light/dark theme"
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
          </button>

          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="btn"
            style={{ padding: "0.5rem" }}
            title="Help for the current tab"
            aria-label="Open help for the current tab"
          >
            <BookOpen size={14} />
          </button>

          <button
            type="button"
            onClick={() => setInspectorView(inspectorView.type === "settings" ? { type: "none" } : { type: "settings" })}
            className="btn"
            style={{ padding: "0.5rem", color: inspectorView.type === "settings" ? "var(--accent-cyan)" : "inherit" }}
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

      <HelpModal
        tabId={activeTab}
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
      />

      <AlarmsHeader
        activeAlarms={activeAlarms}
        onAcknowledgeAll={handleAcknowledgeAll}
      />

      {/* Main content — the inspector is now a slide-in drawer (below), not a column */}
      <div className="main-layout-grid" style={{ gridTemplateColumns: "1fr" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Freeze the tab bar just below the sticky header so the active tab
              stays visible while scrolling the panel below. */}
          <div
            style={{
              position: "sticky",
              top: headerHeight,
              zIndex: 250,
              background: "var(--bg-color)",
              paddingBottom: "0.4rem",
            }}
          >
            <TabBar
              activeTab={activeTab}
              visibleTabs={visibleTabs}
              onSelect={setActiveTab}
              order={tabOrder}
              onReorder={setTabOrder}
              onHide={handleTabVisibilityToggle}
            />
          </div>

          <Suspense
            fallback={
              <div style={{ padding: "2rem", color: "var(--text-muted)" }}>
                Loading…
              </div>
            }
          >
          {activeTab === "powerSupply" && visibleTabs.powerSupply && (
            <PowerSupplyControl
              liveStatus={powerSupplyStatus}
              onOpenCapture={() => setInspectorView({ type: "export" })}
            />
          )}

          {activeTab === "temperature" && visibleTabs.temperature && (
            <TemperatureControl liveStatus={temperatureStatus} />
          )}

          {activeTab === "diagnostics" && visibleTabs.diagnostics && (
            <SignalDiagnostics history={history} historyTimes={historyTimes} />
          )}

          {/* Render Tab Contents */}
          {activeTab === "trends" && visibleTabs.trends && (
            <PanelStack
              regionId="trends"
              panels={[
                {
                  id: "trend",
                  // Drag to reorder / resize the live customizable graph.
                  node: <TrendChart history={history} tagValues={tagValues} />,
                },
                {
                  id: "monitors",
                  node: (
                    /* 32 Tag Broker Monitor Grid */
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
                        {TAG_INDICES.map((i) => {
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
                  ),
                },
              ]}
            />
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
            <div style={{ display: "flex", gap: "1rem", flexDirection: "column" }}>
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
                </div>
                <div style={{ marginTop: "0.75rem" }}>
                  <RoutingMatrix config={config} onUpdate={setConfig} tagValues={tagValues} />
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

              <div className="glass-panel">
                <ConfigurationWorkflowPanel />
              </div>
            </div>
          )}

          {activeTab === "events" && visibleTabs.events && (
            <div style={{ display: "grid", gap: "1rem" }}>
              <div className="glass-panel">
                <SystemHealthPanel />
              </div>
              <div className="glass-panel">
                <ProfessionalAlarmPanel />
              </div>
              <div className="glass-panel h-96">
                <EventLogView events={eventsHistory} />
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

          {activeTab === "ladder" && visibleTabs.ladder && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
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

          {activeTab === "explorer" && visibleTabs.explorer && (
            <DataExplorer triggerNotification={triggerNotification} />
          )}
          </Suspense>
        </div>

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
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="panel-header" style={{ borderBottom: "1px solid var(--panel-border)", paddingBottom: "0.5rem" }}>
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
                style={{ background: "none", border: "none", color: "var(--text-secondary)", cursor: "pointer" }}
                aria-label="Close panel"
              >
                <X size={16} />
              </button>
            </div>

            {inspectorView.type === "export" && <DataCapturePanel />}

            {/* Legacy default content (kept for tag/pid detail flows; the
                drawer only opens on an explicit selection or the Export button) */}
            {inspectorView.type === "none" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                {/* CSV Log Exporter inside Default Sidebar view */}
                <CsvExporter triggerNotification={triggerNotification} />

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

            {/* Tag register editor (Unified for both legacy tag index and custom named tags) */}
            {(inspectorView.type === "tag" || inspectorView.type === "custom_tag") && (
              <TagInspector
                view={inspectorView}
                allTags={allTags}
                tagsDict={tagsDict}
                tagValues={tagValues}
                config={config}
                overrideVal={overrideVal}
                showOverrideConfirm={showOverrideConfirm}
                deploying={deploying}
                onOverrideValChange={setOverrideVal}
                onShowOverrideConfirmChange={setShowOverrideConfirm}
                onExecuteOverride={executeOverride}
                onInterlockChange={handleInterlockChange}
                onDeploy={handleDeploy}
              />
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
                      {TAG_INDICES.map((i) => (
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
                      {TAG_INDICES.map((i) => (
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
                  {deploying ? "Creating Draft..." : "Create Protected PID Draft"}
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
                  {deploying ? "Creating Draft..." : "Create Protected Matrix Draft"}
                </button>
              </div>
            )}

            {/* Settings inspector view */}
            {inspectorView.type === "settings" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-primary)", fontWeight: 700, textTransform: "uppercase" }}>
                  Visible tabs
                </span>
                {
                  // ⚡ Bolt Optimization: Use the centralized TABS array to avoid allocating a new array of tuples on every render
                  TABS.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    aria-pressed={visibleTabs[tab.id]}
                    className={`tab-toggle ${visibleTabs[tab.id] ? "on" : ""}`}
                    onClick={() => handleTabVisibilityToggle(tab.id)}
                  >
                    <span>{tab.settingsLabel}</span>
                    <span className="tab-toggle-switch" />
                  </button>
                ))}
              </div>
            )}

            {inspectorView.type === "alicat" && (
              <AlicatInspector
                deviceId={inspectorView.deviceId}
                alicats={alicats}
                setpointValue={alicatSetpointVal}
                onSetpointValueChange={setAlicatSetpointVal}
                onSetpoint={handleAlicatSetpoint}
                onGasChange={handleAlicatGas}
                triggerNotification={triggerNotification}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};

export default App;
