/**
 * Multi-model pendulum app with Double, Triple, and Golfer modes.
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { PRESETS, Preset } from "./presets";
import { PRESETS_TRIPLE, PresetTriple } from "./presets_triple";
import { PRESETS_GOLFER, PresetGolfer } from "./presets_golfer";
import {
  makePendulumParams,
  makePolynomialTorque,
  runSimulation,
} from "./physics";
import {
  makeTripleParams,
  makePolynomialTorque3,
  runSimulation3,
} from "./physics_triple";
import {
  makeGolferParams,
  makePolynomialTorque_golfer,
  runSimulation_golfer,
} from "./physics_golfer";
import type { SimulationResult, JointLimits, TorqueClamp } from "./physics";
import type { SimulationResult3 } from "./physics_triple";
import type { SimulationResult_golfer, StateGolfer } from "./physics_golfer";
import type { UnitPreferences } from "./units";
import {
  DEFAULT_UNITS,
  FORCE_UNITS,
  TORQUE_UNITS,
  SPEED_UNITS,
  ANGULAR_SPEED_UNITS,
  ENERGY_UNITS,
} from "./units";
import type {
  TorqueUnit,
  ForceUnit,
  SpeedUnit,
  AngularSpeedUnit,
  EnergyUnit,
} from "./units";
import { PendulumCanvas } from "./components/PendulumCanvas";
import { TriplePendulumCanvas } from "./components/TriplePendulumCanvas";
import { GolferCanvas } from "./components/GolferCanvas";
import { ForceSourceLab } from "./components/ForceSourceLab";
import { UnitSelector } from "./components/UnitSelector";
import { CompanionGuide } from "./components/CompanionGuide";
import {
  isDoubleSimulationResult,
  isGolferSimulationResult,
  isTripleSimulationResult,
} from "./modelGuards";
import "./App.css";

const RotatingBaseStudy = React.lazy(() =>
  import("./components/RotatingBaseStudy").then((module) => ({
    default: module.RotatingBaseStudy,
  })),
);

// ── Animation hook ────────────────────────────────────────────────────────────

function useAnimationLoop(
  result: SimulationResult | SimulationResult3 | SimulationResult_golfer | null,
  playing: boolean,
  speed: number,
) {
  const [frameIdx, setFrameIdx] = useState(0);
  const rafRef = useRef<number | null>(null);
  const lastRef = useRef<number>(0);

  useEffect(() => {
    if (!result) return;
    setFrameIdx(0);
  }, [result]);

  useEffect(() => {
    if (!result || !playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    const dt = result.t.length > 1 ? result.t[1] - result.t[0] : 0.005;

    const tick = (ts: number) => {
      const elapsed = ts - lastRef.current;
      if (elapsed > 16) {
        lastRef.current = ts;
        const stepsPerFrame = Math.max(1, Math.round(speed / (1 / dt) / 60));
        setFrameIdx((prev) => {
          const next = prev + stepsPerFrame;
          return next >= result.states.length ? 0 : next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [result, playing, speed]);

  return frameIdx;
}

// ── Slider component ─────────────────────────────────────────────────────────

const Slider: React.FC<{
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
}> = ({ id, label, value, min, max, step, onChange, unit }) => (
  <div className="param-row" id={`param-${id}`}>
    <label className="param-label">{label}</label>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="param-slider"
    />
    <span className="param-value">
      {value.toFixed(step < 0.1 ? 3 : step < 1 ? 2 : 1)}
      {unit ? ` ${unit}` : ""}
    </span>
  </div>
);

type ModelType = "double" | "triple" | "golfer";

// ── Main App ──────────────────────────────────────────────────────────────────

function buildStateFromDeg(
  theta1Deg: number,
  phiDeg: number,
  dtheta1: number,
  dphi: number,
) {
  const DEG = Math.PI / 180;
  return [theta1Deg * DEG, phiDeg * DEG, dtheta1, dphi] as [
    number,
    number,
    number,
    number,
  ];
}

function buildStateFromDegTriple(
  theta1Deg: number,
  phi2Deg: number,
  phi3Deg: number,
  dtheta1: number,
  dphi2: number,
  dphi3: number,
) {
  const DEG = Math.PI / 180;
  return [
    theta1Deg * DEG,
    phi2Deg * DEG,
    phi3Deg * DEG,
    dtheta1,
    dphi2,
    dphi3,
  ] as [number, number, number, number, number, number];
}

function buildStateFromDegGolfer(
  theta_hub_deg: number,
  alpha_rs_deg: number,
  alpha_re_deg: number,
  alpha_rh_deg: number,
  alpha_ls_deg: number,
  alpha_le_deg: number,
  alpha_lh_deg: number,
  theta_club_deg: number,
  dtheta_hub: number,
  dalpha_rs: number,
  dalpha_re: number,
  dalpha_rh: number,
  dalpha_ls: number,
  dalpha_le: number,
  dalpha_lh: number,
  dtheta_club: number,
) {
  const DEG = Math.PI / 180;
  return [
    theta_hub_deg * DEG,
    alpha_rs_deg * DEG,
    alpha_re_deg * DEG,
    alpha_rh_deg * DEG,
    alpha_ls_deg * DEG,
    alpha_le_deg * DEG,
    alpha_lh_deg * DEG,
    theta_club_deg * DEG,
    dtheta_hub,
    dalpha_rs,
    dalpha_re,
    dalpha_rh,
    dalpha_ls,
    dalpha_le,
    dalpha_lh,
    dtheta_club,
  ] as StateGolfer;
}

export default function App() {
  // ── Model selection ──────────────────────────────────────────────────
  const [modelType, setModelType] = useState<ModelType>("double");

  // ── DOUBLE PENDULUM STATE ────────────────────────────────────────────
  const [selectedPreset, setSelectedPreset] = useState<Preset>(PRESETS[0]);
  const [m1, setM1] = useState(PRESETS[0].params.m1);
  const [m2, setM2] = useState(PRESETS[0].params.m2);
  const [mClub, setMClub] = useState(PRESETS[0].params.mClub);
  const [L1, setL1] = useState(PRESETS[0].params.L1);
  const [L2, setL2] = useState(PRESETS[0].params.L2);
  const [b1, setB1] = useState(PRESETS[0].params.b1);
  const [b2, setB2] = useState(PRESETS[0].params.b2);
  const [mu1, setMu1] = useState(PRESETS[0].params.mu1);
  const [mu2, setMu2] = useState(PRESETS[0].params.mu2);
  const [theta1Deg, setTheta1Deg] = useState(PRESETS[0].theta1Deg);
  const [phiDeg, setPhiDeg] = useState(PRESETS[0].phiDeg);
  const [coeffsShoulder, setCoeffsShoulder] = useState(
    PRESETS[0].coeffsShoulder,
  );
  const [coeffsWrist, setCoeffsWrist] = useState(PRESETS[0].coeffsWrist);
  const [shoulderStr, setShoulderStr] = useState(
    PRESETS[0].coeffsShoulder.join(", "),
  );
  const [wristStr, setWristStr] = useState(PRESETS[0].coeffsWrist.join(", "));

  // ── TRIPLE PENDULUM STATE ────────────────────────────────────────────
  const [selectedPresetTriple, setSelectedPresetTriple] =
    useState<PresetTriple>(PRESETS_TRIPLE[0]);
  const [m1_t, setM1_t] = useState(PRESETS_TRIPLE[0].params.m1);
  const [m2_t, setM2_t] = useState(PRESETS_TRIPLE[0].params.m2);
  const [m3_t, setM3_t] = useState(PRESETS_TRIPLE[0].params.m3);
  const [mClub_t, setMClub_t] = useState(PRESETS_TRIPLE[0].params.mClub);
  const [L1_t, setL1_t] = useState(PRESETS_TRIPLE[0].params.L1);
  const [L2_t, setL2_t] = useState(PRESETS_TRIPLE[0].params.L2);
  const [L3_t, setL3_t] = useState(PRESETS_TRIPLE[0].params.L3);
  const [b1_t, setB1_t] = useState(PRESETS_TRIPLE[0].params.b1);
  const [b2_t, setB2_t] = useState(PRESETS_TRIPLE[0].params.b2);
  const [b3_t, setB3_t] = useState(PRESETS_TRIPLE[0].params.b3);
  const [theta1Deg_t, setTheta1Deg_t] = useState(PRESETS_TRIPLE[0].theta1Deg);
  const [phi2Deg_t, setPhi2Deg_t] = useState(PRESETS_TRIPLE[0].phi2Deg);
  const [phi3Deg_t, setPhi3Deg_t] = useState(PRESETS_TRIPLE[0].phi3Deg);
  const [coeffsShoulderT, setCoeffsShoulderT] = useState(
    PRESETS_TRIPLE[0].coeffsShoulder,
  );
  const [coeffsElbowT, setCoeffsElbowT] = useState(
    PRESETS_TRIPLE[0].coeffsElbow,
  );
  const [coeffsWristT, setCoeffsWristT] = useState(
    PRESETS_TRIPLE[0].coeffsWrist,
  );
  const [shoulderStrT, setShoulderStrT] = useState(
    PRESETS_TRIPLE[0].coeffsShoulder.join(", "),
  );
  const [elbowStrT, setElbowStrT] = useState(
    PRESETS_TRIPLE[0].coeffsElbow.join(", "),
  );
  const [wristStrT, setWristStrT] = useState(
    PRESETS_TRIPLE[0].coeffsWrist.join(", "),
  );

  // ── GOLFER STATE ─────────────────────────────────────────────────────
  const [selectedPresetGolfer, setSelectedPresetGolfer] =
    useState<PresetGolfer>(PRESETS_GOLFER[0]);
  const [m_hub, setM_hub] = useState(PRESETS_GOLFER[0].params.m_hub);
  const [m_r_upper, setM_r_upper] = useState(
    PRESETS_GOLFER[0].params.m_r_upper,
  );
  const [m_r_fore, setM_r_fore] = useState(PRESETS_GOLFER[0].params.m_r_fore);
  const [m_l_upper, setM_l_upper] = useState(
    PRESETS_GOLFER[0].params.m_l_upper,
  );
  const [m_l_fore, setM_l_fore] = useState(PRESETS_GOLFER[0].params.m_l_fore);
  const [m_club, setM_club] = useState(PRESETS_GOLFER[0].params.m_club);
  const [theta_hub_deg_g, setTheta_hub_deg_g] = useState(
    PRESETS_GOLFER[0].theta_hub_deg,
  );
  const [alpha_rs_deg_g, setAlpha_rs_deg_g] = useState(
    PRESETS_GOLFER[0].alpha_rs_deg,
  );

  // ── UI State ─────────────────────────────────────────────────────────
  const [speed, setSpeed] = useState(1.0);
  const [result, setResult] = useState<
    SimulationResult | SimulationResult3 | SimulationResult_golfer | null
  >(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("Ready — select a preset and model");
  const [tEnd, setTEnd] = useState(PRESETS[0].tEnd);
  const [tEnd_t, setTEnd_t] = useState(PRESETS_TRIPLE[0].tEnd);
  const [tEnd_g, setTEnd_g] = useState(PRESETS_GOLFER[0].tEnd);

  // ── Joint limits & torque clamping ───────────────────────────────────
  const [enableLimits] = useState(false);
  const [phiMinDeg] = useState(-90);
  const [phiMaxDeg] = useState(90);
  const [limitStiffness] = useState(500);
  const [enableClamp] = useState(false);
  const [maxTorque1] = useState(50);
  const [maxTorque2] = useState(20);

  // ── Units ────────────────────────────────────────────────────────────
  const [units, setUnits] = useState<UnitPreferences>(DEFAULT_UNITS);

  const frameIdx = useAnimationLoop(result, playing, speed);

  // ── Unit updater ─────────────────────────────────────────────────────
  const updateUnit = useCallback(
    <K extends keyof UnitPreferences>(key: K, val: UnitPreferences[K]) => {
      setUnits((prev) => ({ ...prev, [key]: val }));
    },
    [],
  );

  // ── Preset loading ───────────────────────────────────────────────────
  const loadPreset = useCallback((p: Preset) => {
    setSelectedPreset(p);
    setM1(p.params.m1);
    setM2(p.params.m2);
    setMClub(p.params.mClub);
    setL1(p.params.L1);
    setL2(p.params.L2);
    setB1(p.params.b1);
    setB2(p.params.b2);
    setMu1(p.params.mu1);
    setMu2(p.params.mu2);
    setTheta1Deg(p.theta1Deg);
    setPhiDeg(p.phiDeg);
    setTEnd(p.tEnd);
    setCoeffsShoulder(p.coeffsShoulder);
    setCoeffsWrist(p.coeffsWrist);
    setShoulderStr(p.coeffsShoulder.join(", "));
    setWristStr(p.coeffsWrist.join(", "));
    setResult(null);
    setPlaying(false);
    setError(null);
  }, []);

  const loadPresetTriple = useCallback((p: PresetTriple) => {
    setSelectedPresetTriple(p);
    setM1_t(p.params.m1);
    setM2_t(p.params.m2);
    setM3_t(p.params.m3);
    setMClub_t(p.params.mClub);
    setL1_t(p.params.L1);
    setL2_t(p.params.L2);
    setL3_t(p.params.L3);
    setB1_t(p.params.b1);
    setB2_t(p.params.b2);
    setB3_t(p.params.b3);
    setTheta1Deg_t(p.theta1Deg);
    setPhi2Deg_t(p.phi2Deg);
    setPhi3Deg_t(p.phi3Deg);
    setTEnd_t(p.tEnd);
    setCoeffsShoulderT(p.coeffsShoulder);
    setCoeffsElbowT(p.coeffsElbow);
    setCoeffsWristT(p.coeffsWrist);
    setShoulderStrT(p.coeffsShoulder.join(", "));
    setElbowStrT(p.coeffsElbow.join(", "));
    setWristStrT(p.coeffsWrist.join(", "));
    setResult(null);
    setPlaying(false);
    setError(null);
  }, []);

  const loadPresetGolfer = useCallback((p: PresetGolfer) => {
    setSelectedPresetGolfer(p);
    setM_hub(p.params.m_hub);
    setM_r_upper(p.params.m_r_upper);
    setM_r_fore(p.params.m_r_fore);
    setM_l_upper(p.params.m_l_upper);
    setM_l_fore(p.params.m_l_fore);
    setM_club(p.params.m_club);
    setTheta_hub_deg_g(p.theta_hub_deg);
    setAlpha_rs_deg_g(p.alpha_rs_deg);
    setTEnd_g(p.tEnd);
    setResult(null);
    setPlaying(false);
    setError(null);
  }, []);

  // ── Parse torque coefficients ────────────────────────────────────────
  const parseCoeffs = (str: string): number[] | null => {
    try {
      const parts = str.split(",").map((s) => parseFloat(s.trim()));
      if (parts.some(isNaN) || parts.length < 1) return null;
      return parts;
    } catch {
      return null;
    }
  };

  // ── Build limits ─────────────────────────────────────────────────────
  const buildLimits = (): JointLimits | undefined => {
    if (!enableLimits) return undefined;
    const DEG = Math.PI / 180;
    return {
      phiMin: phiMinDeg * DEG,
      phiMax: phiMaxDeg * DEG,
      stiffness: limitStiffness,
      damping: 20,
    };
  };

  const buildClamp = (): TorqueClamp | undefined => {
    if (!enableClamp) return undefined;
    return { maxTorque1, maxTorque2 };
  };

  // ── Run simulation (DOUBLE) ──────────────────────────────────────────
  const runSimDouble = useCallback(() => {
    setError(null);
    setPlaying(false);
    try {
      const cShoulder = parseCoeffs(shoulderStr) ?? coeffsShoulder;
      const cWrist = parseCoeffs(wristStr) ?? coeffsWrist;
      setCoeffsShoulder(cShoulder);
      setCoeffsWrist(cWrist);

      const params = makePendulumParams({
        m1,
        m2,
        mClub,
        L1,
        L2,
        g: 9.81,
        b1,
        b2,
        mu1,
        mu2,
      });
      const init = buildStateFromDeg(theta1Deg, phiDeg, 0, 0);
      const tf = makePolynomialTorque(cShoulder, cWrist);
      const limits = buildLimits();
      const clamp = buildClamp();
      const r = runSimulation(params, init, tEnd, tf, 0.005, limits, clamp);
      setResult(r);
      setStatus(
        `Done: ${r.t.length} steps, t = 0…${r.t[r.t.length - 1].toFixed(2)} s`,
      );
      setPlaying(true);
    } catch (e) {
      setError(String(e));
      setStatus("Simulation error");
    }
  }, [
    m1,
    m2,
    mClub,
    L1,
    L2,
    b1,
    b2,
    mu1,
    mu2,
    theta1Deg,
    phiDeg,
    tEnd,
    shoulderStr,
    wristStr,
    coeffsShoulder,
    coeffsWrist,
    enableLimits,
    phiMinDeg,
    phiMaxDeg,
    limitStiffness,
    enableClamp,
    maxTorque1,
    maxTorque2,
  ]);

  // ── Run simulation (TRIPLE) ──────────────────────────────────────────
  const runSimTriple = useCallback(() => {
    setError(null);
    setPlaying(false);
    try {
      const cShoulder = parseCoeffs(shoulderStrT) ?? coeffsShoulderT;
      const cElbow = parseCoeffs(elbowStrT) ?? coeffsElbowT;
      const cWrist = parseCoeffs(wristStrT) ?? coeffsWristT;

      const params = makeTripleParams({
        m1: m1_t,
        m2: m2_t,
        m3: m3_t,
        mClub: mClub_t,
        L1: L1_t,
        L2: L2_t,
        L3: L3_t,
        g: 9.81,
        b1: b1_t,
        b2: b2_t,
        b3: b3_t,
      });
      const init = buildStateFromDegTriple(
        theta1Deg_t,
        phi2Deg_t,
        phi3Deg_t,
        0,
        0,
        0,
      );
      const tf = makePolynomialTorque3(cShoulder, cElbow, cWrist);
      const r = runSimulation3(params, init, tEnd_t, tf, 0.005);
      setResult(r);
      setStatus(
        `Done: ${r.t.length} steps, t = 0…${r.t[r.t.length - 1].toFixed(2)} s`,
      );
      setPlaying(true);
    } catch (e) {
      setError(String(e));
      setStatus("Simulation error");
    }
  }, [
    m1_t,
    m2_t,
    m3_t,
    mClub_t,
    L1_t,
    L2_t,
    L3_t,
    b1_t,
    b2_t,
    b3_t,
    theta1Deg_t,
    phi2Deg_t,
    phi3Deg_t,
    tEnd_t,
    shoulderStrT,
    elbowStrT,
    wristStrT,
    coeffsShoulderT,
    coeffsElbowT,
    coeffsWristT,
  ]);

  // ── Run simulation (GOLFER) ──────────────────────────────────────────
  const runSimGolfer = useCallback(() => {
    setError(null);
    setPlaying(false);
    try {
      const params = makeGolferParams({
        m_hub,
        m_r_upper,
        m_r_fore,
        m_l_upper,
        m_l_fore,
        m_club,
        L_hub: 0.25,
        L_r_upper: 0.3,
        L_r_fore: 0.25,
        L_l_upper: 0.3,
        L_l_fore: 0.25,
        L_club: 1.1,
        d_rs: 0.15,
        d_ls: 0.15,
        grip_right: 0.1,
        grip_left: 0.1,
        m_clubhead: 0.2,
        g: 9.81,
        b_hub: 0.1,
        b_rs: 0.08,
        b_re: 0.06,
        b_rh: 0.04,
        b_ls: 0.08,
        b_le: 0.06,
        b_lh: 0.04,
      });
      const init = buildStateFromDegGolfer(
        theta_hub_deg_g,
        alpha_rs_deg_g,
        -30,
        0,
        -35,
        25,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      );
      const tf = makePolynomialTorque_golfer([0], [0], [0], [0], [0], [0], [0]);
      const r = runSimulation_golfer(params, init, tEnd_g, tf, 0.005);
      setResult(r);
      setStatus(
        `Done: ${r.t.length} steps, t = 0…${r.t[r.t.length - 1].toFixed(2)} s`,
      );
      setPlaying(true);
    } catch (e) {
      setError(String(e));
      setStatus("Simulation error");
    }
  }, [
    m_hub,
    m_r_upper,
    m_r_fore,
    m_l_upper,
    m_l_fore,
    m_club,
    theta_hub_deg_g,
    alpha_rs_deg_g,
    tEnd_g,
  ]);

  const currentDoubleParams = makePendulumParams({
    m1, m2, mClub, L1, L2, g: 9.81, b1, b2, mu1, mu2,
  });
  const currentDoubleInitialState = buildStateFromDeg(theta1Deg, phiDeg, 0, 0);

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <span className="app-title">Pendulum Models — Golf Swing Dynamics</span>
        <span className="app-status">{status}</span>
      </header>
      <CompanionGuide onSelectModel={setModelType} />
      <React.Suspense
        fallback={
          <section className="rotating-base-study" aria-busy="true">
            Loading Qualified Rotating-Base Evidence…
          </section>
        }
      >
        <RotatingBaseStudy />
      </React.Suspense>

      <div
        className="app-body"
        role="tabpanel"
        id="main-panel"
        aria-labelledby={`tab-${modelType}`}
      >
        {/* ── Model selector tabs ── */}
        <div
          style={{
            padding: "10px",
            backgroundColor: "#222",
            borderBottom: "2px solid #444",
          }}
        >
          <div
            role="tablist"
            aria-label="Model Types"
            style={{ display: "flex", gap: "10px", marginBottom: "10px" }}
          >
            {(["double", "triple", "golfer"] as ModelType[]).map((m) => (
              <button
                key={m}
                id={`tab-${m}`}
                role="tab"
                aria-selected={modelType === m}
                aria-controls="main-panel"
                onClick={() => {
                  setModelType(m);
                  setResult(null);
                  setPlaying(false);
                }}
                style={{
                  padding: "8px 16px",
                  backgroundColor: modelType === m ? "#0066cc" : "#333",
                  color: "#fff",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontWeight: modelType === m ? "bold" : "normal",
                }}
              >
                {m === "double"
                  ? "Double Pendulum (2-DOF)"
                  : m === "triple"
                    ? "Triple Pendulum (3-DOF)"
                    : "Golfer (8-DOF)"}
              </button>
            ))}
          </div>
        </div>

        {/* ── Left panel: controls (model-specific) ── */}
        <aside className="controls-panel">
          {modelType === "double" && (
            <>
              {/* DOUBLE PENDULUM CONTROLS */}
              <div className="panel-section">
                <h3 className="section-title">Preset</h3>
                <select
                  id="preset-select"
                  className="preset-select"
                  value={selectedPreset.name}
                  onChange={(e) => {
                    const p = PRESETS.find((pr) => pr.name === e.target.value);
                    if (p) loadPreset(p);
                  }}
                >
                  {PRESETS.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <p className="preset-desc">{selectedPreset.description}</p>
              </div>

              <div className="panel-section">
                <h3 className="section-title">Arms (Segment 1)</h3>
                <Slider
                  id="m1"
                  label="Mass"
                  value={m1}
                  min={1}
                  max={10}
                  step={0.1}
                  onChange={setM1}
                  unit="kg"
                />
                <Slider
                  id="L1"
                  label="Length"
                  value={L1}
                  min={0.3}
                  max={1.0}
                  step={0.05}
                  onChange={setL1}
                  unit="m"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Shaft (Segment 2)</h3>
                <Slider
                  id="m2"
                  label="Shaft mass"
                  value={m2}
                  min={0.05}
                  max={1.0}
                  step={0.05}
                  onChange={setM2}
                  unit="kg"
                />
                <Slider
                  id="L2"
                  label="Shaft len"
                  value={L2}
                  min={0.5}
                  max={1.5}
                  step={0.05}
                  onChange={setL2}
                  unit="m"
                />
                <Slider
                  id="mClub"
                  label="Clubhead"
                  value={mClub}
                  min={0}
                  max={0.5}
                  step={0.01}
                  onChange={setMClub}
                  unit="kg"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Friction & Damping</h3>
                <Slider
                  id="b1"
                  label="b₁ viscous"
                  value={b1}
                  min={0}
                  max={2}
                  step={0.01}
                  onChange={setB1}
                  unit="N·m·s"
                />
                <Slider
                  id="b2"
                  label="b₂ viscous"
                  value={b2}
                  min={0}
                  max={2}
                  step={0.01}
                  onChange={setB2}
                  unit="N·m·s"
                />
                <Slider
                  id="mu1"
                  label="μ₁ Coulomb"
                  value={mu1}
                  min={0}
                  max={0.5}
                  step={0.005}
                  onChange={setMu1}
                  unit="N·m"
                />
                <Slider
                  id="mu2"
                  label="μ₂ Coulomb"
                  value={mu2}
                  min={0}
                  max={0.5}
                  step={0.005}
                  onChange={setMu2}
                  unit="N·m"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Initial Conditions</h3>
                <Slider
                  id="th1"
                  label="θ₁₀ arms"
                  value={theta1Deg}
                  min={-180}
                  max={180}
                  step={1}
                  onChange={setTheta1Deg}
                  unit="deg"
                />
                <Slider
                  id="phi"
                  label="φ₀ wrist"
                  value={phiDeg}
                  min={-180}
                  max={180}
                  step={1}
                  onChange={setPhiDeg}
                  unit="deg"
                />
                <Slider
                  id="tend"
                  label="Duration"
                  value={tEnd}
                  min={0.5}
                  max={10}
                  step={0.5}
                  onChange={setTEnd}
                  unit="s"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Torque Polynomials</h3>
                <div className="coeff-row">
                  <label className="param-label">Shoulder</label>
                  <input
                    className="coeff-input"
                    value={shoulderStr}
                    onChange={(e) => setShoulderStr(e.target.value)}
                    placeholder="-25, 10"
                    spellCheck="false"
                    autoCorrect="off"
                    autoCapitalize="none"
                  />
                </div>
                <div className="coeff-row">
                  <label className="param-label">Wrist</label>
                  <input
                    className="coeff-input"
                    value={wristStr}
                    onChange={(e) => setWristStr(e.target.value)}
                    placeholder="0"
                    spellCheck="false"
                    autoCorrect="off"
                    autoCapitalize="none"
                  />
                </div>
              </div>

              <div className="panel-section">
                <button
                  id="btn-run"
                  className="btn btn-primary"
                  onClick={runSimDouble}
                >
                  Run Simulation
                </button>
                {result && (
                  <button
                    id="btn-play-pause"
                    className="btn btn-secondary"
                    onClick={() => setPlaying((p) => !p)}
                    aria-label={
                      playing ? "Pause simulation" : "Play simulation"
                    }
                  >
                    {playing ? "Pause" : "Play"}
                  </button>
                )}
              </div>
            </>
          )}

          {modelType === "triple" && (
            <>
              {/* TRIPLE PENDULUM CONTROLS */}
              <div className="panel-section">
                <h3 className="section-title">Preset</h3>
                <select
                  id="preset-select"
                  className="preset-select"
                  value={selectedPresetTriple.name}
                  onChange={(e) => {
                    const p = PRESETS_TRIPLE.find(
                      (pr) => pr.name === e.target.value,
                    );
                    if (p) loadPresetTriple(p);
                  }}
                >
                  {PRESETS_TRIPLE.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <p className="preset-desc">
                  {selectedPresetTriple.description}
                </p>
              </div>

              <div className="panel-section">
                <h3 className="section-title">Segment 1 (Shoulder–Elbow)</h3>
                <Slider
                  id="m1_t"
                  label="Mass"
                  value={m1_t}
                  min={0.5}
                  max={5}
                  step={0.1}
                  onChange={setM1_t}
                  unit="kg"
                />
                <Slider
                  id="L1_t"
                  label="Length"
                  value={L1_t}
                  min={0.1}
                  max={0.5}
                  step={0.05}
                  onChange={setL1_t}
                  unit="m"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Segment 2 (Elbow–Wrist)</h3>
                <Slider
                  id="m2_t"
                  label="Mass"
                  value={m2_t}
                  min={0.5}
                  max={5}
                  step={0.1}
                  onChange={setM2_t}
                  unit="kg"
                />
                <Slider
                  id="L2_t"
                  label="Length"
                  value={L2_t}
                  min={0.1}
                  max={0.5}
                  step={0.05}
                  onChange={setL2_t}
                  unit="m"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Segment 3 (Wrist–Tip)</h3>
                <Slider
                  id="m3_t"
                  label="Mass"
                  value={m3_t}
                  min={0.1}
                  max={2}
                  step={0.1}
                  onChange={setM3_t}
                  unit="kg"
                />
                <Slider
                  id="L3_t"
                  label="Length"
                  value={L3_t}
                  min={0.05}
                  max={0.3}
                  step={0.05}
                  onChange={setL3_t}
                  unit="m"
                />
                <Slider
                  id="mClub_t"
                  label="Clubhead"
                  value={mClub_t}
                  min={0}
                  max={0.5}
                  step={0.01}
                  onChange={setMClub_t}
                  unit="kg"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Damping</h3>
                <Slider
                  id="b1_t"
                  label="b₁"
                  value={b1_t}
                  min={0}
                  max={0.2}
                  step={0.01}
                  onChange={setB1_t}
                  unit="N·m·s"
                />
                <Slider
                  id="b2_t"
                  label="b₂"
                  value={b2_t}
                  min={0}
                  max={0.2}
                  step={0.01}
                  onChange={setB2_t}
                  unit="N·m·s"
                />
                <Slider
                  id="b3_t"
                  label="b₃"
                  value={b3_t}
                  min={0}
                  max={0.2}
                  step={0.01}
                  onChange={setB3_t}
                  unit="N·m·s"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Initial Conditions</h3>
                <Slider
                  id="th1_t"
                  label="θ₁"
                  value={theta1Deg_t}
                  min={-180}
                  max={180}
                  step={1}
                  onChange={setTheta1Deg_t}
                  unit="deg"
                />
                <Slider
                  id="ph2_t"
                  label="φ₂"
                  value={phi2Deg_t}
                  min={-180}
                  max={180}
                  step={1}
                  onChange={setPhi2Deg_t}
                  unit="deg"
                />
                <Slider
                  id="ph3_t"
                  label="φ₃"
                  value={phi3Deg_t}
                  min={-180}
                  max={180}
                  step={1}
                  onChange={setPhi3Deg_t}
                  unit="deg"
                />
                <Slider
                  id="tend_t"
                  label="Duration"
                  value={tEnd_t}
                  min={0.5}
                  max={10}
                  step={0.5}
                  onChange={setTEnd_t}
                  unit="s"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Torques</h3>
                <div className="coeff-row">
                  <label className="param-label">Shoulder</label>
                  <input
                    className="coeff-input"
                    value={shoulderStrT}
                    onChange={(e) => setShoulderStrT(e.target.value)}
                    spellCheck="false"
                    autoCorrect="off"
                    autoCapitalize="none"
                  />
                </div>
                <div className="coeff-row">
                  <label className="param-label">Elbow</label>
                  <input
                    className="coeff-input"
                    value={elbowStrT}
                    onChange={(e) => setElbowStrT(e.target.value)}
                    spellCheck="false"
                    autoCorrect="off"
                    autoCapitalize="none"
                  />
                </div>
                <div className="coeff-row">
                  <label className="param-label">Wrist</label>
                  <input
                    className="coeff-input"
                    value={wristStrT}
                    onChange={(e) => setWristStrT(e.target.value)}
                    spellCheck="false"
                    autoCorrect="off"
                    autoCapitalize="none"
                  />
                </div>
              </div>

              <div className="panel-section">
                <button
                  id="btn-run"
                  className="btn btn-primary"
                  onClick={runSimTriple}
                >
                  Run Simulation
                </button>
                {result && (
                  <button
                    id="btn-play-pause"
                    className="btn btn-secondary"
                    onClick={() => setPlaying((p) => !p)}
                    aria-label={
                      playing ? "Pause simulation" : "Play simulation"
                    }
                  >
                    {playing ? "Pause" : "Play"}
                  </button>
                )}
              </div>
            </>
          )}

          {modelType === "golfer" && (
            <>
              {/* GOLFER CONTROLS */}
              <div className="panel-section">
                <h3 className="section-title">Preset</h3>
                <select
                  id="preset-select-golfer"
                  className="preset-select"
                  value={selectedPresetGolfer.name}
                  onChange={(e) => {
                    const p = PRESETS_GOLFER.find(
                      (pr) => pr.name === e.target.value,
                    );
                    if (p) loadPresetGolfer(p);
                  }}
                >
                  {PRESETS_GOLFER.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <p className="preset-desc">
                  {selectedPresetGolfer.description}
                </p>
              </div>

              <div className="panel-section">
                <h3 className="section-title">Body & Arms</h3>
                <Slider
                  id="m_hub"
                  label="Hub mass"
                  value={m_hub}
                  min={1}
                  max={5}
                  step={0.1}
                  onChange={setM_hub}
                  unit="kg"
                />
                <Slider
                  id="m_r_upper"
                  label="R upper"
                  value={m_r_upper}
                  min={1}
                  max={5}
                  step={0.1}
                  onChange={setM_r_upper}
                  unit="kg"
                />
                <Slider
                  id="m_r_fore"
                  label="R forearm"
                  value={m_r_fore}
                  min={0.5}
                  max={3}
                  step={0.1}
                  onChange={setM_r_fore}
                  unit="kg"
                />
                <Slider
                  id="m_l_upper"
                  label="L upper"
                  value={m_l_upper}
                  min={1}
                  max={5}
                  step={0.1}
                  onChange={setM_l_upper}
                  unit="kg"
                />
                <Slider
                  id="m_l_fore"
                  label="L forearm"
                  value={m_l_fore}
                  min={0.5}
                  max={3}
                  step={0.1}
                  onChange={setM_l_fore}
                  unit="kg"
                />
                <Slider
                  id="m_club"
                  label="Club"
                  value={m_club}
                  min={0.1}
                  max={1}
                  step={0.05}
                  onChange={setM_club}
                  unit="kg"
                />
              </div>

              <div className="panel-section">
                <h3 className="section-title">Initial Posture</h3>
                <Slider
                  id="th_hub_g"
                  label="Hub angle"
                  value={theta_hub_deg_g}
                  min={-90}
                  max={90}
                  step={5}
                  onChange={setTheta_hub_deg_g}
                  unit="deg"
                />
                <Slider
                  id="alpha_rs_g"
                  label="R shoulder"
                  value={alpha_rs_deg_g}
                  min={-90}
                  max={90}
                  step={5}
                  onChange={setAlpha_rs_deg_g}
                  unit="deg"
                />
                <Slider
                  id="tend_g"
                  label="Duration"
                  value={tEnd_g}
                  min={0.5}
                  max={10}
                  step={0.5}
                  onChange={setTEnd_g}
                  unit="s"
                />
              </div>

              <div className="panel-section">
                <p style={{ fontSize: "12px", color: "#aaa" }}>
                  Full 8-DOF golfer model with constrained club grip (4
                  holonomic constraints).
                </p>
              </div>

              <div className="panel-section">
                <button
                  id="btn-run-golfer"
                  className="btn btn-primary"
                  onClick={runSimGolfer}
                >
                  Run Simulation
                </button>
                {result && (
                  <button
                    id="btn-play-pause-golfer"
                    className="btn btn-secondary"
                    onClick={() => setPlaying((p) => !p)}
                    aria-label={
                      playing ? "Pause simulation" : "Play simulation"
                    }
                  >
                    {playing ? "Pause" : "Play"}
                  </button>
                )}
              </div>
            </>
          )}

          {error && <div className="error-box">{error}</div>}
        </aside>

        {/* ── Center: canvas ── */}
        <main className="center-panel">
          <div className="canvas-wrapper">
            {modelType === "double" && isDoubleSimulationResult(result) && (
              <PendulumCanvas
                states={result.states}
                params={result.params}
                currentIdx={frameIdx}
                width={420}
                height={460}
              />
            )}
            {modelType === "triple" && isTripleSimulationResult(result) && (
              <TriplePendulumCanvas
                states={result.states}
                params={result.params}
                currentIdx={frameIdx}
                width={420}
                height={460}
              />
            )}
            {modelType === "golfer" && isGolferSimulationResult(result) && (
              <GolferCanvas
                states={result.states}
                params={result.params}
                currentIdx={frameIdx}
                width={500}
                height={500}
              />
            )}
            {!result && (
              <div className="no-result">
                Run a simulation to see animation.
              </div>
            )}
            {result && (
              <div className="speed-row">
                <label className="param-label">Speed</label>
                <input
                  type="range"
                  min={0.25}
                  max={4}
                  step={0.25}
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="param-slider"
                />
                <span className="param-value">{speed}x</span>
              </div>
            )}
          </div>
        </main>

        {/* ── Right panel: units ── */}
        <aside className="matrix-panel">
          <div className="panel-section compact-section">
            <h3 className="section-title">Units</h3>
            <UnitSelector
              label="Force"
              value={units.force}
              options={FORCE_UNITS}
              onChange={(v) => updateUnit("force", v as ForceUnit)}
            />
            <UnitSelector
              label="Torque"
              value={units.torque}
              options={TORQUE_UNITS}
              onChange={(v) => updateUnit("torque", v as TorqueUnit)}
            />
            <UnitSelector
              label="Speed"
              value={units.speed}
              options={SPEED_UNITS}
              onChange={(v) => updateUnit("speed", v as SpeedUnit)}
            />
            <UnitSelector
              label="Ang. Speed"
              value={units.angularSpeed}
              options={ANGULAR_SPEED_UNITS}
              onChange={(v) =>
                updateUnit("angularSpeed", v as AngularSpeedUnit)
              }
            />
            <UnitSelector
              label="Energy"
              value={units.energy}
              options={ENERGY_UNITS}
              onChange={(v) => updateUnit("energy", v as EnergyUnit)}
            />
          </div>
        </aside>
      </div>
      {modelType === 'double' && (
        <ForceSourceLab
          params={currentDoubleParams}
          initialState={currentDoubleInitialState}
          onUsePose={(armAngleDeg, wristCockDeg) => {
            setTheta1Deg(armAngleDeg);
            setPhiDeg(wristCockDeg);
          }}
        />
      )}
    </div>
  );
}
