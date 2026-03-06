import React, { useState, useEffect, useRef, useCallback } from 'react';
import { PRESETS, Preset } from './presets';
import {
  makePendulumParams, makePolynomialTorque, runSimulation,
  massMatrixComponents, PendulumParams,
  jointVelocities, baseForce, computeAccelerations, controlVector,
  kineticEnergy, potentialEnergy,
} from './physics';
import type { SimulationResult, JointLimits, TorqueClamp } from './physics';
import type { UnitPreferences } from './units';
import {
  DEFAULT_UNITS,
  FORCE_UNITS, TORQUE_UNITS, SPEED_UNITS, ANGULAR_SPEED_UNITS, ENERGY_UNITS,
  speedFromSI, forceFromSI, energyFromSI,
  angularSpeedFromSI,
} from './units';
import type { TorqueUnit, ForceUnit, SpeedUnit, AngularSpeedUnit, EnergyUnit } from './units';
import { PendulumCanvas } from './components/PendulumCanvas';
import { AnalysisPlots, PLOT_IDS } from './components/AnalysisPlots';
import { OptimizerPanel } from './components/OptimizerPanel';
import { UnitSelector } from './components/UnitSelector';
import './App.css';

// ── Animation hook ────────────────────────────────────────────────────────────

function useAnimationLoop(
  result: SimulationResult | null,
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
        setFrameIdx(prev => {
          const next = prev + stepsPerFrame;
          return next >= result.states.length ? 0 : next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [result, playing, speed]);

  return frameIdx;
}

// ── Slider component (DRY) ────────────────────────────────────────────────────

const Slider: React.FC<{
  id: string; label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; unit?: string;
}> = ({ id, label, value, min, max, step, onChange, unit }) => (
  <div className="param-row" id={`param-${id}`}>
    <label className="param-label">{label}</label>
    <input
      type="range"
      min={min} max={max} step={step}
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="param-slider"
    />
    <span className="param-value">
      {value.toFixed(step < 0.1 ? 3 : step < 1 ? 2 : 1)}
      {unit ? ` ${unit}` : ''}
    </span>
  </div>
);

// ── Tab types ─────────────────────────────────────────────────────────────────

type MainTab = 'animation' | 'analysis' | 'plots' | 'optimizer';

// ── Main App ──────────────────────────────────────────────────────────────────

function buildStateFromDeg(theta1Deg: number, phiDeg: number,
  dtheta1: number, dphi: number) {
  const DEG = Math.PI / 180;
  return [theta1Deg * DEG, phiDeg * DEG, dtheta1, dphi] as [number, number, number, number];
}

export default function App() {
  // ── Preset state ────────────────────────────────────────────────────
  const [selectedPreset, setSelectedPreset] = useState<Preset>(PRESETS[0]);

  // ── Physical params ─────────────────────────────────────────────────
  const [m1, setM1] = useState(PRESETS[0].params.m1);
  const [m2, setM2] = useState(PRESETS[0].params.m2);
  const [mClub, setMClub] = useState(PRESETS[0].params.mClub);
  const [L1, setL1] = useState(PRESETS[0].params.L1);
  const [L2, setL2] = useState(PRESETS[0].params.L2);
  const [b1, setB1] = useState(PRESETS[0].params.b1);
  const [b2, setB2] = useState(PRESETS[0].params.b2);
  const [mu1, setMu1] = useState(PRESETS[0].params.mu1);
  const [mu2, setMu2] = useState(PRESETS[0].params.mu2);

  // ── Initial conditions ──────────────────────────────────────────────
  const [theta1Deg, setTheta1Deg] = useState(PRESETS[0].theta1Deg);
  const [phiDeg, setPhiDeg] = useState(PRESETS[0].phiDeg);
  const [tEnd, setTEnd] = useState(PRESETS[0].tEnd);

  // ── Torque coefficients ─────────────────────────────────────────────
  const [coeffsShoulder, setCoeffsShoulder] = useState(PRESETS[0].coeffsShoulder);
  const [coeffsWrist, setCoeffsWrist] = useState(PRESETS[0].coeffsWrist);
  const [shoulderStr, setShoulderStr] = useState(PRESETS[0].coeffsShoulder.join(', '));
  const [wristStr, setWristStr] = useState(PRESETS[0].coeffsWrist.join(', '));

  // ── Joint limits ────────────────────────────────────────────────────
  const [enableLimits, setEnableLimits] = useState(false);
  const [phiMinDeg, setPhiMinDeg] = useState(-90);
  const [phiMaxDeg, setPhiMaxDeg] = useState(90);
  const [limitStiffness, setLimitStiffness] = useState(500);

  // ── Torque clamping ─────────────────────────────────────────────────
  const [enableClamp, setEnableClamp] = useState(false);
  const [maxTorque1, setMaxTorque1] = useState(50);
  const [maxTorque2, setMaxTorque2] = useState(20);

  // ── Unit preferences ────────────────────────────────────────────────
  const [units, setUnits] = useState<UnitPreferences>(DEFAULT_UNITS);

  // ── Simulation/playback state ───────────────────────────────────────
  const [speed, setSpeed] = useState(1.0);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState<MainTab>('animation');
  const [detailPlot, setDetailPlot] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('Ready — select a preset and run the simulation');

  const frameIdx = useAnimationLoop(result, playing, speed);

  // ── Unit updater (DRY) ──────────────────────────────────────────────
  const updateUnit = useCallback(<K extends keyof UnitPreferences>(key: K, val: UnitPreferences[K]) => {
    setUnits(prev => ({ ...prev, [key]: val }));
  }, []);

  // ── Preset loading ──────────────────────────────────────────────────
  const loadPreset = useCallback((p: Preset) => {
    setSelectedPreset(p);
    setM1(p.params.m1); setM2(p.params.m2); setMClub(p.params.mClub);
    setL1(p.params.L1); setL2(p.params.L2);
    setB1(p.params.b1); setB2(p.params.b2);
    setMu1(p.params.mu1); setMu2(p.params.mu2);
    setTheta1Deg(p.theta1Deg); setPhiDeg(p.phiDeg);
    setTEnd(p.tEnd);
    setCoeffsShoulder(p.coeffsShoulder);
    setCoeffsWrist(p.coeffsWrist);
    setShoulderStr(p.coeffsShoulder.join(', '));
    setWristStr(p.coeffsWrist.join(', '));
    setResult(null); setPlaying(false); setError(null);
  }, []);

  // ── Parse torque coefficients ───────────────────────────────────────
  const parseCoeffs = (str: string): number[] | null => {
    try {
      const parts = str.split(',').map(s => parseFloat(s.trim()));
      if (parts.some(isNaN) || parts.length < 1) return null;
      return parts;
    } catch { return null; }
  };

  // ── Build sim config ────────────────────────────────────────────────
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

  // ── Run simulation ──────────────────────────────────────────────────
  const runSim = useCallback(() => {
    setError(null); setPlaying(false);
    try {
      const cShoulder = parseCoeffs(shoulderStr) ?? coeffsShoulder;
      const cWrist = parseCoeffs(wristStr) ?? coeffsWrist;
      setCoeffsShoulder(cShoulder);
      setCoeffsWrist(cWrist);

      const params = makePendulumParams({
        m1, m2, mClub, L1, L2, g: 9.81, b1, b2, mu1, mu2,
      });
      const init = buildStateFromDeg(theta1Deg, phiDeg, 0, 0);
      const tf = makePolynomialTorque(cShoulder, cWrist);
      const limits = buildLimits();
      const clamp = buildClamp();
      const r = runSimulation(params, init, tEnd, tf, 0.005, limits, clamp);
      setResult(r);
      setStatus(`Done: ${r.t.length} steps, t = 0…${r.t[r.t.length - 1].toFixed(2)} s`);
      setPlaying(true);
    } catch (e) {
      setError(String(e));
      setStatus('Simulation error');
    }
  }, [m1, m2, mClub, L1, L2, b1, b2, mu1, mu2, theta1Deg, phiDeg, tEnd,
      shoulderStr, wristStr, coeffsShoulder, coeffsWrist,
      enableLimits, phiMinDeg, phiMaxDeg, limitStiffness,
      enableClamp, maxTorque1, maxTorque2]);

  // ── Current frame data ──────────────────────────────────────────────
  const mmc = result && frameIdx < result.states.length
    ? massMatrixComponents(result.states[frameIdx][1], result.params)
    : null;

  const currentFrameData = result && frameIdx < result.states.length
    ? (() => {
        const state = result.states[frameIdx];
        const t = result.t[frameIdx];
        const jv = jointVelocities(state, result.params);
        const qdd = computeAccelerations(state, t, result.params, result.torqueFunc, result.limits, result.clamp);
        const bf = baseForce(state, qdd, result.params);
        const cv = controlVector(state, qdd, result.params, result.limits);
        const ke = kineticEnergy(state, result.params);
        const pe = potentialEnergy(state, result.params);
        return { state, t, jv, bf, cv, ke, pe };
      })()
    : null;

  // ── Optimizer callback ──────────────────────────────────────────────
  const handleOptimized = useCallback((cShoulder: number[], cWrist: number[]) => {
    setCoeffsShoulder(cShoulder);
    setCoeffsWrist(cWrist);
    setShoulderStr(cShoulder.map(c => c.toFixed(3)).join(', '));
    setWristStr(cWrist.map(c => c.toFixed(3)).join(', '));
  }, []);

  const currentParams: PendulumParams = {
    m1, m2, mClub, L1, L2, g: 9.81, b1, b2, mu1, mu2,
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <span className="app-title">Double Pendulum — Golf Swing Dynamics</span>
        <span className="app-status">{status}</span>
      </header>

      <div className="app-body">
        {/* ── Left panel: controls ── */}
        <aside className="controls-panel">
          {/* Preset */}
          <div className="panel-section">
            <h3 className="section-title">Preset</h3>
            <select
              id="preset-select"
              className="preset-select"
              value={selectedPreset.name}
              onChange={e => {
                const p = PRESETS.find(pr => pr.name === e.target.value);
                if (p) loadPreset(p);
              }}
            >
              {PRESETS.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
            </select>
            <p className="preset-desc">{selectedPreset.description}</p>
          </div>

          {/* Arms (segment 1) */}
          <div className="panel-section">
            <h3 className="section-title">Arms (Segment 1)</h3>
            <Slider id="m1" label="Mass" value={m1} min={1} max={10} step={0.1} onChange={setM1} unit="kg" />
            <Slider id="L1" label="Length" value={L1} min={0.3} max={1.0} step={0.05} onChange={setL1} unit="m" />
          </div>

          {/* Shaft + Clubhead (segment 2) */}
          <div className="panel-section">
            <h3 className="section-title">Shaft (Segment 2)</h3>
            <Slider id="m2" label="Shaft mass" value={m2} min={0.05} max={1.0} step={0.05} onChange={setM2} unit="kg" />
            <Slider id="L2" label="Shaft len" value={L2} min={0.5} max={1.5} step={0.05} onChange={setL2} unit="m" />
            <Slider id="mClub" label="Clubhead" value={mClub} min={0} max={0.5} step={0.01} onChange={setMClub} unit="kg" />
          </div>

          {/* Friction & Damping */}
          <div className="panel-section">
            <h3 className="section-title">Friction & Damping</h3>
            <Slider id="b1" label="b₁ viscous" value={b1} min={0} max={2} step={0.01} onChange={setB1} unit="N·m·s" />
            <Slider id="b2" label="b₂ viscous" value={b2} min={0} max={2} step={0.01} onChange={setB2} unit="N·m·s" />
            <Slider id="mu1" label="μ₁ Coulomb" value={mu1} min={0} max={0.5} step={0.005} onChange={setMu1} unit="N·m" />
            <Slider id="mu2" label="μ₂ Coulomb" value={mu2} min={0} max={0.5} step={0.005} onChange={setMu2} unit="N·m" />
          </div>

          {/* Joint Limits */}
          <div className="panel-section">
            <h3 className="section-title">Joint Limits</h3>
            <div className="param-row">
              <label className="param-label">Enable</label>
              <input
                type="checkbox"
                checked={enableLimits}
                onChange={e => setEnableLimits(e.target.checked)}
              />
            </div>
            {enableLimits && (
              <>
                <Slider id="phiMin" label="φ min" value={phiMinDeg} min={-180} max={0} step={5} onChange={setPhiMinDeg} unit="deg" />
                <Slider id="phiMax" label="φ max" value={phiMaxDeg} min={0} max={180} step={5} onChange={setPhiMaxDeg} unit="deg" />
                <Slider id="limitK" label="Stiffness" value={limitStiffness} min={50} max={2000} step={50} onChange={setLimitStiffness} unit="N·m/rad" />
              </>
            )}
          </div>

          {/* Torque Clamping */}
          <div className="panel-section">
            <h3 className="section-title">Torque Saturation</h3>
            <div className="param-row">
              <label className="param-label">Enable</label>
              <input
                type="checkbox"
                checked={enableClamp}
                onChange={e => setEnableClamp(e.target.checked)}
              />
            </div>
            {enableClamp && (
              <>
                <Slider id="maxT1" label="Max τ₁" value={maxTorque1} min={5} max={200} step={5} onChange={setMaxTorque1} unit="N·m" />
                <Slider id="maxT2" label="Max τ₂" value={maxTorque2} min={1} max={100} step={1} onChange={setMaxTorque2} unit="N·m" />
              </>
            )}
          </div>

          {/* Initial Conditions & Torque */}
          <div className="panel-section">
            <h3 className="section-title">Initial Conditions</h3>
            <Slider id="th1" label="θ₁₀ arms" value={theta1Deg} min={-180} max={180} step={1} onChange={setTheta1Deg} unit="deg" />
            <Slider id="phi" label="φ₀ wrist" value={phiDeg} min={-180} max={180} step={1} onChange={setPhiDeg} unit="deg" />
            <Slider id="tend" label="Duration" value={tEnd} min={0.5} max={10} step={0.5} onChange={setTEnd} unit="s" />
          </div>

          {/* Torque polynomials */}
          <div className="panel-section">
            <h3 className="section-title">Torque Polynomials</h3>
            <div className="coeff-row">
              <label className="param-label">Shoulder</label>
              <input
                className="coeff-input"
                value={shoulderStr}
                onChange={e => setShoulderStr(e.target.value)}
                placeholder="-25, 10"
              />
            </div>
            <div className="coeff-row">
              <label className="param-label">Wrist</label>
              <input
                className="coeff-input"
                value={wristStr}
                onChange={e => setWristStr(e.target.value)}
                placeholder="0"
              />
            </div>
            <p className="preset-desc">Coefficients: c₀, c₁, c₂, … for τ(t) = c₀ + c₁t + c₂t² + …</p>
          </div>

          {/* Run button */}
          <div className="panel-section">
            <button id="btn-run" className="btn btn-primary" onClick={runSim}>Run Simulation</button>
            {result && (
              <button
                id="btn-play-pause"
                className="btn btn-secondary"
                onClick={() => setPlaying(p => !p)}
              >
                {playing ? 'Pause' : 'Play'}
              </button>
            )}
          </div>

          {error && <div className="error-box">{error}</div>}
        </aside>

        {/* ── Center: canvas/plots/optimizer ── */}
        <main className="center-panel">
          <div className="tab-bar">
            {(['animation', 'analysis', 'plots', 'optimizer'] as MainTab[]).map(t => (
              <button
                key={t}
                id={`tab-${t}`}
                className={`tab-btn ${tab === t ? 'active' : ''}`}
                onClick={() => { setTab(t); if (t !== 'plots') setDetailPlot(null); }}
              >
                {t === 'animation' ? 'Animation' :
                 t === 'analysis' ? 'Analysis' :
                 t === 'plots' ? 'Plots' : 'Optimizer'}
              </button>
            ))}
          </div>

          {/* Animation tab */}
          {tab === 'animation' && (
            <div className="canvas-wrapper">
              <PendulumCanvas
                states={result ? result.states : []}
                params={result ? result.params : currentParams}
                currentIdx={frameIdx}
                width={420}
                height={460}
              />
              {result && (
                <div className="speed-row">
                  <label className="param-label">Speed</label>
                  <input
                    type="range" min={0.25} max={4} step={0.25} value={speed}
                    onChange={e => setSpeed(parseFloat(e.target.value))}
                    className="param-slider"
                  />
                  <span className="param-value">{speed}x</span>
                </div>
              )}
            </div>
          )}

          {/* Analysis tab (all plots compact) */}
          {tab === 'analysis' && result && (
            <div className="analysis-scroll">
              <AnalysisPlots result={result} units={units} />
            </div>
          )}
          {tab === 'analysis' && !result && (
            <div className="no-result">Run a simulation first to see analysis plots.</div>
          )}

          {/* Plots tab (single plot detail) */}
          {tab === 'plots' && result && (
            <div className="plots-detail-wrapper">
              <div className="plot-selector-bar">
                {PLOT_IDS.map(p => (
                  <button
                    key={p.id}
                    className={`plot-select-btn ${detailPlot === p.id ? 'active' : ''}`}
                    onClick={() => setDetailPlot(p.id)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              {detailPlot ? (
                <div className="detail-plot-area">
                  <AnalysisPlots result={result} units={units} detailPlot={detailPlot} />
                </div>
              ) : (
                <div className="no-result">Select a plot above to view in detail.</div>
              )}
            </div>
          )}
          {tab === 'plots' && !result && (
            <div className="no-result">Run a simulation first to see detailed plots.</div>
          )}

          {/* Optimizer tab */}
          {tab === 'optimizer' && (
            <div className="optimizer-scroll">
              <OptimizerPanel
                params={currentParams}
                initialState={buildStateFromDeg(theta1Deg, phiDeg, 0, 0)}
                tEnd={tEnd}
                limits={buildLimits()}
                clamp={buildClamp()}
                units={units}
                currentCoeffs={{ shoulder: coeffsShoulder, wrist: coeffsWrist }}
                onOptimized={handleOptimized}
              />
            </div>
          )}
        </main>

        {/* ── Right panel: matrix + units + live data ── */}
        <aside className="matrix-panel">
          {/* Units section */}
          <div className="panel-section compact-section">
            <h3 className="section-title">Units</h3>
            <UnitSelector label="Force" value={units.force} options={FORCE_UNITS} onChange={v => updateUnit('force', v as ForceUnit)} />
            <UnitSelector label="Torque" value={units.torque} options={TORQUE_UNITS} onChange={v => updateUnit('torque', v as TorqueUnit)} />
            <UnitSelector label="Speed" value={units.speed} options={SPEED_UNITS} onChange={v => updateUnit('speed', v as SpeedUnit)} />
            <UnitSelector label="Ang. Speed" value={units.angularSpeed} options={ANGULAR_SPEED_UNITS} onChange={v => updateUnit('angularSpeed', v as AngularSpeedUnit)} />
            <UnitSelector label="Energy" value={units.energy} options={ENERGY_UNITS} onChange={v => updateUnit('energy', v as EnergyUnit)} />
          </div>

          {/* Mass Matrix */}
          <h3 className="section-title">Mass Matrix M(q)</h3>
          {mmc ? (
            <>
              <div className="matrix-grid">
                {[
                  ['M₁₁', mmc.M11, 'Arms self'],
                  ['M₁₂', mmc.M12, 'Coupling'],
                  ['M₂₁', mmc.M21, 'Coupling'],
                  ['M₂₂', mmc.M22, 'Shaft self'],
                ].map(([key, val, desc]) => (
                  <div key={key as string} className="matrix-cell">
                    <span className="matrix-key">{key}</span>
                    <span className="matrix-val">{(val as number).toFixed(4)}</span>
                    <span className="matrix-desc">{desc}</span>
                  </div>
                ))}
              </div>
              <div className="coupling-bar-wrapper">
                <div className="coupling-label">Coupling M₁₂/M₁₁</div>
                <div className="coupling-bar-bg">
                  <div
                    className="coupling-bar-fill"
                    style={{ width: `${Math.min(100, Math.abs(mmc.M12 / mmc.M11) * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
            </>
          ) : (
            <div className="no-result" style={{ fontSize: 12 }}>
              Run simulation to see live matrix values.
            </div>
          )}

          {/* Live frame data */}
          {currentFrameData && (
            <div className="live-data-section">
              <h3 className="section-title">Live Data</h3>
              <div className="live-data-grid">
                <LiveRow label="Time" value={currentFrameData.t.toFixed(3)} unit="s" />
                <LiveRow label="θ₁ arms" value={(currentFrameData.state[0] * 180 / Math.PI).toFixed(1)} unit="deg" />
                <LiveRow label="φ wrist" value={(currentFrameData.state[1] * 180 / Math.PI).toFixed(1)} unit="deg" />
                <LiveRow label="ω arms" value={angularSpeedFromSI(currentFrameData.state[2], units.angularSpeed).toFixed(1)} unit={units.angularSpeed} />
                <LiveRow label="ω wrist" value={angularSpeedFromSI(currentFrameData.state[3], units.angularSpeed).toFixed(1)} unit={units.angularSpeed} />
                <LiveRow label="Wrist spd" value={speedFromSI(currentFrameData.jv.wristSpeed, units.speed).toFixed(2)} unit={units.speed} />
                <LiveRow label="Tip spd" value={speedFromSI(currentFrameData.jv.tipSpeed, units.speed).toFixed(2)} unit={units.speed} />
                <LiveRow label="|F base|" value={forceFromSI(currentFrameData.bf.magnitude, units.force).toFixed(1)} unit={units.force} />
                <LiveRow label="|CV|" value={forceFromSI(currentFrameData.cv.magnitude, units.force).toFixed(1)} unit={units.force} />
                <LiveRow label="KE" value={energyFromSI(currentFrameData.ke, units.energy).toFixed(2)} unit={units.energy} />
                <LiveRow label="PE" value={energyFromSI(currentFrameData.pe, units.energy).toFixed(2)} unit={units.energy} />
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

/** Small live data row component (DRY). */
const LiveRow: React.FC<{ label: string; value: string; unit: string }> = ({ label, value, unit }) => (
  <div className="live-row">
    <span className="live-label">{label}</span>
    <span className="live-value">{value}</span>
    <span className="live-unit">{unit}</span>
  </div>
);
