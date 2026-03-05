import React, { useState, useEffect, useRef, useCallback } from 'react';
import { PRESETS, Preset } from './presets';
import {
  makePendulumParams, makePolynomialTorque, runSimulation,
  massMatrixComponents, SimulationResult,
} from './physics';
import { PendulumCanvas } from './components/PendulumCanvas';
import { AnalysisPlots } from './components/AnalysisPlots';
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
  const dtPerFrame = useRef<number>(1);

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
    dtPerFrame.current = dt;

    const tick = (ts: number) => {
      const elapsed = ts - lastRef.current;
      if (elapsed > 16) {  // ~60 fps cap
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
  onChange: (v: number) => void;
}> = ({ id, label, value, min, max, step, onChange }) => (
  <div className="param-row" id={`param-${id}`}>
    <label className="param-label">{label}</label>
    <input
      type="range"
      min={min} max={max} step={step}
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="param-slider"
    />
    <span className="param-value">{value.toFixed(step < 0.1 ? 3 : 2)}</span>
  </div>
);

// ── Main App ──────────────────────────────────────────────────────────────────

function buildStateFromDeg(theta1Deg: number, phiDeg: number,
  dtheta1: number, dphi: number) {
  const DEG = Math.PI / 180;
  return [theta1Deg * DEG, phiDeg * DEG, dtheta1, dphi] as [number, number, number, number];
}

export default function App() {
  const [selectedPreset, setSelectedPreset] = useState<Preset>(PRESETS[0]);
  const [m1, setM1] = useState(PRESETS[0].params.m1);
  const [m2, setM2] = useState(PRESETS[0].params.m2);
  const [L1, setL1] = useState(PRESETS[0].params.L1);
  const [L2, setL2] = useState(PRESETS[0].params.L2);
  const [b1, setB1] = useState(PRESETS[0].params.b1);
  const [b2, setB2] = useState(PRESETS[0].params.b2);
  const [mu1, setMu1] = useState(PRESETS[0].params.mu1);
  const [mu2, setMu2] = useState(PRESETS[0].params.mu2);
  const [theta1Deg, setTheta1Deg] = useState(PRESETS[0].theta1Deg);
  const [phiDeg, setPhiDeg] = useState(PRESETS[0].phiDeg);
  const [tEnd, setTEnd] = useState(PRESETS[0].tEnd);
  const [speed, setSpeed] = useState(1.0);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState<'animation' | 'analysis'>('animation');
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('Ready — select a preset and run the simulation');

  const frameIdx = useAnimationLoop(result, playing, speed);

  const loadPreset = useCallback((p: Preset) => {
    setSelectedPreset(p);
    setM1(p.params.m1); setM2(p.params.m2);
    setL1(p.params.L1); setL2(p.params.L2);
    setB1(p.params.b1); setB2(p.params.b2);
    setMu1(p.params.mu1); setMu2(p.params.mu2);
    setTheta1Deg(p.theta1Deg); setPhiDeg(p.phiDeg);
    setTEnd(p.tEnd);
    setResult(null); setPlaying(false); setError(null);
  }, []);

  const runSim = useCallback(() => {
    setError(null); setPlaying(false);
    try {
      const params = makePendulumParams({ m1, m2, L1, L2, g: 9.81, b1, b2, mu1, mu2 });
      const init = buildStateFromDeg(theta1Deg, phiDeg, 0, 0);
      const tf = makePolynomialTorque(
        selectedPreset.coeffsShoulder, selectedPreset.coeffsWrist
      );
      const r = runSimulation(params, init, tEnd, tf, 0.005);
      setResult(r);
      setStatus(`✅ ${r.t.length} steps, t = ${r.t[0].toFixed(2)}…${r.t[r.t.length - 1].toFixed(2)} s`);
      setPlaying(true);
    } catch (e) {
      setError(String(e));
      setStatus('❌ Simulation error');
    }
  }, [m1, m2, L1, L2, b1, b2, mu1, mu2, theta1Deg, phiDeg, tEnd, selectedPreset]);

  // Mass matrix display
  const mmc = result && frameIdx < result.states.length
    ? massMatrixComponents(result.states[frameIdx][1], result.params)
    : null;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <span className="app-title">⚙ Double Pendulum — Golf Swing Dynamics</span>
        <span className="app-status">{status}</span>
      </header>

      <div className="app-body">
        {/* ── Left panel: controls ── */}
        <aside className="controls-panel">
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

          <div className="panel-section">
            <h3 className="section-title">Physical Parameters</h3>
            <Slider id="m1" label="m₁ (kg)" value={m1} min={0.1} max={10} step={0.1} onChange={setM1} />
            <Slider id="m2" label="m₂ (kg)" value={m2} min={0.05} max={2} step={0.05} onChange={setM2} />
            <Slider id="L1" label="L₁ (m)" value={L1} min={0.1} max={2} step={0.05} onChange={setL1} />
            <Slider id="L2" label="L₂ (m)" value={L2} min={0.1} max={2} step={0.05} onChange={setL2} />
          </div>

          <div className="panel-section">
            <h3 className="section-title">Friction &amp; Damping</h3>
            <Slider id="b1" label="b₁ (N·m·s)" value={b1} min={0} max={2} step={0.01} onChange={setB1} />
            <Slider id="b2" label="b₂ (N·m·s)" value={b2} min={0} max={2} step={0.01} onChange={setB2} />
            <Slider id="mu1" label="μ₁ Coulomb" value={mu1} min={0} max={0.5} step={0.005} onChange={setMu1} />
            <Slider id="mu2" label="μ₂ Coulomb" value={mu2} min={0} max={0.5} step={0.005} onChange={setMu2} />
          </div>

          <div className="panel-section">
            <h3 className="section-title">Initial Conditions</h3>
            <Slider id="th1" label="θ₁₀ (°)" value={theta1Deg} min={-180} max={180} step={1} onChange={setTheta1Deg} />
            <Slider id="phi" label="φ₀ (°)" value={phiDeg} min={-180} max={180} step={1} onChange={setPhiDeg} />
            <Slider id="tend" label="Duration (s)" value={tEnd} min={0.5} max={10} step={0.5} onChange={setTEnd} />
          </div>

          <div className="panel-section">
            <button id="btn-run" className="btn btn-primary" onClick={runSim}>▶ Run Simulation</button>
            {result && (
              <button
                id="btn-play-pause"
                className="btn btn-secondary"
                onClick={() => setPlaying(p => !p)}
              >
                {playing ? '⏸ Pause' : '▶ Play'}
              </button>
            )}
          </div>

          {error && <div className="error-box">{error}</div>}
        </aside>

        {/* ── Center: canvas or plots ── */}
        <main className="center-panel">
          <div className="tab-bar">
            <button
              id="tab-animation"
              className={`tab-btn ${tab === 'animation' ? 'active' : ''}`}
              onClick={() => setTab('animation')}
            >🎬 Animation</button>
            <button
              id="tab-analysis"
              className={`tab-btn ${tab === 'analysis' ? 'active' : ''}`}
              onClick={() => setTab('analysis')}
            >📊 Analysis</button>
          </div>

          {tab === 'animation' && (
            <div className="canvas-wrapper">
              <PendulumCanvas
                states={result ? result.states : []}
                params={result ? result.params : selectedPreset.params}
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
                  <span className="param-value">{speed}×</span>
                </div>
              )}
            </div>
          )}

          {tab === 'analysis' && result && (
            <div className="analysis-scroll">
              <AnalysisPlots result={result} />
            </div>
          )}

          {tab === 'analysis' && !result && (
            <div className="no-result">Run a simulation first to see analysis plots.</div>
          )}
        </main>

        {/* ── Right: mass matrix ── */}
        <aside className="matrix-panel">
          <h3 className="section-title">Mass Matrix M(q)</h3>
          {mmc ? (
            <>
              <div className="matrix-grid">
                {[
                  ['M₁₁', mmc.M11, 'Self-inertia arm'],
                  ['M₁₂', mmc.M12, 'Cross-coupling'],
                  ['M₂₁', mmc.M21, 'Cross-coupling'],
                  ['M₂₂', mmc.M22, 'Self-inertia club'],
                ].map(([key, val, desc]) => (
                  <div key={key as string} className="matrix-cell">
                    <span className="matrix-key">{key}</span>
                    <span className="matrix-val">{(val as number).toFixed(4)}</span>
                    <span className="matrix-desc">{desc}</span>
                  </div>
                ))}
              </div>
              <div className="coupling-bar-wrapper">
                <div className="coupling-label">Coupling ratio M₁₂/M₁₁</div>
                <div className="coupling-bar-bg">
                  <div
                    className="coupling-bar-fill"
                    style={{ width: `${Math.min(100, Math.abs(mmc.M12 / mmc.M11) * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
              {result && frameIdx < result.states.length && (
                <div className="energy-readout">
                  <div>t = {result.t[frameIdx]?.toFixed(3)} s</div>
                  <div>θ₁ = {(result.states[frameIdx][0] * 180 / Math.PI).toFixed(1)}°</div>
                  <div>φ = {(result.states[frameIdx][1] * 180 / Math.PI).toFixed(1)}°</div>
                </div>
              )}
            </>
          ) : (
            <div className="no-result" style={{ fontSize: 12 }}>
              Run simulation to see live matrix values.
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
