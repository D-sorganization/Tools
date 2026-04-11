/**
 * Optimizer panel UI — configures and runs torque profile optimization.
 *
 * DRY: Reuses shared Slider pattern, optimizer module.
 * DbC: Validates inputs before launching optimization.
 */
import React, { useState, useCallback, useRef } from 'react';
import type { PendulumParams, State, JointLimits, TorqueClamp } from '../physics';
import type { OptimizerProgress, OptimizerResult } from '../optimizer';
import { optimizeTorqueProfile, defaultOptimizerConfig } from '../optimizer';
import type { UnitPreferences } from '../units';
import { speedFromSI } from '../units';

interface OptimizerPanelProps {
    params: PendulumParams;
    initialState: State;
    tEnd: number;
    limits?: JointLimits;
    clamp?: TorqueClamp;
    units: UnitPreferences;
    currentCoeffs?: { shoulder: number[]; wrist: number[] };
    onOptimized: (coeffsShoulder: number[], coeffsWrist: number[]) => void;
}

export const OptimizerPanel: React.FC<OptimizerPanelProps> = ({
    params, initialState, tEnd, limits, clamp, units,
    currentCoeffs, onOptimized,
}) => {
    const [shoulderDegree, setShoulderDegree] = useState(2);
    const [wristDegree, setWristDegree] = useState(1);
    const [maxIterations, setMaxIterations] = useState(300);
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState<OptimizerProgress | null>(null);
    const [result, setResult] = useState<OptimizerResult | null>(null);
    const cancelRef = useRef(false);

    const runOptimizer = useCallback(() => {
        setRunning(true);
        setProgress(null);
        setResult(null);
        cancelRef.current = false;

        const config = defaultOptimizerConfig(params, initialState, tEnd, limits, clamp);
        config.shoulderDegree = shoulderDegree;
        config.wristDegree = wristDegree;
        config.maxIterations = maxIterations;

        // Run in a setTimeout to allow UI to update with progress
        setTimeout(() => {
            const r = optimizeTorqueProfile(
                config,
                (p) => {
                    setProgress({ ...p });
                },
                currentCoeffs,
            );
            setResult(r);
            setRunning(false);
        }, 50);
    }, [params, initialState, tEnd, limits, clamp, shoulderDegree, wristDegree, maxIterations, currentCoeffs]);

    const applyResult = useCallback(() => {
        if (result) {
            onOptimized(result.coeffsShoulder, result.coeffsWrist);
        }
    }, [result, onOptimized]);

    const progressPct = progress
        ? Math.min(100, (progress.iteration / progress.maxIterations) * 100)
        : 0;

    return (
        <div className="optimizer-panel">
            <h3 className="section-title">Torque Optimizer</h3>
            <p className="optimizer-desc">
                Maximize horizontal tip velocity at bottom of arc
                by optimizing polynomial torque coefficients.
            </p>

            <div className="optimizer-config">
                <div className="param-row">
                    <label className="param-label">Shoulder deg</label>
                    <select
                        className="unit-select"
                        value={shoulderDegree}
                        onChange={e => setShoulderDegree(parseInt(e.target.value))}
                        disabled={running}
                    >
                        {[0, 1, 2, 3, 4, 5].map(d => (
                            <option key={d} value={d}>{d}</option>
                        ))}
                    </select>
                </div>
                <div className="param-row">
                    <label className="param-label">Wrist deg</label>
                    <select
                        className="unit-select"
                        value={wristDegree}
                        onChange={e => setWristDegree(parseInt(e.target.value))}
                        disabled={running}
                    >
                        {[0, 1, 2, 3, 4, 5].map(d => (
                            <option key={d} value={d}>{d}</option>
                        ))}
                    </select>
                </div>
                <div className="param-row">
                    <label className="param-label">Max iters</label>
                    <select
                        className="unit-select"
                        value={maxIterations}
                        onChange={e => setMaxIterations(parseInt(e.target.value))}
                        disabled={running}
                    >
                        {[100, 200, 300, 500, 1000].map(n => (
                            <option key={n} value={n}>{n}</option>
                        ))}
                    </select>
                </div>
            </div>

            <button
                className={`btn ${running ? 'btn-secondary' : 'btn-primary'}`}
                onClick={runOptimizer}
                disabled={running}
            >
                {running ? 'Optimizing...' : 'Optimize'}
            </button>

            {/* Progress bar */}
            {(running || progress) && (
                <div className="optimizer-progress">
                    <div className="progress-bar-bg">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${progressPct}%` }}
                        />
                    </div>
                    {progress && (
                        <div className="progress-info">
                            <span>Iter {progress.iteration}/{progress.maxIterations}</span>
                            <span>
                                Best: {speedFromSI(progress.bestHorizontalSpeed, units.speed).toFixed(2)} {units.speed}
                            </span>
                        </div>
                    )}
                </div>
            )}

            {/* Result */}
            {result && !running && (
                <div className="optimizer-result">
                    <div className="result-header">
                        {result.converged ? 'Converged' : 'Stopped'} after {result.iterations} iters
                    </div>
                    <div className="result-speed">
                        Max horizontal tip speed: <strong>
                            {speedFromSI(result.bestHorizontalSpeed, units.speed).toFixed(2)} {units.speed}
                        </strong>
                    </div>
                    <div className="result-coeffs">
                        <div>Shoulder: [{result.coeffsShoulder.map(c => c.toFixed(3)).join(', ')}]</div>
                        <div>Wrist: [{result.coeffsWrist.map(c => c.toFixed(3)).join(', ')}]</div>
                    </div>
                    <button className="btn btn-primary" onClick={applyResult}>
                        Apply Optimized Coefficients
                    </button>
                </div>
            )}
        </div>
    );
};
