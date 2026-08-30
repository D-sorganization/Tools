import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import type { PendulumParams, State } from '../physics';
import {
    interpolateSeries,
    pendulumThumbnailGeometry,
    wrappedDegrees,
} from '../forceSourceView';
import {
    artifactWithScenario,
    FORCE_SOURCE_OBJECTIVES,
    OBJECTIVE_LABELS,
    optimizeForceSource,
    parseForceSourceArtifact,
    type ForceSourceArtifact,
    type ForceSourceObjective,
    type ForceSourceScenario,
    type SearchThoroughness,
} from '../forceSourceStudy';

interface ForceSourceLabProps {
    params: PendulumParams;
    initialState: State;
    onUsePose: (armAngleDeg: number, wristCockDeg: number) => void;
}

const COLORS: Record<ForceSourceObjective, string> = {
    coriolis_impulse: '#31d6c3',
    coriolis_energy_transfer: '#50a7ff',
    centrifugal_impulse: '#ffbd4a',
    centrifugal_energy_transfer: '#ff7a68',
    clubhead_speed: '#b58cff',
};

const CHART_STYLE: React.CSSProperties = {
    background: '#151927',
    border: '1px solid #30384e',
    borderRadius: 10,
    padding: 10,
    minHeight: 245,
};

function interpolate(time: number[], values: number[], target: number): number | null {
    if (target < time[0] || target > time[time.length - 1]) return null;
    return interpolateSeries(time, values, target);
}

function PendulumThumbnail({ scenario, time, params }: {
    scenario: ForceSourceScenario;
    time: number;
    params: PendulumParams;
}) {
    const sampleTime = Math.min(time, scenario.impact_time_s);
    const arm = interpolateSeries(
        scenario.series.time_s,
        scenario.series.arm_angle_rad,
        sampleTime,
    );
    const wrist = interpolateSeries(
        scenario.series.time_s,
        scenario.series.wrist_cock_rad,
        sampleTime,
    );
    const { originX, originY, wristX, wristY, tipX, tipY } = pendulumThumbnailGeometry(arm, wrist, params);
    const finalIndex = scenario.series.time_s.length - 1;
    const impact = pendulumThumbnailGeometry(
        scenario.series.arm_angle_rad[finalIndex],
        scenario.series.wrist_cock_rad[finalIndex],
        params,
    );
    return (
        <svg viewBox="0 0 192 176" role="img" aria-label={`${OBJECTIVE_LABELS[scenario.objective]} swing at ${time.toFixed(3)} seconds`}>
            <line x1="14" y1={impact.tipY} x2="178" y2={impact.tipY} stroke="#5a647d" strokeDasharray="4 5" />
            <circle cx={impact.tipX} cy={impact.tipY} r="3.5" fill="#ffffff" stroke="#aab4ca" />
            <circle cx={originX} cy={originY} r="4" fill="#eaf0ff" />
            <line x1={originX} y1={originY} x2={wristX} y2={wristY} stroke="#56d6c8" strokeWidth="7" strokeLinecap="round" />
            <line x1={wristX} y1={wristY} x2={tipX} y2={tipY} stroke="#e8edf9" strokeWidth="4" strokeLinecap="round" />
            <circle cx={wristX} cy={wristY} r="5" fill="#ffbd4a" />
            <circle cx={tipX} cy={tipY} r="7" fill={COLORS[scenario.objective]} />
        </svg>
    );
}

function ComparisonPlot({ title, unit, scenarios, field }: {
    title: string;
    unit: string;
    scenarios: ForceSourceScenario[];
    field: 'clubhead_speed_m_s' | 'shoulder_torque_nm' | 'wrist_torque_nm';
}) {
    const data = useMemo(() => {
        const maxTime = Math.max(...scenarios.map(scenario => scenario.impact_time_s));
        return Array.from({ length: 181 }, (_, index) => {
            const time = maxTime * index / 180;
            const row: Record<string, number | null> = { time: +time.toFixed(4) };
            for (const scenario of scenarios) {
                row[scenario.objective] = interpolate(
                    scenario.series.time_s,
                    scenario.series[field],
                    time,
                );
            }
            return row;
        });
    }, [field, scenarios]);
    return (
        <div style={CHART_STYLE} data-testid={`force-source-plot-${field}`}>
            <h4>{title} <span className="force-source-unit">[{unit}]</span></h4>
            <ResponsiveContainer width="100%" height={205}>
                <LineChart data={data} margin={{ top: 6, right: 12, bottom: 2, left: 4 }}>
                    <CartesianGrid stroke="#30384e" strokeOpacity={0.65} />
                    <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} stroke="#8791aa" tick={{ fontSize: 10 }} label={{ value: 'Time [s]', position: 'insideBottomRight', offset: -2 }} />
                    <YAxis stroke="#8791aa" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: '#111522', border: '1px solid #3c455e' }} />
                    <Legend wrapperStyle={{ fontSize: 10 }} formatter={(value: string) => OBJECTIVE_LABELS[value as ForceSourceObjective]} />
                    {scenarios.map(scenario => (
                        <Line key={scenario.objective} type="monotone" dataKey={scenario.objective} stroke={COLORS[scenario.objective]} dot={false} strokeWidth={2} connectNulls={false} />
                    ))}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export const ForceSourceLab: React.FC<ForceSourceLabProps> = ({ params, initialState, onUsePose }) => {
    const [artifact, setArtifact] = useState<ForceSourceArtifact | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [selected, setSelected] = useState<Set<ForceSourceObjective>>(new Set(FORCE_SOURCE_OBJECTIVES));
    const [playing, setPlaying] = useState(false);
    const [playbackRate, setPlaybackRate] = useState(0.35);
    const [time, setTime] = useState(0);
    const lastTimestamp = useRef<number | null>(null);
    const [objective, setObjective] = useState<ForceSourceObjective>('clubhead_speed');
    const [thoroughness, setThoroughness] = useState<SearchThoroughness>('quick');
    const [wristLimit, setWristLimit] = useState(30);
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState({ completed: 0, total: 1, bestScore: Number.NEGATIVE_INFINITY });

    useEffect(() => {
        let active = true;
        fetch('/force-source-comparison.json')
            .then(response => {
                if (!response.ok) throw new Error(`Built-in study returned HTTP ${response.status}`);
                return response.json();
            })
            .then(value => {
                if (active) setArtifact(parseForceSourceArtifact(value));
            })
            .catch(error => {
                if (active) setLoadError(`Load a registered study JSON to compare scenarios. ${String(error)}`);
            });
        return () => { active = false; };
    }, []);

    const visibleScenarios = useMemo(
        () => artifact?.scenarios.filter(scenario => selected.has(scenario.objective)) ?? [],
        [artifact, selected],
    );
    const maxTime = visibleScenarios.length
        ? Math.max(...visibleScenarios.map(scenario => scenario.impact_time_s))
        : 0;

    useEffect(() => {
        if (!playing || maxTime <= 0) return;
        let frame = 0;
        const tick = (timestamp: number) => {
            if (lastTimestamp.current !== null) {
                const elapsed = Math.min((timestamp - lastTimestamp.current) / 1000, 0.05);
                setTime(previous => (previous + elapsed * playbackRate) % maxTime);
            }
            lastTimestamp.current = timestamp;
            frame = requestAnimationFrame(tick);
        };
        frame = requestAnimationFrame(tick);
        return () => {
            cancelAnimationFrame(frame);
            lastTimestamp.current = null;
        };
    }, [maxTime, playbackRate, playing]);

    const importArtifact = async (file: File) => {
        try {
            const parsed = parseForceSourceArtifact(JSON.parse(await file.text()));
            setArtifact(parsed);
            setLoadError(null);
            setTime(0);
            setSelected(new Set(parsed.scenarios.map(scenario => scenario.objective)));
        } catch (error) {
            setLoadError(`Study import failed: ${String(error)}`);
        }
    };

    const runOptimization = async () => {
        setRunning(true);
        setLoadError(null);
        try {
            const scenario = await optimizeForceSource(
                { params, initialState, objective, wristTorqueLimitNm: wristLimit, thoroughness },
                setProgress,
            );
            setArtifact(previous => artifactWithScenario(previous, scenario, initialState, params));
            setSelected(previous => new Set([...previous, objective]));
            setTime(0);
        } catch (error) {
            setLoadError(`Optimization failed: ${String(error)}`);
        } finally {
            setRunning(false);
        }
    };

    const toggleScenario = (objectiveName: ForceSourceObjective) => {
        setSelected(previous => {
            const next = new Set(previous);
            if (next.has(objectiveName)) next.delete(objectiveName);
            else next.add(objectiveName);
            return next;
        });
    };

    return (
        <section className="force-source-lab" aria-labelledby="force-source-heading">
            <div className="force-source-heading-row">
                <div>
                    <p className="force-source-kicker">Double-pendulum research workspace</p>
                    <h2 id="force-source-heading">Force-Source Optimization Lab</h2>
                    <p>Compare synchronized motion, clubhead speed, and shoulder/wrist loading across five coordinate-explicit objectives. Only a non-looping first pass with near-horizontal clubhead motion near the bottom of the arc qualifies as impact.</p>
                </div>
                <label className="btn btn-secondary force-source-import">
                    Import study JSON
                    <input type="file" accept="application/json,.json" onChange={event => {
                        const file = event.target.files?.[0];
                        if (file) void importArtifact(file);
                    }} />
                </label>
            </div>

            <div className="force-source-config">
                <div>
                    <label htmlFor="force-objective">Objective</label>
                    <select id="force-objective" value={objective} onChange={event => setObjective(event.target.value as ForceSourceObjective)} disabled={running}>
                        {FORCE_SOURCE_OBJECTIVES.map(value => <option key={value} value={value}>{OBJECTIVE_LABELS[value]}</option>)}
                    </select>
                </div>
                <div>
                    <label htmlFor="force-search-depth">Search</label>
                    <select id="force-search-depth" value={thoroughness} onChange={event => setThoroughness(event.target.value as SearchThoroughness)} disabled={running}>
                        <option value="quick">Quick — coarse + 2 refinements</option>
                        <option value="thorough">Thorough — 5 N m grid + 1 N m refinement</option>
                    </select>
                </div>
                <div className="force-source-limit">
                    <label htmlFor="force-wrist-limit">Wrist limit: {wristLimit} N m</label>
                    <input id="force-wrist-limit" type="range" min="5" max="30" step="1" value={wristLimit} onChange={event => setWristLimit(Number(event.target.value))} disabled={running} />
                </div>
                <div className="force-source-start">
                    <span>Next optimization start</span>
                    <strong>Arm {(initialState[0] * 180 / Math.PI).toFixed(1)}° absolute · Wrist {(initialState[1] * 180 / Math.PI).toFixed(1)}° relative</strong>
                    <span>Club {wrappedDegrees(initialState[0] + initialState[1]).toFixed(1)}° absolute</span>
                    <button
                        type="button"
                        className="force-source-text-button"
                        onClick={() => artifact && onUsePose(
                            artifact.initial_pose.arm_angle_rad * 180 / Math.PI,
                            artifact.initial_pose.wrist_cock_rad * 180 / Math.PI,
                        )}
                        disabled={running || !artifact}
                    >
                        Apply loaded comparison pose
                    </button>
                </div>
                <button className="btn btn-primary" onClick={() => void runOptimization()} disabled={running}>
                    {running ? 'Optimizing…' : 'Optimize from this pose'}
                </button>
            </div>

            {running && (
                <div className="force-source-progress" role="status">
                    <span style={{ width: `${Math.min(100, 100 * progress.completed / progress.total)}%` }} />
                    <p>{progress.completed}/{progress.total} candidates · best {Number.isFinite(progress.bestScore) ? progress.bestScore.toFixed(3) : '—'}</p>
                </div>
            )}
            {loadError && <div className="error-box">{loadError}</div>}

            {artifact && (
                <>
                    <div className="force-source-provenance">
                        <span>{artifact.model}</span>
                        <span>
                            Loaded start: arm {(artifact.initial_pose.arm_angle_rad * 180 / Math.PI).toFixed(1)}° absolute · wrist {(artifact.initial_pose.wrist_cock_rad * 180 / Math.PI).toFixed(1)}° relative · club {wrappedDegrees(artifact.initial_pose.arm_angle_rad + artifact.initial_pose.wrist_cock_rad).toFixed(1)}° absolute
                        </span>
                        <span>{artifact.evaluated_count} evaluated · {artifact.qualified_count} qualified</span>
                        {artifact.qualification_counts?.unqualified_loop_or_joint_excursion !== undefined && (
                            <span>{artifact.qualification_counts.unqualified_loop_or_joint_excursion} loop/joint rejects</span>
                        )}
                        {artifact.qualification_counts?.unqualified_impact_geometry !== undefined && (
                            <span>{artifact.qualification_counts.unqualified_impact_geometry} impact-path rejects</span>
                        )}
                        <span>Coordinates: shoulder absolute / wrist relative</span>
                    </div>
                    <div className="force-source-scenario-toggles" aria-label="Visible optimization scenarios">
                        {artifact.scenarios.map(scenario => (
                            <label key={scenario.objective} style={{ borderColor: COLORS[scenario.objective] }}>
                                <input type="checkbox" checked={selected.has(scenario.objective)} onChange={() => toggleScenario(scenario.objective)} />
                                {OBJECTIVE_LABELS[scenario.objective]}
                            </label>
                        ))}
                    </div>

                    <div className="force-source-playback">
                        <button className="btn btn-secondary" onClick={() => setPlaying(value => !value)}>{playing ? 'Pause' : 'Play'}</button>
                        <button className="btn btn-secondary" onClick={() => setTime(0)}>Restart</button>
                        <label className="force-source-rate">
                            Speed
                            <select aria-label="Playback speed" value={playbackRate} onChange={event => setPlaybackRate(Number(event.target.value))}>
                                {[0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2].map(rate => (
                                    <option key={rate} value={rate}>{rate.toFixed(2)}×</option>
                                ))}
                            </select>
                        </label>
                        <input aria-label="Comparison time" type="range" min="0" max={Math.max(maxTime, 0.001)} step="0.0005" value={Math.min(time, maxTime)} onChange={event => { setTime(Number(event.target.value)); setPlaying(false); }} />
                        <output>{time.toFixed(4)} s</output>
                    </div>

                    <div className="force-source-animation-grid">
                        {visibleScenarios.map(scenario => (
                            <article key={scenario.objective} style={{ borderColor: COLORS[scenario.objective] }}>
                                <h3>{OBJECTIVE_LABELS[scenario.objective]}</h3>
                                <PendulumThumbnail scenario={scenario} time={Math.min(time, scenario.impact_time_s)} params={params} />
                                <dl>
                                    <div><dt>Impact speed</dt><dd>{scenario.series.clubhead_speed_m_s[scenario.series.clubhead_speed_m_s.length - 1].toFixed(2)} m/s</dd></div>
                                    {scenario.impact_diagnostics && <div><dt>Impact path</dt><dd>{scenario.impact_diagnostics.path_angle_deg.toFixed(1)}° · {(100 * scenario.impact_diagnostics.bottom_reach_fraction).toFixed(0)}% reach</dd></div>}
                                    <div><dt>Wrist program</dt><dd>−{scenario.candidate.wrist_restrain_nm.toFixed(1)} / +{scenario.candidate.wrist_drive_nm.toFixed(1)} N m</dd></div>
                                    <div>
                                        <dt>Robust qualification</dt>
                                        <dd>
                                            {scenario.robustness.sample_count <= 1
                                                ? "Nominal only"
                                                : `${(100 * scenario.robustness.qualification_rate).toFixed(0)}%`}
                                        </dd>
                                    </div>
                                </dl>
                                {scenario.boundary_hits.length > 0 && <p className="force-source-boundary">Boundary: {scenario.boundary_hits.join(', ')}</p>}
                            </article>
                        ))}
                    </div>

                    {visibleScenarios.length > 0 && (
                        <div className="force-source-plot-grid">
                            <ComparisonPlot title="Clubhead speed" unit="m/s" scenarios={visibleScenarios} field="clubhead_speed_m_s" />
                            <ComparisonPlot title="Shoulder torque" unit="N m" scenarios={visibleScenarios} field="shoulder_torque_nm" />
                            <ComparisonPlot title="Wrist torque" unit="N m" scenarios={visibleScenarios} field="wrist_torque_nm" />
                        </div>
                    )}
                    <aside className="force-source-caveat">
                        <strong>Interpretation boundary.</strong> {artifact.interpretation_limits.join(' ')}
                    </aside>
                </>
            )}
        </section>
    );
};
