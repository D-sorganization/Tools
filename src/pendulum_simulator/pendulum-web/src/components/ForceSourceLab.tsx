import { useEffect, useMemo, useRef, useState } from 'react';

import { ForceSourceResults, OBJECTIVE_COLORS } from './ForceSourceResults';
import { wrappedDegrees, type AnimationAlignment } from '../forceSourceView';
import {
    artifactWithScenarios,
    buildOptimizationContract,
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    FORCE_SOURCE_OBJECTIVES,
    OBJECTIVE_LABELS,
    optimizeForceSourceComparison,
    parseForceSourceArtifact,
    type ForceSourceArtifact,
    type ForceSourceConstraints,
    type ForceSourceObjective,
    type ForceSourceStudyMode,
    type SearchThoroughness,
} from '../forceSourceStudy';
import type { PendulumParams, State } from '../physics';

interface ForceSourceLabProps {
    params: PendulumParams;
    initialState: State;
    onUsePose: (armAngleDeg: number, wristCockDeg: number) => void;
}

interface NumericFieldProps {
    id: string; label: string; value: number; title: string;
    onChange: (value: number) => void; min?: number; max?: number; step?: number; disabled?: boolean;
}

function NumericField({ id, label, value, title, onChange, ...inputProps }: NumericFieldProps) {
    return <label htmlFor={id} title={title} className="force-source-number">
        <span>{label}</span>
        <input id={id} type="number" value={value} onChange={event => onChange(Number(event.target.value))} title={title} {...inputProps} />
    </label>;
}

const degrees = (radians: number) => radians * 180 / Math.PI;
const radians = (value: number) => value * Math.PI / 180;
type BundledStudy = 'equal_speed' | 'equal_effort';

export function ForceSourceLab({ params, initialState, onUsePose }: ForceSourceLabProps) {
    const [artifact, setArtifact] = useState<ForceSourceArtifact | null>(null);
    const [bundledStudy, setBundledStudy] = useState<BundledStudy>('equal_speed');
    const [message, setMessage] = useState<string | null>(null);
    const [selected, setSelected] = useState(new Set<ForceSourceObjective>(FORCE_SOURCE_OBJECTIVES));
    const [objective, setObjective] = useState<ForceSourceObjective>('clubhead_speed');
    const [thoroughness, setThoroughness] = useState<SearchThoroughness>('thorough');
    const [constraints, setConstraints] = useState<ForceSourceConstraints>({ ...DEFAULT_OPTIMIZATION_CONSTRAINTS });
    const [startArm, setStartArm] = useState(degrees(initialState[0]));
    const [startWrist, setStartWrist] = useState(degrees(initialState[1]));
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState({ completed: 0, total: 1, bestScore: -Infinity, label: '' });
    const [playing, setPlaying] = useState(false);
    const [playbackRate, setPlaybackRate] = useState(0.35);
    const [time, setTime] = useState(0);
    const [alignment, setAlignment] = useState<AnimationAlignment>('fixed_hub');
    const lastTimestamp = useRef<number | null>(null);

    useEffect(() => {
        let active = true;
        const path = bundledStudy === 'equal_speed'
            ? '/force-source-comparison.json'
            : '/force-source-comparison-equal-effort.json';
        fetch(path).then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        }).then(value => {
            if (!active) return;
            const parsed = parseForceSourceArtifact(value);
            setArtifact(parsed);
            setConstraints(structuredClone(parsed.comparison_contract.constraints));
            setThoroughness(parsed.comparison_contract.thoroughness);
            setStartArm(+degrees(parsed.initial_pose.arm_angle_rad).toFixed(2));
            setStartWrist(+degrees(parsed.initial_pose.wrist_cock_rad).toFixed(2));
        })
            .catch(error => { if (active) setMessage(`Built-in comparison unavailable: ${String(error)}`); });
        return () => { active = false; };
    }, [bundledStudy]);

    const visible = useMemo(() => artifact?.scenarios.filter(item => selected.has(item.objective)) ?? [], [artifact, selected]);
    const maxTime = visible.length ? Math.max(...visible.map(item => item.impact_time_s)) : 0;

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
        return () => { cancelAnimationFrame(frame); lastTimestamp.current = null; };
    }, [maxTime, playbackRate, playing]);

    const setConstraint = <K extends keyof ForceSourceConstraints>(key: K, value: ForceSourceConstraints[K]) => {
        setConstraints(previous => ({ ...previous, [key]: value }));
    };

    const optimizationState = (): State => [radians(startArm), radians(startWrist), initialState[2], initialState[3]];

    const runObjectives = async (objectives: readonly ForceSourceObjective[]) => {
        setRunning(true); setMessage(null);
        try {
            if (objectives.length === 0) throw new RangeError('Select at least one objective');
            const initialState = optimizationState();
            const baseConfig = { params, initialState, thoroughness, constraints };
            const contract = buildOptimizationContract({ ...baseConfig, objective: objectives[0] });
            const changedContract = artifact !== null && artifact.comparison_contract.id !== contract.id;
            const scenarios = await optimizeForceSourceComparison(
                baseConfig,
                objectives,
                artifact?.scenarios ?? [],
                update => setProgress({
                    ...update,
                    label: OBJECTIVE_LABELS[update.objective ?? objectives[0]],
                }),
            );
            const current = artifactWithScenarios(
                artifact,
                scenarios,
                { ...baseConfig, objective: objectives[0] },
            );
            setArtifact(current);
            setSelected(new Set(current.scenarios.map(item => item.objective)));
            if (changedContract) {
                setMessage('Started a new comparison because the pose or search contract changed; stale scenarios were removed.');
            }
            setTime(0);
        } catch (error) {
            setMessage(`Optimization stopped: ${String(error)}`);
        } finally { setRunning(false); }
    };

    const importArtifact = async (file: File) => {
        try {
            const parsed = parseForceSourceArtifact(JSON.parse(await file.text()));
            setArtifact(parsed); setMessage(null); setTime(0);
            setConstraints(structuredClone(parsed.comparison_contract.constraints));
            setThoroughness(parsed.comparison_contract.thoroughness);
            setSelected(new Set(parsed.scenarios.map(item => item.objective)));
        } catch (error) { setMessage(`Study import failed: ${String(error)}`); }
    };

    const exportArtifact = () => {
        if (!artifact) return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(artifact, null, 2)], { type: 'application/json' }));
        const link = document.createElement('a');
        link.href = url; link.download = 'force-source-comparison.json'; link.click();
        URL.revokeObjectURL(url);
    };

    const toggleScenario = (name: ForceSourceObjective) => setSelected(previous => {
        const next = new Set(previous);
        if (next.has(name)) next.delete(name); else next.add(name);
        return next;
    });

    return <section className="force-source-lab" aria-labelledby="force-source-heading">
        <div className="force-source-heading-row">
            <div><p className="force-source-kicker">Double-pendulum research workspace</p>
                <h2 id="force-source-heading">Force-Source Optimization Lab</h2>
                <p>Compare six objectives using bounded continuous sixth-order torque profiles under explicit equal-speed, equal-effort, or common-bound contracts. Work and activation are accounted separately from the optimized force-source metric.</p></div>
            <label className="btn btn-secondary force-source-import" title="Open a registered comparison artifact">Import study JSON<input type="file" accept="application/json,.json" onChange={event => { const file = event.target.files?.[0]; if (file) void importArtifact(file); }} /></label>
        </div>

        <div className="force-source-config">
            <label title="Load a checked-in equal-output or equal-input research comparison"><span>Bundled comparison</span>
                <select id="force-bundled-study" value={bundledStudy} onChange={event => setBundledStudy(event.target.value as BundledStudy)} disabled={running}>
                    <option value="equal_speed">Equal speed</option>
                    <option value="equal_effort">Equal effort</option>
                </select></label>
            <label title="Quantity maximized by the selected optimization"><span>Objective</span>
                <select id="force-objective" value={objective} onChange={event => setObjective(event.target.value as ForceSourceObjective)} disabled={running}>
                    {FORCE_SOURCE_OBJECTIVES.map(value => <option key={value} value={value}>{OBJECTIVE_LABELS[value]}</option>)}
                </select></label>
            <label title="Quick explores broadly; thorough and research add local refinement"><span>Search depth</span>
                <select id="force-search-depth" value={thoroughness} onChange={event => setThoroughness(event.target.value as SearchThoroughness)} disabled={running}>
                    <option value="quick">Quick</option><option value="thorough">Thorough</option><option value="research">Research</option>
                </select></label>
            <label title="Equal speed enforces a target band and input caps; equal effort enforces only common input caps; common bounds applies only the torque, slew, and motion limits"><span>Comparison basis</span>
                <select id="force-study-mode" value={constraints.studyMode} onChange={event => setConstraint('studyMode', event.target.value as ForceSourceStudyMode)} disabled={running}>
                    <option value="equal_speed">Equal speed + input caps</option>
                    <option value="equal_effort">Equal effort caps</option>
                    <option value="common_bounds">Common torque bounds</option>
                </select></label>
            <NumericField id="force-start-arm" label="Start arm [deg]" value={startArm} step={0.1} onChange={setStartArm} disabled={running} title="Absolute arm angle; direct entry is not slider-limited" />
            <NumericField id="force-start-wrist" label="Start wrist [deg]" value={startWrist} step={0.1} onChange={setStartWrist} disabled={running} title="Wrist cock relative to the arm; more negative means more initial cock in this coordinate convention" />
            <NumericField id="force-wrist-limit" label="Wrist limit [N m]" value={constraints.wristTorqueLimitNm} min={0.1} max={30} step={0.5} onChange={value => setConstraint('wristTorqueLimitNm', value)} disabled={running} title="Absolute wrist restrain and drive torque limit; supported maximum is 30 N m" />
            <NumericField id="force-wrist-step" label="Wrist coefficient step [N m]" value={constraints.wristTorqueStepNm} min={0.05} max={30} step={0.05} onChange={value => setConstraint('wristTorqueStepNm', value)} disabled={running} title="Degree-6 wrist control-point granularity; smaller values make refinement finer" />
            <button className="btn btn-primary" onClick={() => void runObjectives([objective])} disabled={running}>Optimize selected</button>
            <button className="btn btn-secondary" onClick={() => void runObjectives(FORCE_SOURCE_OBJECTIVES)} disabled={running}>Optimize all 6</button>
        </div>

        <details className="force-source-advanced"><summary>Advanced constraints and robustness</summary>
            <div className="force-source-constraint-grid">
                <NumericField id="force-candidate-budget" label="Candidate budget" value={constraints.candidateBudget} min={8} max={10000} step={1} onChange={value => setConstraint('candidateBudget', value)} title="Number of deterministic low-discrepancy global candidates per objective" />
                <NumericField id="force-robustness" label="Robustness trials" value={constraints.robustnessTrials} min={1} max={101} step={2} onChange={value => setConstraint('robustnessTrials', value)} title="Held-out perturbations around the winning program" />
                <NumericField id="force-path-angle" label="Max impact path [deg]" value={constraints.maxImpactPathAngleDeg} min={1} max={30} step={0.5} onChange={value => setConstraint('maxImpactPathAngleDeg', value)} title="Maximum vertical deviation from horizontal at impact" />
                <NumericField id="force-bottom-reach" label="Minimum bottom reach" value={constraints.minBottomReachFraction} min={0.8} max={1} step={0.01} onChange={value => setConstraint('minBottomReachFraction', value)} title="Minimum downward clubhead reach as a fraction of total link length" />
                <NumericField id="force-shoulder-min" label="Shoulder min [N m]" value={constraints.shoulderTorqueNm.min} step={1} onChange={value => setConstraint('shoulderTorqueNm', { ...constraints.shoulderTorqueNm, min: value })} title="Lower shoulder-torque search bound" />
                <NumericField id="force-shoulder-max" label="Shoulder max [N m]" value={constraints.shoulderTorqueNm.max} step={1} onChange={value => setConstraint('shoulderTorqueNm', { ...constraints.shoulderTorqueNm, max: value })} title="Upper shoulder-torque search bound" />
                <NumericField id="force-shoulder-step" label="Shoulder step [N m]" value={constraints.shoulderTorqueNm.step} min={0.1} step={0.1} onChange={value => setConstraint('shoulderTorqueNm', { ...constraints.shoulderTorqueNm, step: value })} title="Shoulder-torque quantization" />
                <NumericField id="force-profile-min" label="Profile duration min [s]" value={constraints.profileDurationS.min} min={0.1} step={0.005} onChange={value => setConstraint('profileDurationS', { ...constraints.profileDurationS, min: value })} title="Shortest polynomial forcing window" />
                <NumericField id="force-profile-max" label="Profile duration max [s]" value={constraints.profileDurationS.max} min={0.1} step={0.005} onChange={value => setConstraint('profileDurationS', { ...constraints.profileDurationS, max: value })} title="Longest polynomial forcing window" />
                <NumericField id="force-profile-step" label="Duration step [s]" value={constraints.profileDurationS.step} min={0.001} step={0.001} onChange={value => setConstraint('profileDurationS', { ...constraints.profileDurationS, step: value })} title="Forcing-window timing granularity" />
                <NumericField id="force-slew-limit" label="Max torque slew [N m/s]" value={constraints.maxTorqueSlewNmS} min={1} step={25} onChange={value => setConstraint('maxTorqueSlewNmS', value)} title="Upper bound on the derivative of either continuous torque profile" />
                <NumericField id="force-transition-torque" label="Transition band [N m]" value={constraints.transitionTorqueNm} min={0.1} max={29.9} step={0.1} onChange={value => setConstraint('transitionTorqueNm', value)} title="Wrist torque magnitude treated as slack around direction reversal" />
                <NumericField id="force-transition-duration" label="Minimum transition [s]" value={constraints.minWristTransitionS} min={0.001} step={0.001} onChange={value => setConstraint('minWristTransitionS', value)} title="Required continuous time inside the low-torque wrist transition band" />
                <NumericField id="force-target-speed" label="Speed target [m/s]" value={constraints.targetClubheadSpeedMps} min={1} step={0.1} onChange={value => setConstraint('targetClubheadSpeedMps', value)} title="Center of the required impact-speed band in equal-speed mode; a marker in the other modes" />
                <NumericField id="force-speed-tolerance" label="Speed band width [m/s]" value={constraints.speedToleranceMps} min={0.05} step={0.05} onChange={value => setConstraint('speedToleranceMps', value)} title="Allowed speed range above the target in equal-speed mode" />
                <NumericField id="force-positive-work-budget" label="Positive work cap [J]" value={constraints.maxPositiveActuatorWorkJ} min={1} step={5} onChange={value => setConstraint('maxPositiveActuatorWorkJ', value)} title="Common cap on summed positive shoulder and wrist actuator work" />
                <NumericField id="force-squared-effort-budget" label="Squared effort cap [N² m² s]" value={constraints.maxSquaredTorqueEffortNm2S} min={1} step={50} onChange={value => setConstraint('maxSquaredTorqueEffortNm2S', value)} title="Common cap on the time integral of shoulder-torque squared plus wrist-torque squared" />
                <NumericField id="force-min-robustness" label="Minimum robust qualification" value={constraints.minimumRobustQualificationRate} min={0} max={1} step={0.05} onChange={value => setConstraint('minimumRobustQualificationRate', value)} title="Minimum held-out fraction that must retain the golf, speed, and effort contract" />
                <NumericField id="force-elite-count" label="Elite starts" value={constraints.eliteCandidateCount} min={1} max={64} step={1} onChange={value => setConstraint('eliteCandidateCount', value)} title="Number of globally sampled profiles retained for multi-start coefficient refinement" />
                <NumericField id="force-arm-min" label="Arm angle min [deg]" value={constraints.armAngleDeg.min} step={1} onChange={value => setConstraint('armAngleDeg', { ...constraints.armAngleDeg, min: value })} title="Lower allowable absolute arm angle" />
                <NumericField id="force-arm-max" label="Arm angle max [deg]" value={constraints.armAngleDeg.max} step={1} onChange={value => setConstraint('armAngleDeg', { ...constraints.armAngleDeg, max: value })} title="Upper allowable absolute arm angle" />
                <NumericField id="force-wrist-min" label="Wrist angle min [deg]" value={constraints.wristAngleDeg.min} step={1} onChange={value => setConstraint('wristAngleDeg', { ...constraints.wristAngleDeg, min: value })} title="Lower allowable relative wrist angle" />
                <NumericField id="force-wrist-max" label="Wrist angle max [deg]" value={constraints.wristAngleDeg.max} step={1} onChange={value => setConstraint('wristAngleDeg', { ...constraints.wristAngleDeg, max: value })} title="Upper allowable relative wrist angle" />
                <NumericField id="force-arm-travel" label="Max arm travel [deg]" value={constraints.maxArmTravelDeg} min={1} step={1} onChange={value => setConstraint('maxArmTravelDeg', value)} title="Anti-loop limit on total arm angular excursion before impact" />
                <NumericField id="force-club-travel" label="Max club travel [deg]" value={constraints.maxClubTravelDeg} min={1} step={1} onChange={value => setConstraint('maxClubTravelDeg', value)} title="Anti-loop limit on total absolute-club angular excursion before impact" />
                <NumericField id="force-duration" label="Duration [s]" value={constraints.simulationDurationS} min={0.2} step={0.05} onChange={value => setConstraint('simulationDurationS', value)} title="Maximum simulated time before a candidate is rejected" />
                <NumericField id="force-integration-step" label="Integration step [s]" value={constraints.integrationStepS} min={0.00025} step={0.00025} onChange={value => setConstraint('integrationStepS', value)} title="RK4 simulation resolution; smaller values are smoother and slower" />
                <NumericField id="force-pose-perturbation" label="Pose perturbation [deg]" value={constraints.posePerturbationDeg} min={0} step={0.25} onChange={value => setConstraint('posePerturbationDeg', value)} title="Held-out start-angle perturbation used for robustness" />
                <NumericField id="force-torque-perturbation" label="Torque perturbation [fraction]" value={constraints.torquePerturbationFraction} min={0} max={0.25} step={0.01} onChange={value => setConstraint('torquePerturbationFraction', value)} title="Held-out multiplicative torque perturbation used for robustness" />
            </div>
        </details>

        <div className="force-source-pose-actions">
            <button className="force-source-text-button" onClick={() => { setStartArm(degrees(initialState[0])); setStartWrist(degrees(initialState[1])); }}>Use simulator pose</button>
            <button className="force-source-text-button" onClick={() => onUsePose(startArm, startWrist)}>Apply entered pose to simulator</button>
            <span>Club {wrappedDegrees(radians(startArm + startWrist)).toFixed(1)}° absolute</span>
            <label title="Fixed hub uses one physical reference frame. Impact alignment moves only the camera for each card."><input id="force-fixed-hub" type="checkbox" checked={alignment === 'fixed_hub'} onChange={event => setAlignment(event.target.checked ? 'fixed_hub' : 'impact_aligned')} /> Fixed hub comparison</label>
        </div>
        {running && <div className="force-source-progress" role="status"><span style={{ width: `${Math.min(100, 100 * progress.completed / progress.total)}%` }} /><p>{progress.label}: {progress.completed}/{progress.total} candidates · best {Number.isFinite(progress.bestScore) ? progress.bestScore.toFixed(3) : '—'}</p></div>}
        {message && <div className="error-box">{message}</div>}

        {artifact && <>
            <div className="force-source-provenance"><span>{artifact.model}</span><span>Control: bounded Bernstein degree 6</span><span>Basis: {artifact.comparison_contract.constraints.studyMode.replace('_', ' ')}</span><span>Coordinates: shoulder absolute / wrist relative</span><span>Contract: {artifact.comparison_contract.id}</span><span>{artifact.scenarios.length}/6 certified objectives</span><button className="force-source-text-button" onClick={exportArtifact}>Export current study JSON</button></div>
            <div className="force-source-scenario-toggles" aria-label="Visible optimization scenarios">{artifact.scenarios.map(item => <label key={item.objective} style={{ borderColor: OBJECTIVE_COLORS[item.objective] }}><input type="checkbox" checked={selected.has(item.objective)} onChange={() => toggleScenario(item.objective)} />{OBJECTIVE_LABELS[item.objective]}</label>)}</div>
            <div className="force-source-playback">
                <button className="btn btn-secondary" onClick={() => setPlaying(value => !value)}>{playing ? 'Pause' : 'Play'}</button>
                <button className="btn btn-secondary" onClick={() => setTime(0)}>Restart</button>
                <label>Speed<select aria-label="Playback speed" value={playbackRate} onChange={event => setPlaybackRate(Number(event.target.value))}>{[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3].map(rate => <option key={rate} value={rate}>{rate.toFixed(2)}×</option>)}</select></label>
                <input aria-label="Comparison time" type="range" min="0" max={Math.max(maxTime, 0.001)} step="0.00025" value={Math.min(time, maxTime)} onChange={event => { setTime(Number(event.target.value)); setPlaying(false); }} />
                <output>{time.toFixed(4)} s</output>
            </div>
            <p className="force-source-frame-note">{alignment === 'fixed_hub' ? 'Fixed physical frame: every card uses the same shoulder hub, scale, and registered starting pose. No impact-camera guides are shown.' : 'Impact-aligned camera: each card is translated so its impact reaches the camera crosshair; this is not physical hub motion.'}</p>
            <ForceSourceResults scenarios={visible} time={time} params={artifact.parameters ?? params} alignment={alignment} constraints={artifact.comparison_contract.constraints} />
            <aside className="force-source-caveat"><strong>Interpretation boundary.</strong> {artifact.interpretation_limits.join(' ')}</aside>
        </>}
    </section>;
}
