import { useMemo } from 'react';
import {
    CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine,
    ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

import {
    interpolateSeries, pendulumThumbnailGeometry, SHARED_IMPACT,
    thumbnailOrigin, THUMBNAIL_VIEWBOX, type AnimationAlignment,
} from '../forceSourceView';
import {
    FORCE_SOURCE_OBJECTIVES, OBJECTIVE_LABELS, profileDiagnostics,
    scoreForceSourceSeries, type ForceSourceObjective,
    type ForceSourceConstraints, type ForceSourceScenario, type ForceSourceSeries,
} from '../forceSourceStudy';
import type { PendulumParams } from '../physics';

export const OBJECTIVE_COLORS: Record<ForceSourceObjective, string> = {
    coriolis_impulse: '#31d6c3', coriolis_energy_transfer: '#50a7ff',
    centrifugal_impulse: '#ffbd4a', centrifugal_energy_transfer: '#ff7a68',
    clubhead_speed: '#b58cff', hand_path_impulse: '#f38fd6',
};

const OBJECTIVE_UNITS: Record<ForceSourceObjective, string> = {
    coriolis_impulse: 'N s', coriolis_energy_transfer: 'J',
    centrifugal_impulse: 'N s', centrifugal_energy_transfer: 'J',
    clubhead_speed: 'm/s', hand_path_impulse: 'N s',
};

function registeredCoordinate(value: number): number {
    return Number(value.toFixed(6));
}

function PendulumThumbnail({ scenario, time, params, alignment }: {
    scenario: ForceSourceScenario; time: number; params: PendulumParams; alignment: AnimationAlignment;
}) {
    const finalIndex = scenario.series.time_s.length - 1;
    const neutralImpact = pendulumThumbnailGeometry(
        scenario.series.arm_angle_rad[finalIndex], scenario.series.wrist_cock_rad[finalIndex], params,
    );
    const origin = thumbnailOrigin(alignment, neutralImpact);
    const impact = pendulumThumbnailGeometry(
        scenario.series.arm_angle_rad[finalIndex], scenario.series.wrist_cock_rad[finalIndex], params, origin,
    );
    const sampleTime = Math.min(time, scenario.impact_time_s);
    const arm = interpolateSeries(scenario.series.time_s, scenario.series.arm_angle_rad, sampleTime);
    const wrist = interpolateSeries(scenario.series.time_s, scenario.series.wrist_cock_rad, sampleTime);
    const geometry = pendulumThumbnailGeometry(arm, wrist, params, origin);
    return <div className="force-source-animation-stage">
        <svg
            data-animation-frame data-alignment={alignment}
            data-hub-x={registeredCoordinate(geometry.originX)}
            data-hub-y={registeredCoordinate(geometry.originY)}
            data-impact-x={registeredCoordinate(impact.tipX)}
            data-impact-y={registeredCoordinate(impact.tipY)}
            data-start-arm={registeredCoordinate(scenario.series.arm_angle_rad[0])}
            data-start-wrist={registeredCoordinate(scenario.series.wrist_cock_rad[0])}
            width={THUMBNAIL_VIEWBOX.width} height={THUMBNAIL_VIEWBOX.height}
            viewBox={`0 0 ${THUMBNAIL_VIEWBOX.width} ${THUMBNAIL_VIEWBOX.height}`}
            preserveAspectRatio="xMidYMid meet" role="img"
            aria-label={`${OBJECTIVE_LABELS[scenario.objective]} swing at ${time.toFixed(3)} seconds`}
        >
            <title>{alignment === 'fixed_hub'
                ? 'Physical frame: the shoulder hub remains fixed at the same coordinate in every card.'
                : 'Camera-aligned frame: the crosshair is a visual impact target, not a physical joint.'}</title>
            {alignment === 'impact_aligned' && <g data-role="camera-impact-target" aria-label="Camera-only impact target">
                <line x1={SHARED_IMPACT.x - 5} y1={SHARED_IMPACT.y} x2={SHARED_IMPACT.x + 5} y2={SHARED_IMPACT.y} stroke="#8b96ad" />
                <line x1={SHARED_IMPACT.x} y1={SHARED_IMPACT.y - 5} x2={SHARED_IMPACT.x} y2={SHARED_IMPACT.y + 5} stroke="#8b96ad" />
            </g>}
            <circle data-role="hub-guide" cx={geometry.originX} cy={geometry.originY} r="8" fill="none" stroke="#69758f" strokeWidth="1" />
            <circle data-role="hub" aria-label="Fixed shoulder hub" cx={geometry.originX} cy={geometry.originY} r="4" fill="#111522" stroke="#eaf0ff" strokeWidth="2" />
            <line x1={geometry.originX} y1={geometry.originY} x2={geometry.wristX} y2={geometry.wristY} stroke="#56d6c8" strokeWidth="7" strokeLinecap="round" />
            <line x1={geometry.wristX} y1={geometry.wristY} x2={geometry.tipX} y2={geometry.tipY} stroke="#e8edf9" strokeWidth="4" strokeLinecap="round" />
            <circle data-role="wrist-joint" cx={geometry.wristX} cy={geometry.wristY} r="5" fill="#ffbd4a" />
            <circle data-role="clubhead" cx={geometry.tipX} cy={geometry.tipY} r="7" fill={OBJECTIVE_COLORS[scenario.objective]} />
        </svg>
    </div>;
}

type PlotField = Exclude<keyof ForceSourceSeries, 'time_s'>;
interface PlotSpec { title: string; unit: string; field: PlotField; transform?: (value: number) => number }
const degrees = (value: number) => value * 180 / Math.PI;
const PLOT_SPECS: PlotSpec[] = [
    { title: 'Clubhead speed', unit: 'm/s', field: 'clubhead_speed_m_s' },
    { title: 'Shoulder torque', unit: 'N m', field: 'shoulder_torque_nm' },
    { title: 'Wrist torque', unit: 'N m', field: 'wrist_torque_nm' },
    { title: 'Arm angle', unit: 'deg', field: 'arm_angle_rad', transform: degrees },
    { title: 'Wrist cock', unit: 'deg', field: 'wrist_cock_rad', transform: degrees },
    { title: 'Arm angular velocity', unit: 'rad/s', field: 'arm_angular_velocity_rad_s' },
    { title: 'Wrist angular velocity', unit: 'rad/s', field: 'wrist_angular_velocity_rad_s' },
    { title: 'Coriolis tangential force', unit: 'N', field: 'coriolis_tangent_force_n' },
    { title: 'Coriolis transfer power', unit: 'W', field: 'coriolis_power_w' },
    { title: 'Centrifugal tangential force', unit: 'N', field: 'squared_speed_tangent_force_n' },
    { title: 'Centrifugal transfer power', unit: 'W', field: 'squared_speed_power_w' },
    { title: 'Hand-path tangential force', unit: 'N', field: 'hand_path_tangent_force_n' },
    { title: 'Shoulder actuator power', unit: 'W', field: 'shoulder_actuator_power_w' },
    { title: 'Wrist actuator power', unit: 'W', field: 'wrist_actuator_power_w' },
    { title: 'Total actuator power', unit: 'W', field: 'total_actuator_power_w' },
    { title: 'Cumulative positive actuator work', unit: 'J', field: 'cumulative_positive_actuator_work_j' },
    { title: 'Cumulative net actuator work', unit: 'J', field: 'cumulative_net_actuator_work_j' },
];

function ComparisonPlot({ spec, scenarios, constraints }: {
    spec: PlotSpec; scenarios: ForceSourceScenario[]; constraints: ForceSourceConstraints;
}) {
    const data = useMemo(() => {
        const maxTime = Math.max(...scenarios.map(item => item.impact_time_s));
        return Array.from({ length: 321 }, (_, index) => {
            const time = maxTime * index / 320;
            const row: Record<string, number | null> = { time: +time.toFixed(5) };
            for (const scenario of scenarios) {
                const values = scenario.series[spec.field];
                const value = time <= scenario.impact_time_s && values
                    ? interpolateSeries(scenario.series.time_s, values, time) : null;
                row[scenario.objective] = value === null ? null : (spec.transform?.(value) ?? value);
            }
            return row;
        });
    }, [scenarios, spec]);
    return <div className="force-source-chart" data-testid={`force-source-plot-${spec.field}`}>
        <h4>{spec.title} <span className="force-source-unit">[{spec.unit}]</span></h4>
        <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data} margin={{ top: 6, right: 12, bottom: 2, left: 4 }}>
                <CartesianGrid stroke="#30384e" strokeOpacity={0.65} />
                <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} stroke="#8791aa" tick={{ fontSize: 10 }} label={{ value: 'time [s]', position: 'insideBottomRight', offset: -2 }} />
                <YAxis stroke="#8791aa" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#111522', border: '1px solid #3c455e' }} />
                <Legend wrapperStyle={{ fontSize: 10 }} formatter={(value: string) => OBJECTIVE_LABELS[value as ForceSourceObjective]} />
                {spec.field === 'clubhead_speed_m_s' && constraints.studyMode === 'equal_speed' && <ReferenceArea
                    y1={constraints.targetClubheadSpeedMps}
                    y2={constraints.targetClubheadSpeedMps + constraints.speedToleranceMps}
                    fill="#f4f7ff" fillOpacity={0.08}
                />}
                {spec.field === 'clubhead_speed_m_s' && <ReferenceLine y={constraints.targetClubheadSpeedMps} stroke="#f4f7ff" strokeDasharray="5 4" label={{ value: `target ${constraints.targetClubheadSpeedMps.toFixed(1)}`, fill: '#f4f7ff', fontSize: 10 }} />}
                {scenarios.map(item => <Line key={item.objective} type="monotone" dataKey={item.objective} stroke={OBJECTIVE_COLORS[item.objective]} dot={false} strokeWidth={2} connectNulls={false} />)}
            </LineChart>
        </ResponsiveContainer>
    </div>;
}

function objectiveRanks(scenarios: ForceSourceScenario[]) {
    const scores = new Map(scenarios.map(scenario => [scenario.objective, scoreForceSourceSeries(scenario.series)]));
    const ranks = new Map<string, number>();
    for (const objective of FORCE_SOURCE_OBJECTIVES) {
        const sorted = [...scenarios].sort((left, right) => (scores.get(right.objective)?.[objective] ?? -Infinity)
            - (scores.get(left.objective)?.[objective] ?? -Infinity));
        let priorValue: number | null = null;
        let priorRank = 0;
        sorted.forEach((scenario, index) => {
            const value = scores.get(scenario.objective)?.[objective] ?? -Infinity;
            const tied = priorValue !== null
                && Math.abs(value - priorValue) <= 1e-10 * Math.max(1, Math.abs(value), Math.abs(priorValue));
            const rank = tied ? priorRank : index + 1;
            ranks.set(`${scenario.objective}|${objective}`, rank);
            priorValue = value;
            priorRank = rank;
        });
    }
    return { scores, ranks };
}

function paretoRanks(scenarios: ForceSourceScenario[]): Map<ForceSourceObjective, number> {
    const scoreMap = new Map(scenarios.map(item => [item.objective, scoreForceSourceSeries(item.series)]));
    const remaining = new Set(scenarios.map(item => item.objective));
    const ranks = new Map<ForceSourceObjective, number>();
    let rank = 1;
    while (remaining.size > 0) {
        const front = [...remaining].filter(candidate => ![...remaining].some(other => {
            if (candidate === other) return false;
            const candidateScores = scoreMap.get(candidate)!;
            const otherScores = scoreMap.get(other)!;
            return FORCE_SOURCE_OBJECTIVES.every(objective => otherScores[objective] >= candidateScores[objective])
                && FORCE_SOURCE_OBJECTIVES.some(objective => otherScores[objective] > candidateScores[objective]);
        }));
        for (const objective of front) { ranks.set(objective, rank); remaining.delete(objective); }
        rank += 1;
    }
    return ranks;
}

function scalarRanks(
    scenarios: ForceSourceScenario[],
    value: (scenario: ForceSourceScenario) => number,
    lowerIsBetter = false,
): Map<ForceSourceObjective, number> {
    const sorted = [...scenarios].sort((left, right) =>
        (lowerIsBetter ? 1 : -1) * (value(left) - value(right)));
    const ranks = new Map<ForceSourceObjective, number>();
    let priorValue: number | null = null;
    let priorRank = 0;
    sorted.forEach((scenario, index) => {
        const current = value(scenario);
        const tied = priorValue !== null
            && Math.abs(current - priorValue) <= 1e-10 * Math.max(1, Math.abs(current), Math.abs(priorValue));
        const rank = tied ? priorRank : index + 1;
        ranks.set(scenario.objective, rank);
        priorValue = current;
        priorRank = rank;
    });
    return ranks;
}

function inputParetoRanks(scenarios: ForceSourceScenario[]): Map<ForceSourceObjective, number> {
    const remaining = new Set(scenarios.map(item => item.objective));
    const ranks = new Map<ForceSourceObjective, number>();
    const speed = (item: ForceSourceScenario) => item.series.clubhead_speed_m_s[
        item.series.clubhead_speed_m_s.length - 1
    ] ?? 0;
    let rank = 1;
    while (remaining.size > 0) {
        const front = [...remaining].filter(candidate => ![...remaining].some(other => {
            if (candidate === other) return false;
            const left = scenarios.find(item => item.objective === candidate)!;
            const right = scenarios.find(item => item.objective === other)!;
            return speed(right) >= speed(left)
                && right.effort.total_positive_work_j <= left.effort.total_positive_work_j
                && right.effort.squared_torque_effort_nm2_s <= left.effort.squared_torque_effort_nm2_s
                && (speed(right) > speed(left)
                    || right.effort.total_positive_work_j < left.effort.total_positive_work_j
                    || right.effort.squared_torque_effort_nm2_s < left.effort.squared_torque_effort_nm2_s);
        }));
        for (const objective of front) { ranks.set(objective, rank); remaining.delete(objective); }
        rank += 1;
    }
    return ranks;
}

function TradeoffTable({ scenarios, targetClubheadSpeedMps }: {
    scenarios: ForceSourceScenario[]; targetClubheadSpeedMps: number;
}) {
    const { scores, ranks } = useMemo(() => objectiveRanks(scenarios), [scenarios]);
    const pareto = useMemo(() => paretoRanks(scenarios), [scenarios]);
    return <div className="force-source-table-wrap">
        <h3>Cross-objective tradeoffs and ranks</h3>
        <p>Each cell shows the measured value and rank among visible strategies. Pareto rank 1 means no visible strategy improves every objective simultaneously.</p>
        <table className="force-source-tradeoff-table">
            <thead><tr><th>Optimized strategy</th><th>Pareto</th>{FORCE_SOURCE_OBJECTIVES.map(objective => <th key={objective}>{OBJECTIVE_LABELS[objective]} [{OBJECTIVE_UNITS[objective]}]</th>)}</tr></thead>
            <tbody>{scenarios.map(scenario => {
                const speed = scenario.series.clubhead_speed_m_s[
                    scenario.series.clubhead_speed_m_s.length - 1
                ] ?? 0;
                return <tr key={scenario.objective}><th>{OBJECTIVE_LABELS[scenario.objective]}<small>{speed >= targetClubheadSpeedMps ? ' target reached' : ` ${(speed - targetClubheadSpeedMps).toFixed(2)} m/s`}</small></th>
                    <td>{pareto.get(scenario.objective)}</td>
                    {FORCE_SOURCE_OBJECTIVES.map(objective => <td key={objective}>{scores.get(scenario.objective)?.[objective].toFixed(2)} <small>#{ranks.get(`${scenario.objective}|${objective}`)}</small></td>)}
                </tr>;
            })}</tbody>
        </table>
    </div>;
}

function StrategyTable({ scenarios, constraints }: {
    scenarios: ForceSourceScenario[]; constraints: ForceSourceConstraints;
}) {
    const profileGroups = new Map<string, ForceSourceScenario[]>();
    for (const scenario of scenarios) {
        profileGroups.set(scenario.profile_id, [...(profileGroups.get(scenario.profile_id) ?? []), scenario]);
    }
    const shoulderLimit = Math.max(Math.abs(constraints.shoulderTorqueNm.min), Math.abs(constraints.shoulderTorqueNm.max));
    const speedValue = (scenario: ForceSourceScenario) => scenario.series.clubhead_speed_m_s[
        scenario.series.clubhead_speed_m_s.length - 1
    ] ?? 0;
    const speedRanks = scalarRanks(scenarios, speedValue);
    const workRanks = scalarRanks(scenarios, scenario => scenario.effort.total_positive_work_j, true);
    const activationRanks = scalarRanks(scenarios, scenario => scenario.effort.squared_torque_effort_nm2_s, true);
    const inputPareto = inputParetoRanks(scenarios);
    return <div className="force-source-table-wrap">
        <h3>Input work, activation, and control strategy</h3>
        <p>Positive work counts energy supplied by either actuator; negative work records braking separately. Input Pareto rank 1 means no displayed strategy is at least as fast with no more positive work and squared activation. Equal profile IDs mean the objectives selected exactly the same polynomial controls.</p>
        <table className="force-source-strategy-table">
            <thead><tr><th>Strategy / profile</th><th>Input Pareto</th><th>Speed [m/s]</th><th>Positive / net / braking work [J]</th><th>Shoulder / wrist net work [J]</th><th>Peak shoulder / wrist [N m]</th><th>Peak utilization</th><th>RMS shoulder / wrist [N m]</th><th>∫|τ|dt [N m s]</th><th>∫τ²dt [N² m² s]</th><th>Peak power shoulder / wrist [W]</th><th>Peak slew shoulder / wrist [N m/s]</th><th>Reversal / low-torque [s]</th></tr></thead>
            <tbody>{scenarios.map(scenario => {
                const diagnostic = profileDiagnostics(scenario.candidate, constraints.transitionTorqueNm);
                const speed = scenario.series.clubhead_speed_m_s[
                    scenario.series.clubhead_speed_m_s.length - 1
                ] ?? 0;
                const group = profileGroups.get(scenario.profile_id) ?? [];
                return <tr key={scenario.objective}><th>{OBJECTIVE_LABELS[scenario.objective]}<small>{scenario.profile_id}{group.length > 1 ? ` · shared by ${group.length} objectives` : ''}</small></th>
                    <td>{inputPareto.get(scenario.objective)}</td>
                    <td>{speed.toFixed(2)}<small>speed #{speedRanks.get(scenario.objective)}</small></td>
                    <td>{scenario.effort.total_positive_work_j.toFixed(1)} / {scenario.effort.total_net_work_j.toFixed(1)} / {scenario.effort.total_negative_work_j.toFixed(1)}<small>positive-work #{workRanks.get(scenario.objective)}</small></td>
                    <td>{scenario.effort.shoulder_net_work_j.toFixed(1)} / {scenario.effort.wrist_net_work_j.toFixed(1)}</td>
                    <td>{diagnostic.peak_shoulder_torque_nm.toFixed(1)} / {diagnostic.peak_wrist_torque_nm.toFixed(1)}</td>
                    <td>{(100 * diagnostic.peak_shoulder_torque_nm / shoulderLimit).toFixed(0)}% / {(100 * diagnostic.peak_wrist_torque_nm / constraints.wristTorqueLimitNm).toFixed(0)}%</td>
                    <td>{diagnostic.rms_shoulder_torque_nm.toFixed(1)} / {diagnostic.rms_wrist_torque_nm.toFixed(1)}</td>
                    <td>{scenario.effort.absolute_torque_impulse_nm_s.toFixed(1)}</td>
                    <td>{scenario.effort.squared_torque_effort_nm2_s.toFixed(0)}<small>activation #{activationRanks.get(scenario.objective)}</small></td>
                    <td>{scenario.effort.peak_shoulder_power_w.toFixed(0)} / {scenario.effort.peak_wrist_power_w.toFixed(0)}</td>
                    <td>{diagnostic.peak_shoulder_slew_nm_s.toFixed(0)} / {diagnostic.peak_wrist_slew_nm_s.toFixed(0)}</td>
                    <td>{diagnostic.wrist_reversal_time_s?.toFixed(3) ?? '—'} / {diagnostic.wrist_transition_duration_s.toFixed(3)}</td>
                </tr>;
            })}</tbody>
        </table>
    </div>;
}

function CoefficientTable({ scenarios }: { scenarios: ForceSourceScenario[] }) {
    return <details className="force-source-coefficients"><summary>Sixth-order polynomial coefficients</summary>
        <div className="force-source-table-wrap"><table><thead><tr><th>Strategy</th><th>Joint</th>{Array.from({ length: 7 }, (_, index) => <th key={index}>C{index} [N m]</th>)}<th>Duration [s]</th></tr></thead>
            <tbody>{scenarios.flatMap(scenario => (['shoulder', 'wrist'] as const).map(joint => <tr key={`${scenario.objective}-${joint}`}>
                <th>{OBJECTIVE_LABELS[scenario.objective]}</th><td>{joint}</td>
                {(joint === 'shoulder' ? scenario.candidate.shoulder_coefficients_nm : scenario.candidate.wrist_coefficients_nm).map((value, index) => <td key={index}>{value.toFixed(2)}</td>)}
                <td>{scenario.candidate.profile_duration_s.toFixed(3)}</td>
            </tr>))}</tbody></table></div>
    </details>;
}

export function ForceSourceResults({ scenarios, time, params, alignment, constraints }: {
    scenarios: ForceSourceScenario[]; time: number; params: PendulumParams;
    alignment: AnimationAlignment; constraints: ForceSourceConstraints;
}) {
    const profileGroups = useMemo(() => new Map(
        [...new Set(scenarios.map(item => item.profile_id))].map(profileId => [
            profileId, scenarios.filter(item => item.profile_id === profileId),
        ]),
    ), [scenarios]);
    return <>
        <div className="force-source-animation-key" aria-label="Animation marker key">
            <span><i className="force-source-marker force-source-marker--hub" />Fixed shoulder hub</span>
            <span><i className="force-source-marker force-source-marker--wrist" />Wrist joint</span>
            <span><i className="force-source-marker force-source-marker--clubhead" />Clubhead</span>
            {alignment === 'impact_aligned' && <span><i className="force-source-marker force-source-marker--camera" />Camera-only impact target</span>}
        </div>
        <div className="force-source-animation-grid">{scenarios.map(scenario => {
            const diagnostics = profileDiagnostics(scenario.candidate, constraints.transitionTorqueNm);
            const speed = scenario.series.clubhead_speed_m_s[
                scenario.series.clubhead_speed_m_s.length - 1
            ] ?? 0;
            return <article key={scenario.objective} style={{ borderColor: OBJECTIVE_COLORS[scenario.objective] }}>
                <h3>{OBJECTIVE_LABELS[scenario.objective]}</h3>
                <PendulumThumbnail scenario={scenario} time={time} params={params} alignment={alignment} />
                <dl>
                    <div><dt>Objective score</dt><dd>{scenario.score.toFixed(2)} {OBJECTIVE_UNITS[scenario.objective]}</dd></div>
                    <div><dt>Control profile</dt><dd>{scenario.profile_id}{(profileGroups.get(scenario.profile_id)?.length ?? 0) > 1 ? ` · shared by ${profileGroups.get(scenario.profile_id)?.length} objectives` : ''}</dd></div>
                    <div><dt>Impact speed</dt><dd>{speed.toFixed(2)} m/s · {(speed * 2.23694).toFixed(1)} mph</dd></div>
                    <div><dt>Target gap</dt><dd>{(speed - constraints.targetClubheadSpeedMps).toFixed(2)} m/s</dd></div>
                    <div><dt>Positive work</dt><dd>{scenario.effort.total_positive_work_j.toFixed(1)} J</dd></div>
                    {scenario.impact_diagnostics && <div><dt>Impact path</dt><dd>{scenario.impact_diagnostics.path_angle_deg.toFixed(1)}° · {(100 * scenario.impact_diagnostics.bottom_reach_fraction).toFixed(0)}% reach</dd></div>}
                    <div><dt>Wrist reversal</dt><dd>{diagnostics.wrist_reversal_time_s?.toFixed(3) ?? '—'} s · {diagnostics.wrist_transition_duration_s.toFixed(3)} s low torque</dd></div>
                    <div><dt>Robust qualification</dt><dd>{scenario.robustness.sample_count <= 1 ? 'Nominal only' : `${(100 * scenario.robustness.qualification_rate).toFixed(0)}%`}</dd></div>
                </dl>
            </article>;
        })}</div>
        {scenarios.length > 0 && <>
            <TradeoffTable scenarios={scenarios} targetClubheadSpeedMps={constraints.targetClubheadSpeedMps} />
            <StrategyTable scenarios={scenarios} constraints={constraints} />
            <CoefficientTable scenarios={scenarios} />
            <div className="force-source-plot-grid">{PLOT_SPECS.map(spec => <ComparisonPlot key={spec.field} spec={spec} scenarios={scenarios} constraints={constraints} />)}</div>
        </>}
    </>;
}
