import { useMemo } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { interpolateSeries, pendulumThumbnailGeometry, thumbnailOrigin, type AnimationAlignment } from '../forceSourceView';
import { OBJECTIVE_LABELS, type ForceSourceObjective, type ForceSourceScenario } from '../forceSourceStudy';
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
    return <svg viewBox="0 0 192 176" role="img" aria-label={`${OBJECTIVE_LABELS[scenario.objective]} swing at ${time.toFixed(3)} seconds`}>
        <line x1="14" y1={impact.tipY} x2="178" y2={impact.tipY} stroke="#5a647d" strokeDasharray="4 5" />
        <circle cx={impact.tipX} cy={impact.tipY} r="3.5" fill="#fff" stroke="#aab4ca" />
        <circle cx={geometry.originX} cy={geometry.originY} r="4" fill="#eaf0ff" />
        <line x1={geometry.originX} y1={geometry.originY} x2={geometry.wristX} y2={geometry.wristY} stroke="#56d6c8" strokeWidth="7" strokeLinecap="round" />
        <line x1={geometry.wristX} y1={geometry.wristY} x2={geometry.tipX} y2={geometry.tipY} stroke="#e8edf9" strokeWidth="4" strokeLinecap="round" />
        <circle cx={geometry.wristX} cy={geometry.wristY} r="5" fill="#ffbd4a" />
        <circle cx={geometry.tipX} cy={geometry.tipY} r="7" fill={OBJECTIVE_COLORS[scenario.objective]} />
    </svg>;
}

type PlotField = 'clubhead_speed_m_s' | 'shoulder_torque_nm' | 'wrist_torque_nm';

function ComparisonPlot({ title, unit, scenarios, field }: {
    title: string; unit: string; scenarios: ForceSourceScenario[]; field: PlotField;
}) {
    const data = useMemo(() => {
        const maxTime = Math.max(...scenarios.map(item => item.impact_time_s));
        return Array.from({ length: 241 }, (_, index) => {
            const time = maxTime * index / 240;
            const row: Record<string, number | null> = { time: +time.toFixed(5) };
            for (const scenario of scenarios) {
                row[scenario.objective] = time <= scenario.impact_time_s
                    ? interpolateSeries(scenario.series.time_s, scenario.series[field], time) : null;
            }
            return row;
        });
    }, [field, scenarios]);
    return <div className="force-source-chart" data-testid={`force-source-plot-${field}`}>
        <h4>{title} <span className="force-source-unit">[{unit}]</span></h4>
        <ResponsiveContainer width="100%" height={205}>
            <LineChart data={data} margin={{ top: 6, right: 12, bottom: 2, left: 4 }}>
                <CartesianGrid stroke="#30384e" strokeOpacity={0.65} />
                <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} stroke="#8791aa" tick={{ fontSize: 10 }} />
                <YAxis stroke="#8791aa" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#111522', border: '1px solid #3c455e' }} />
                <Legend wrapperStyle={{ fontSize: 10 }} formatter={(value: string) => OBJECTIVE_LABELS[value as ForceSourceObjective]} />
                {scenarios.map(item => <Line key={item.objective} type="monotone" dataKey={item.objective} stroke={OBJECTIVE_COLORS[item.objective]} dot={false} strokeWidth={2} />)}
            </LineChart>
        </ResponsiveContainer>
    </div>;
}

export function ForceSourceResults({ scenarios, time, params, alignment }: {
    scenarios: ForceSourceScenario[]; time: number; params: PendulumParams; alignment: AnimationAlignment;
}) {
    return <>
        <div className="force-source-animation-grid">
            {scenarios.map(scenario => <article key={scenario.objective} style={{ borderColor: OBJECTIVE_COLORS[scenario.objective] }}>
                <h3>{OBJECTIVE_LABELS[scenario.objective]}</h3>
                <PendulumThumbnail scenario={scenario} time={time} params={params} alignment={alignment} />
                <dl>
                    <div><dt>Objective score</dt><dd>{scenario.score.toFixed(2)} {OBJECTIVE_UNITS[scenario.objective]}</dd></div>
                    <div><dt>Impact speed</dt><dd>{scenario.series.clubhead_speed_m_s[scenario.series.clubhead_speed_m_s.length - 1]?.toFixed(2)} m/s</dd></div>
                    {scenario.impact_diagnostics && <div><dt>Impact path</dt><dd>{scenario.impact_diagnostics.path_angle_deg.toFixed(1)}° · {(100 * scenario.impact_diagnostics.bottom_reach_fraction).toFixed(0)}% reach</dd></div>}
                    <div><dt>Wrist program</dt><dd>−{scenario.candidate.wrist_restrain_nm.toFixed(1)} / +{scenario.candidate.wrist_drive_nm.toFixed(1)} N m</dd></div>
                    <div><dt>Robust qualification</dt><dd>{scenario.robustness.sample_count <= 1 ? 'Nominal only' : `${(100 * scenario.robustness.qualification_rate).toFixed(0)}%`}</dd></div>
                </dl>
            </article>)}
        </div>
        {scenarios.length > 0 && <div className="force-source-plot-grid">
            <ComparisonPlot title="Clubhead speed" unit="m/s" scenarios={scenarios} field="clubhead_speed_m_s" />
            <ComparisonPlot title="Shoulder torque" unit="N m" scenarios={scenarios} field="shoulder_torque_nm" />
            <ComparisonPlot title="Wrist torque" unit="N m" scenarios={scenarios} field="wrist_torque_nm" />
        </div>}
    </>;
}
