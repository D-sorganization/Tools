import { useMemo } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import {
    interpolateSeries,
    pendulumThumbnailGeometry,
    SHARED_IMPACT,
    thumbnailOrigin,
    THUMBNAIL_VIEWBOX,
    type AnimationAlignment,
} from '../forceSourceView';
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
            data-animation-frame
            data-alignment={alignment}
            data-hub-x={registeredCoordinate(geometry.originX)}
            data-hub-y={registeredCoordinate(geometry.originY)}
            data-impact-x={registeredCoordinate(impact.tipX)}
            data-impact-y={registeredCoordinate(impact.tipY)}
            data-start-arm={registeredCoordinate(scenario.series.arm_angle_rad[0])}
            data-start-wrist={registeredCoordinate(scenario.series.wrist_cock_rad[0])}
            width={THUMBNAIL_VIEWBOX.width}
            height={THUMBNAIL_VIEWBOX.height}
            viewBox={`0 0 ${THUMBNAIL_VIEWBOX.width} ${THUMBNAIL_VIEWBOX.height}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
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
        <div className="force-source-animation-key" aria-label="Animation marker key">
            <span><i className="force-source-marker force-source-marker--hub" />Fixed shoulder hub</span>
            <span><i className="force-source-marker force-source-marker--wrist" />Wrist joint</span>
            <span><i className="force-source-marker force-source-marker--clubhead" />Clubhead</span>
            {alignment === 'impact_aligned' && <span><i className="force-source-marker force-source-marker--camera" />Camera-only impact target</span>}
        </div>
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
