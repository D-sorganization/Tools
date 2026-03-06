import React, { useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import type { SimulationResult } from '../physics';
import {
    kineticEnergy, potentialEnergy, totalEnergy, frictionTorqueVector,
    jointVelocities, baseForce, computeAccelerations, controlVector,
} from '../physics';
import type { UnitPreferences } from '../units';
import {
    speedFromSI, angularSpeedFromSI, forceFromSI, energyFromSI, torqueFromSI,
} from '../units';

// ── Types ────────────────────────────────────────────────────────────────────

interface AnalysisPlotsProps {
    result: SimulationResult;
    units: UnitPreferences;
    /** If set, only render this specific plot in expanded detail. */
    detailPlot?: string | null;
}

// ── Shared chart styling (DRY) ──────────────────────────────────────────────

const COLORS = {
    theta1: '#60aaff', phi: '#ff9955',
    dtheta1: '#44ddaa', dphi: '#ff55aa',
    KE: '#60aaff', PE: '#ff9955', E: '#ffffff',
    tau1: '#60aaff', tau2: '#ff9955',
    wristSpeed: '#44ddaa', tipSpeed: '#ff55aa',
    fx: '#60aaff', fy: '#ff9955', fMag: '#ffffff',
    cvx: '#aa66ff', cvy: '#66ffaa', cvMag: '#ff66cc',
};

const CHART_STYLE: React.CSSProperties = {
    backgroundColor: '#1a1a28',
    border: '1px solid #303048',
    borderRadius: 8,
    padding: 8,
    marginBottom: 12,
};

const DETAIL_CHART_STYLE: React.CSSProperties = {
    ...CHART_STYLE,
    marginBottom: 0,
};

const AXIS_STYLE = { stroke: '#6060a0', fontSize: 10 };
const GRID_STYLE = { stroke: '#303060', strokeOpacity: 0.5 };
const TOOLTIP_STYLE = { backgroundColor: '#1a1a28', border: '1px solid #404060', fontSize: 11 };
const LEGEND_STYLE = { fontSize: 11, color: '#c0c0e0' };
const CHART_MARGIN = { top: 4, right: 12, left: -12, bottom: 0 };

const CHART_HEIGHT_COMPACT = 150;
const CHART_HEIGHT_DETAIL = 400;

/** Title bar for each chart (DRY). */
const ChartTitle: React.FC<{ text: string }> = ({ text }) => (
    <div style={{ color: '#c0c0e0', fontSize: 11, marginBottom: 4 }}>{text}</div>
);

// ── Stride computation (DRY) ────────────────────────────────────────────────

function computeStride(length: number, maxPoints: number = 500): number {
    return Math.max(1, Math.floor(length / maxPoints));
}

function sampleIndices(length: number, stride: number): number[] {
    const indices: number[] = [];
    for (let i = 0; i < length; i += stride) indices.push(i);
    return indices;
}

// ── Main component ──────────────────────────────────────────────────────────

export const AnalysisPlots: React.FC<AnalysisPlotsProps> = ({ result, units, detailPlot }) => {
    const { t, states, params, torqueFunc, limits, clamp } = result;

    const stride = computeStride(t.length);
    const indices = useMemo(() => sampleIndices(t.length, stride), [t.length, stride]);

    // ── Angle data ──────────────────────────────────────────────────────
    const angleData = useMemo(() =>
        indices.map(i => ({
            t: +t[i].toFixed(3),
            'Arms (θ₁)': +(states[i][0] * 180 / Math.PI).toFixed(2),
            'Shaft (φ)': +(states[i][1] * 180 / Math.PI).toFixed(2),
        })),
        [t, states, indices],
    );

    // ── Angular velocity data ───────────────────────────────────────────
    const angVelData = useMemo(() =>
        indices.map(i => ({
            t: +t[i].toFixed(3),
            'Arms ω': +angularSpeedFromSI(states[i][2], units.angularSpeed).toFixed(2),
            'Shaft ω': +angularSpeedFromSI(states[i][2] + states[i][3], units.angularSpeed).toFixed(2),
            'Wrist rel ω': +angularSpeedFromSI(states[i][3], units.angularSpeed).toFixed(2),
        })),
        [t, states, indices, units.angularSpeed],
    );

    // ── Linear speed data ───────────────────────────────────────────────
    const linearSpeedData = useMemo(() =>
        indices.map(i => {
            const jv = jointVelocities(states[i], params);
            return {
                t: +t[i].toFixed(3),
                'Wrist': +speedFromSI(jv.wristSpeed, units.speed).toFixed(2),
                'Tip': +speedFromSI(jv.tipSpeed, units.speed).toFixed(2),
            };
        }),
        [t, states, params, indices, units.speed],
    );

    // ── Energy data ─────────────────────────────────────────────────────
    const energyData = useMemo(() =>
        indices.map(i => ({
            t: +t[i].toFixed(3),
            KE: +energyFromSI(kineticEnergy(states[i], params), units.energy).toFixed(4),
            PE: +energyFromSI(potentialEnergy(states[i], params), units.energy).toFixed(4),
            Total: +energyFromSI(totalEnergy(states[i], params), units.energy).toFixed(4),
        })),
        [t, states, params, indices, units.energy],
    );

    // ── Friction torque data ────────────────────────────────────────────
    const torqueData = useMemo(() =>
        indices.map(i => {
            const [tf1, tf2] = frictionTorqueVector(states[i][2], states[i][3], params);
            return {
                t: +t[i].toFixed(3),
                'Shoulder fric': +torqueFromSI(tf1, units.torque).toFixed(4),
                'Wrist fric': +torqueFromSI(tf2, units.torque).toFixed(4),
            };
        }),
        [t, states, params, indices, units.torque],
    );

    // ── Base force data ─────────────────────────────────────────────────
    const baseForceData = useMemo(() =>
        indices.map(i => {
            const qdd = computeAccelerations(states[i], t[i], params, torqueFunc, limits, clamp);
            const bf = baseForce(states[i], qdd, params);
            return {
                t: +t[i].toFixed(3),
                'Fx': +forceFromSI(bf.fx, units.force).toFixed(2),
                'Fy': +forceFromSI(bf.fy, units.force).toFixed(2),
                '|F|': +forceFromSI(bf.magnitude, units.force).toFixed(2),
            };
        }),
        [t, states, params, torqueFunc, limits, clamp, indices, units.force],
    );

    // ── Control vector data ─────────────────────────────────────────────
    const controlVectorData = useMemo(() =>
        indices.map(i => {
            const qdd = computeAccelerations(states[i], t[i], params, torqueFunc, limits, clamp);
            const cv = controlVector(states[i], qdd, params, limits);
            return {
                t: +t[i].toFixed(3),
                'CVx': +forceFromSI(cv.cvx, units.force).toFixed(2),
                'CVy': +forceFromSI(cv.cvy, units.force).toFixed(2),
                '|CV|': +forceFromSI(cv.magnitude, units.force).toFixed(2),
            };
        }),
        [t, states, params, torqueFunc, limits, clamp, indices, units.force],
    );

    // If detail mode, show just one plot in full height
    if (detailPlot) {
        return renderDetailPlot(detailPlot, {
            angleData, angVelData, linearSpeedData, energyData,
            torqueData, baseForceData, controlVectorData, units,
        });
    }

    return (
        <div style={{ width: '100%' }}>
            <div style={CHART_STYLE} data-plot-id="angles">
                <ChartTitle text="Joint Angles (deg)" />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={angleData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} label={{ value: 's', position: 'right', style: { fontSize: 9, fill: '#6060a0' } }} />
                        <YAxis {...AXIS_STYLE} label={{ value: 'deg', angle: -90, position: 'insideLeft', style: { fontSize: 9, fill: '#6060a0' } }} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="Arms (θ₁)" stroke={COLORS.theta1} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Shaft (φ)" stroke={COLORS.phi} dot={false} strokeWidth={1.5} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE} data-plot-id="angvel">
                <ChartTitle text={`Angular Velocities (${units.angularSpeed})`} />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={angVelData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="Arms ω" stroke={COLORS.dtheta1} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Shaft ω" stroke={COLORS.dphi} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Wrist rel ω" stroke="#aaaaaa" dot={false} strokeWidth={1} strokeDasharray="4 2" />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE} data-plot-id="linspeed">
                <ChartTitle text={`Linear Joint Speeds (${units.speed})`} />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={linearSpeedData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="Wrist" stroke={COLORS.wristSpeed} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Tip" stroke={COLORS.tipSpeed} dot={false} strokeWidth={1.5} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE} data-plot-id="energy">
                <ChartTitle text={`Energy (${units.energy})`} />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={energyData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="KE" stroke={COLORS.KE} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="PE" stroke={COLORS.PE} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Total" stroke={COLORS.E} dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE} data-plot-id="friction">
                <ChartTitle text={`Friction Torques (${units.torque})`} />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={torqueData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="Shoulder fric" stroke={COLORS.tau1} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Wrist fric" stroke={COLORS.tau2} dot={false} strokeWidth={1.5} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE} data-plot-id="baseforce">
                <ChartTitle text={`Base (Shoulder) Force (${units.force})`} />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={baseForceData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="Fx" stroke={COLORS.fx} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="Fy" stroke={COLORS.fy} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="|F|" stroke={COLORS.fMag} dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE} data-plot-id="controlvec">
                <ChartTitle text={`Control Vector (${units.force})`} />
                <ResponsiveContainer width="100%" height={CHART_HEIGHT_COMPACT}>
                    <LineChart data={controlVectorData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="CVx" stroke={COLORS.cvx} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="CVy" stroke={COLORS.cvy} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="|CV|" stroke={COLORS.cvMag} dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

// ── Detail plot renderer (DRY: reuses same data) ────────────────────────────

interface PlotData {
    angleData: Record<string, number>[];
    angVelData: Record<string, number>[];
    linearSpeedData: Record<string, number>[];
    energyData: Record<string, number>[];
    torqueData: Record<string, number>[];
    baseForceData: Record<string, number>[];
    controlVectorData: Record<string, number>[];
    units: UnitPreferences;
}

function renderDetailPlot(plotId: string, data: PlotData): React.ReactElement {
    const h = CHART_HEIGHT_DETAIL;
    const style = DETAIL_CHART_STYLE;

    const plotConfigs: Record<string, { title: string; data: Record<string, number>[]; lines: { key: string; color: string; dash?: string }[] }> = {
        angles: {
            title: 'Joint Angles (deg)',
            data: data.angleData,
            lines: [
                { key: 'Arms (θ₁)', color: COLORS.theta1 },
                { key: 'Shaft (φ)', color: COLORS.phi },
            ],
        },
        angvel: {
            title: `Angular Velocities (${data.units.angularSpeed})`,
            data: data.angVelData,
            lines: [
                { key: 'Arms ω', color: COLORS.dtheta1 },
                { key: 'Shaft ω', color: COLORS.dphi },
                { key: 'Wrist rel ω', color: '#aaaaaa', dash: '4 2' },
            ],
        },
        linspeed: {
            title: `Linear Joint Speeds (${data.units.speed})`,
            data: data.linearSpeedData,
            lines: [
                { key: 'Wrist', color: COLORS.wristSpeed },
                { key: 'Tip', color: COLORS.tipSpeed },
            ],
        },
        energy: {
            title: `Energy (${data.units.energy})`,
            data: data.energyData,
            lines: [
                { key: 'KE', color: COLORS.KE },
                { key: 'PE', color: COLORS.PE },
                { key: 'Total', color: COLORS.E, dash: '4 2' },
            ],
        },
        friction: {
            title: `Friction Torques (${data.units.torque})`,
            data: data.torqueData,
            lines: [
                { key: 'Shoulder fric', color: COLORS.tau1 },
                { key: 'Wrist fric', color: COLORS.tau2 },
            ],
        },
        baseforce: {
            title: `Base (Shoulder) Force (${data.units.force})`,
            data: data.baseForceData,
            lines: [
                { key: 'Fx', color: COLORS.fx },
                { key: 'Fy', color: COLORS.fy },
                { key: '|F|', color: COLORS.fMag, dash: '4 2' },
            ],
        },
        controlvec: {
            title: `Control Vector (${data.units.force})`,
            data: data.controlVectorData,
            lines: [
                { key: 'CVx', color: COLORS.cvx },
                { key: 'CVy', color: COLORS.cvy },
                { key: '|CV|', color: COLORS.cvMag, dash: '4 2' },
            ],
        },
    };

    const cfg = plotConfigs[plotId];
    if (!cfg) return <div style={{ color: '#ff5566' }}>Unknown plot: {plotId}</div>;

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <div style={style}>
                <ChartTitle text={cfg.title} />
                <ResponsiveContainer width="100%" height={h}>
                    <LineChart data={cfg.data} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        {cfg.lines.map(line => (
                            <Line
                                key={line.key}
                                type="monotone"
                                dataKey={line.key}
                                stroke={line.color}
                                dot={false}
                                strokeWidth={line.dash ? 2 : 1.5}
                                strokeDasharray={line.dash}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

/** List of available plot IDs for the detail selector. */
export const PLOT_IDS = [
    { id: 'angles', label: 'Joint Angles' },
    { id: 'angvel', label: 'Angular Velocities' },
    { id: 'linspeed', label: 'Linear Speeds' },
    { id: 'energy', label: 'Energy' },
    { id: 'friction', label: 'Friction Torques' },
    { id: 'baseforce', label: 'Base Force' },
    { id: 'controlvec', label: 'Control Vector' },
] as const;
