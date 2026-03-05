import React, { useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import type { SimulationResult } from '../physics';
import { kineticEnergy, potentialEnergy, totalEnergy, frictionTorqueVector } from '../physics';

interface AnalysisPlotsProps {
    result: SimulationResult;
}

const COLORS = {
    theta1: '#60aaff',
    phi: '#ff9955',
    dtheta1: '#44ddaa',
    dphi: '#ff55aa',
    KE: '#60aaff',
    PE: '#ff9955',
    E: '#ffffff',
    tau1: '#60aaff',
    tau2: '#ff9955',
};

const CHART_STYLE: React.CSSProperties = {
    backgroundColor: '#1a1a28',
    border: '1px solid #303048',
    borderRadius: 8,
    padding: 8,
    marginBottom: 12,
};

const AXIS_STYLE = { stroke: '#6060a0', fontSize: 10 };
const GRID_STYLE = { stroke: '#303060', strokeOpacity: 0.5 };
const TOOLTIP_STYLE = { backgroundColor: '#1a1a28', border: '1px solid #404060', fontSize: 11 };
const LEGEND_STYLE = { fontSize: 11, color: '#c0c0e0' };
const CHART_MARGIN = { top: 4, right: 12, left: -12, bottom: 0 };

export const AnalysisPlots: React.FC<AnalysisPlotsProps> = ({ result }) => {
    const { t, states, params } = result;

    // Sample at most 500 points for performance (DRY: shared stride logic)
    const stride = Math.max(1, Math.floor(t.length / 500));

    const angleData = useMemo(() =>
        t
            .filter((_, i) => i % stride === 0)
            .map((time, fi) => {
                const i = fi * stride;
                return {
                    t: +time.toFixed(3),
                    theta1: +(states[i][0] * 180 / Math.PI).toFixed(2),
                    phi: +(states[i][1] * 180 / Math.PI).toFixed(2),
                };
            }),
        [t, states, stride],
    );

    const velData = useMemo(() =>
        t
            .filter((_, i) => i % stride === 0)
            .map((time, fi) => {
                const i = fi * stride;
                return {
                    t: +time.toFixed(3),
                    dtheta1: +(states[i][2] * 180 / Math.PI).toFixed(2),
                    dphi: +(states[i][3] * 180 / Math.PI).toFixed(2),
                };
            }),
        [t, states, stride],
    );

    const energyData = useMemo(() =>
        t
            .filter((_, i) => i % stride === 0)
            .map((time, fi) => {
                const i = fi * stride;
                return {
                    t: +time.toFixed(3),
                    KE: +kineticEnergy(states[i], params).toFixed(4),
                    PE: +potentialEnergy(states[i], params).toFixed(4),
                    E: +totalEnergy(states[i], params).toFixed(4),
                };
            }),
        [t, states, params, stride],
    );

    const torqueData = useMemo(() =>
        t
            .filter((_, i) => i % stride === 0)
            .map((time, fi) => {
                const i = fi * stride;
                const [tf1, tf2] = frictionTorqueVector(states[i][2], states[i][3], params);
                return {
                    t: +time.toFixed(3),
                    tau_sh_fric: +tf1.toFixed(4),
                    tau_wr_fric: +tf2.toFixed(4),
                };
            }),
        [t, states, params, stride],
    );

    return (
        <div style={{ width: '100%' }}>
            <div style={CHART_STYLE}>
                <div style={{ color: '#c0c0e0', fontSize: 11, marginBottom: 4 }}>Joint Angles (deg)</div>
                <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={angleData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="theta1" name="theta1 (arm)" stroke={COLORS.theta1} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="phi" name="phi (club)" stroke={COLORS.phi} dot={false} strokeWidth={1.5} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE}>
                <div style={{ color: '#c0c0e0', fontSize: 11, marginBottom: 4 }}>Angular Velocities (deg/s)</div>
                <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={velData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="dtheta1" name="dtheta1" stroke={COLORS.dtheta1} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="dphi" name="dphi" stroke={COLORS.dphi} dot={false} strokeWidth={1.5} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE}>
                <div style={{ color: '#c0c0e0', fontSize: 11, marginBottom: 4 }}>Energy (J)</div>
                <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={energyData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="KE" stroke={COLORS.KE} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="PE" stroke={COLORS.PE} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="E" stroke={COLORS.E} dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div style={CHART_STYLE}>
                <div style={{ color: '#c0c0e0', fontSize: 11, marginBottom: 4 }}>Friction Torques (N·m)</div>
                <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={torqueData} margin={CHART_MARGIN}>
                        <CartesianGrid {...GRID_STYLE} />
                        <XAxis dataKey="t" {...AXIS_STYLE} />
                        <YAxis {...AXIS_STYLE} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={LEGEND_STYLE} />
                        <Line type="monotone" dataKey="tau_sh_fric" name="tau_shoulder" stroke={COLORS.tau1} dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="tau_wr_fric" name="tau_wrist" stroke={COLORS.tau2} dot={false} strokeWidth={1.5} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
