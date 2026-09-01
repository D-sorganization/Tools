import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ForceSourceResults } from '../src/components/ForceSourceResults';
import {
    actuatorEffortMetrics,
    candidateProfileId,
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    FORCE_SOURCE_OBJECTIVES,
    type ForceSourceObjective,
    type ForceSourceScenario,
} from '../src/forceSourceStudy';
import type { PendulumParams } from '../src/physics';

const params: PendulumParams = {
    m1: 5, m2: 0.3, mClub: 0.2, L1: 0.65, L2: 1.1,
    g: 9.81, b1: 0.1, b2: 0.05, mu1: 0.02, mu2: 0.01,
};

function scenario(objective: ForceSourceObjective, index: number): ForceSourceScenario {
    const time = [0, 0.05, 0.1];
    const zeroes = [0, 0, 0];
    const series = {
        time_s: time,
        arm_angle_rad: [-2.2, -1.1 + index * 0.08, -0.35 + index * 0.04],
        wrist_cock_rad: [-1.57, -0.7 + index * 0.06, 0.15 + index * 0.03],
        arm_angular_velocity_rad_s: zeroes,
        wrist_angular_velocity_rad_s: zeroes,
        shoulder_torque_nm: zeroes,
        wrist_torque_nm: zeroes,
        clubhead_speed_m_s: [0, 10 + index, 20 + index],
        coriolis_tangent_force_n: zeroes,
        coriolis_power_w: zeroes,
        squared_speed_tangent_force_n: zeroes,
        squared_speed_power_w: zeroes,
        shoulder_actuator_power_w: zeroes,
        wrist_actuator_power_w: zeroes,
        total_actuator_power_w: zeroes,
        cumulative_positive_actuator_work_j: zeroes,
        cumulative_net_actuator_work_j: zeroes,
    };
    const candidate = {
        basis: 'bernstein_6' as const,
        profile_duration_s: 0.5,
        shoulder_coefficients_nm: [0, 20, 50, 80 + index, 70, 30, 0] as [number, number, number, number, number, number, number],
        wrist_coefficients_nm: [0, -10, -8, 0, 12, 20 + index, 0] as [number, number, number, number, number, number, number],
    };
    return {
        objective,
        comparison_contract_id: 'force-source-search/v1-test',
        score: index + 1,
        candidate,
        profile_id: candidateProfileId(candidate),
        effort: actuatorEffortMetrics(series),
        impact_time_s: 0.1,
        robustness: {
            sample_count: 9,
            qualified_count: 9,
            qualification_rate: 1,
            median_score: index + 1,
            worst_score: index,
            best_score: index + 2,
            score_spread: 2,
        },
        near_optimal_count: 1,
        boundary_hits: [],
        convergence: [index + 1],
        series,
    };
}

const scenarios = FORCE_SOURCE_OBJECTIVES.map(scenario);
const constraints = { ...DEFAULT_OPTIMIZATION_CONSTRAINTS, studyMode: 'common_bounds' as const };

function svgTags(html: string): string[] {
    return html.match(/<svg\b[^>]*data-animation-frame[^>]*>/g) ?? [];
}

describe('force-source comparison rendering', () => {
    it('renders all six fixed-hub animations in an identical registered frame', () => {
        for (const time of [-1, 0, 0.037, 0.1, 1]) {
            const html = renderToStaticMarkup(
                <ForceSourceResults scenarios={scenarios} time={time} params={params} alignment="fixed_hub" constraints={{ ...constraints, targetClubheadSpeedMps: 50 }} />,
            );
            const frames = svgTags(html);

            expect(frames).toHaveLength(FORCE_SOURCE_OBJECTIVES.length);
            for (const frame of frames) {
                expect(frame).toContain('viewBox="0 0 192 176"');
                expect(frame).toContain('width="192"');
                expect(frame).toContain('height="176"');
                expect(frame).toContain('preserveAspectRatio="xMidYMid meet"');
                expect(frame).toContain('data-alignment="fixed_hub"');
                expect(frame).toContain('data-hub-x="96"');
                expect(frame).toContain('data-hub-y="88"');
                expect(frame).toContain('data-start-arm="-2.2"');
                expect(frame).toContain('data-start-wrist="-1.57"');
            }
            expect(html.match(/class="force-source-animation-stage"/g)).toHaveLength(6);
            const hubs = html.match(/<circle data-role="hub"[^>]*>/g) ?? [];
            expect(hubs).toHaveLength(6);
            for (const hub of hubs) {
                expect(hub).toContain('cx="96"');
                expect(hub).toContain('cy="88"');
                expect(hub).toContain('aria-label="Fixed shoulder hub"');
            }
            expect(html.match(/data-role="wrist-joint"/g)).toHaveLength(6);
            expect(html.match(/data-role="clubhead"/g)).toHaveLength(6);
            expect(html).toContain('aria-label="Animation marker key"');
            expect(html).not.toContain('data-role="reference-line"');
            expect(html).not.toContain('data-role="comparison-target"');
            expect(html).not.toContain('data-role="impact-location"');
            expect(html).not.toContain('stroke-dasharray');
        }
    });

    it('keeps the target registered in impact-aligned mode for every objective', () => {
        const html = renderToStaticMarkup(
            <ForceSourceResults scenarios={scenarios} time={0.1} params={params} alignment="impact_aligned" constraints={{ ...constraints, targetClubheadSpeedMps: 50 }} />,
        );

        expect(html.match(/data-alignment="impact_aligned"/g)).toHaveLength(6);
        expect(html.match(/data-impact-x="150"/g)).toHaveLength(6);
        expect(html.match(/data-impact-y="148"/g)).toHaveLength(6);
        expect(html.match(/data-role="camera-impact-target"/g)).toHaveLength(6);
        expect(html).not.toContain('data-role="reference-line"');
    });

    it('shows the full cross-objective matrix, strategy diagnostics, coefficients, and every series', () => {
        const html = renderToStaticMarkup(
            <ForceSourceResults scenarios={scenarios} time={0.1} params={params} alignment="fixed_hub" constraints={constraints} />,
        );

        expect(html).toContain('Cross-objective tradeoffs and ranks');
        expect(html).toContain('Input work, activation, and control strategy');
        expect(html).toContain('Input Pareto');
        expect(html).toContain('Sixth-order polynomial coefficients');
        for (const field of [
            'clubhead_speed_m_s', 'shoulder_torque_nm', 'wrist_torque_nm',
            'arm_angle_rad', 'wrist_cock_rad', 'arm_angular_velocity_rad_s',
            'wrist_angular_velocity_rad_s', 'coriolis_tangent_force_n',
            'coriolis_power_w', 'squared_speed_tangent_force_n',
            'squared_speed_power_w', 'hand_path_tangent_force_n',
            'shoulder_actuator_power_w', 'wrist_actuator_power_w',
            'total_actuator_power_w', 'cumulative_positive_actuator_work_j',
            'cumulative_net_actuator_work_j',
        ]) {
            expect(html).toContain(`data-testid="force-source-plot-${field}"`);
        }
    });

    it('assigns equal rank to strategies with identical measured outcomes', () => {
        const tied = scenarios.map(item => ({
            ...item,
            series: structuredClone(scenarios[0].series),
        }));
        const html = renderToStaticMarkup(
            <ForceSourceResults scenarios={tied} time={0.1} params={params} alignment="fixed_hub" constraints={constraints} />,
        );

        expect(html.match(/<small>#1<\/small>/g)).toHaveLength(36);
        expect(html).not.toMatch(/<small>#[2-6]<\/small>/);
    });
});
