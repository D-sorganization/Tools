import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ForceSourceResults } from '../src/components/ForceSourceResults';
import {
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
    return {
        objective,
        score: index + 1,
        candidate: {
            shoulder_torque_nm: 80 + index,
            wrist_drive_nm: 20 + index,
            wrist_restrain_nm: 10 + index,
            onset_s: 0.05,
        },
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
        series: {
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
        },
    };
}

const scenarios = FORCE_SOURCE_OBJECTIVES.map(scenario);

function svgTags(html: string): string[] {
    return html.match(/<svg\b[^>]*data-animation-frame[^>]*>/g) ?? [];
}

describe('force-source comparison rendering', () => {
    it('renders all six fixed-hub animations in an identical registered frame', () => {
        for (const time of [-1, 0, 0.037, 0.1, 1]) {
            const html = renderToStaticMarkup(
                <ForceSourceResults scenarios={scenarios} time={time} params={params} alignment="fixed_hub" />,
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
                expect(frame).toContain('data-reference-y="148"');
            }
            expect(html.match(/class="force-source-animation-stage"/g)).toHaveLength(6);
            const hubs = html.match(/<circle data-role="hub"[^>]*>/g) ?? [];
            expect(hubs).toHaveLength(6);
            for (const hub of hubs) {
                expect(hub).toContain('cx="96"');
                expect(hub).toContain('cy="88"');
            }
            const references = html.match(/<line data-role="reference-line"[^>]*>/g) ?? [];
            expect(references).toHaveLength(6);
            for (const reference of references) {
                expect(reference).toContain('y1="148"');
                expect(reference).toContain('y2="148"');
            }
        }
    });

    it('keeps the target registered in impact-aligned mode for every objective', () => {
        const html = renderToStaticMarkup(
            <ForceSourceResults scenarios={scenarios} time={0.1} params={params} alignment="impact_aligned" />,
        );

        expect(html.match(/data-alignment="impact_aligned"/g)).toHaveLength(6);
        expect(html.match(/data-impact-x="150"/g)).toHaveLength(6);
        expect(html.match(/data-impact-y="148"/g)).toHaveLength(6);
    });
});
