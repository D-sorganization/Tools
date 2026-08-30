import type { PendulumParams, State } from './physics';
import {
    FORCE_SOURCE_OBJECTIVES,
    FORCE_SOURCE_SCHEMA,
    type ForceSourceArtifact,
    type ForceSourceObjective,
    type ForceSourceScenario,
} from './forceSourceTypes';

const SERIES_KEYS = [
    'time_s', 'arm_angle_rad', 'wrist_cock_rad',
    'arm_angular_velocity_rad_s', 'wrist_angular_velocity_rad_s',
    'shoulder_torque_nm', 'wrist_torque_nm', 'clubhead_speed_m_s',
    'coriolis_tangent_force_n', 'coriolis_power_w',
    'squared_speed_tangent_force_n', 'squared_speed_power_w',
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finite(value: unknown, path: string): number {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new TypeError(`${path} must be a finite number`);
    }
    return value;
}

function finiteSeries(value: unknown, path: string): number[] {
    if (!Array.isArray(value) || value.length < 2) {
        throw new TypeError(`${path} must contain at least two samples`);
    }
    return value.map((entry, index) => finite(entry, `${path}[${index}]`));
}

function validateImpact(value: unknown, path: string): void {
    if (value === undefined) return;
    if (!isRecord(value)) throw new TypeError(`${path} must be an object`);
    const pathAngle = finite(value.path_angle_deg, `${path}.path_angle_deg`);
    const reach = finite(value.bottom_reach_fraction, `${path}.bottom_reach_fraction`);
    const xVelocity = finite(value.x_velocity_m_s, `${path}.x_velocity_m_s`);
    finite(value.y_velocity_m_s, `${path}.y_velocity_m_s`);
    finite(value.arm_angle_deg, `${path}.arm_angle_deg`);
    finite(value.club_angle_deg, `${path}.club_angle_deg`);
    if (pathAngle > 15 || reach < 0.9 || xVelocity <= 0) {
        throw new RangeError(`${path} does not satisfy registered golf-like impact qualification`);
    }
}

function validateScenario(raw: unknown, index: number): void {
    const path = `scenarios[${index}]`;
    if (!isRecord(raw) || !FORCE_SOURCE_OBJECTIVES.includes(raw.objective as ForceSourceObjective)) {
        throw new TypeError(`${path}.objective is unsupported`);
    }
    if (!isRecord(raw.candidate) || !isRecord(raw.series)) {
        throw new TypeError(`${path} is missing candidate or series`);
    }
    const series = raw.series;
    finite(raw.score, `${path}.score`);
    finite(raw.impact_time_s, `${path}.impact_time_s`);
    validateImpact(raw.impact_diagnostics, `${path}.impact_diagnostics`);
    finite(raw.candidate.shoulder_torque_nm, `${path}.candidate.shoulder_torque_nm`);
    finite(raw.candidate.onset_s, `${path}.candidate.onset_s`);
    const drive = finite(raw.candidate.wrist_drive_nm, `${path}.candidate.wrist_drive_nm`);
    const restrain = finite(raw.candidate.wrist_restrain_nm, `${path}.candidate.wrist_restrain_nm`);
    if (drive < 0 || drive > 30 || restrain < 0 || restrain > 30) {
        throw new RangeError(`${path} wrist torque exceeds [0, 30] N m`);
    }
    const lengths = SERIES_KEYS.map(key => finiteSeries(series[key], `${path}.series.${key}`).length);
    if (series.hand_path_tangent_force_n !== undefined) {
        lengths.push(finiteSeries(series.hand_path_tangent_force_n, `${path}.series.hand_path_tangent_force_n`).length);
    }
    if (!lengths.every(length => length === lengths[0])) {
        throw new TypeError(`${path} series lengths must match`);
    }
}

export function parseForceSourceArtifact(input: unknown): ForceSourceArtifact {
    if (!isRecord(input) || input.schema_version !== FORCE_SOURCE_SCHEMA) {
        throw new TypeError(`Expected ${FORCE_SOURCE_SCHEMA}`);
    }
    if (!Array.isArray(input.scenarios) || input.scenarios.length === 0) {
        throw new TypeError('scenarios must be a non-empty array');
    }
    input.scenarios.forEach(validateScenario);
    return input as unknown as ForceSourceArtifact;
}

export function artifactWithScenario(
    existing: ForceSourceArtifact | null,
    scenario: ForceSourceScenario,
    initialState: State,
    params: PendulumParams,
): ForceSourceArtifact {
    const scenarios = (existing?.scenarios ?? []).filter(item => item.objective !== scenario.objective);
    scenarios.push(scenario);
    return {
        schema_version: FORCE_SOURCE_SCHEMA,
        force_attribution_schema: 'force-attribution/v1',
        model: 'Tools pendulum-web shared planar model',
        coordinates: ['shoulder_absolute', 'wrist_relative'],
        initial_pose: {
            arm_angle_rad: initialState[0], wrist_cock_rad: initialState[1],
            arm_velocity_rad_s: initialState[2], wrist_velocity_rad_s: initialState[3],
        },
        parameters: params,
        search_profile: { source: 'browser optimization', wrist_limit_nm: 30 },
        evaluated_count: existing?.evaluated_count ?? Number(scenario.provenance?.global_candidate_count ?? 1),
        qualified_count: existing?.qualified_count ?? Number(scenario.provenance?.qualified_evaluation_count ?? 1),
        scenarios,
        interpretation_limits: existing?.interpretation_limits ?? [
            'Force-source terms depend on the declared coordinates.',
            'Hand-path impulse is signed force integrated over time; it is not hand-path work or average force over distance.',
            'This synthetic planar model is exploratory, not individualized swing advice.',
        ],
    };
}
