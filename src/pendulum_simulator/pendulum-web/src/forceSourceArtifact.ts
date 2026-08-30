import type { PendulumParams, State } from './physics';
import {
    buildOptimizationContract,
    scoreForceSourceSeries,
} from './forceSourceOptimization';
import {
    FORCE_SOURCE_OBJECTIVES,
    FORCE_SOURCE_SCHEMA,
    type BrowserOptimizationConfig,
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
    'hand_path_tangent_force_n',
] as const;

const PARAMETER_KEYS = [
    'm1', 'm2', 'mClub', 'L1', 'L2', 'g', 'b1', 'b2', 'mu1', 'mu2',
] as const;
const START_TOLERANCE = 1e-10;
const SCORE_TOLERANCE = 1e-8;

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
    if (!lengths.every(length => length === lengths[0])) {
        throw new TypeError(`${path} series lengths must match`);
    }
}

function registeredConfig(input: Record<string, unknown>): BrowserOptimizationConfig {
    if (!isRecord(input.initial_pose) || !isRecord(input.parameters)
        || !isRecord(input.comparison_contract)) {
        throw new TypeError('artifact is missing initial_pose, parameters, or comparison_contract');
    }
    const pose = input.initial_pose;
    const parameters = input.parameters;
    const contract = input.comparison_contract;
    const params = Object.fromEntries(PARAMETER_KEYS.map(key => [
        key, finite(parameters[key], `parameters.${key}`),
    ])) as unknown as PendulumParams;
    const initialState: State = [
        finite(pose.arm_angle_rad, 'initial_pose.arm_angle_rad'),
        finite(pose.wrist_cock_rad, 'initial_pose.wrist_cock_rad'),
        finite(pose.arm_velocity_rad_s, 'initial_pose.arm_velocity_rad_s'),
        finite(pose.wrist_velocity_rad_s, 'initial_pose.wrist_velocity_rad_s'),
    ];
    if (!isRecord(contract.constraints)
        || !['quick', 'thorough', 'research'].includes(String(contract.thoroughness))) {
        throw new TypeError('comparison_contract is invalid');
    }
    return {
        params,
        initialState,
        objective: 'clubhead_speed',
        thoroughness: contract.thoroughness as BrowserOptimizationConfig['thoroughness'],
        constraints: contract.constraints as unknown as BrowserOptimizationConfig['constraints'],
    };
}

function validateSharedContract(
    input: Record<string, unknown>,
    scenarios: ForceSourceScenario[],
): void {
    const config = registeredConfig(input);
    const expected = buildOptimizationContract(config);
    const declared = input.comparison_contract as Record<string, unknown>;
    if (declared.id !== expected.id) {
        throw new TypeError('comparison_contract id does not match its registered settings');
    }
    for (const scenario of scenarios) {
        if (scenario.comparison_contract_id !== expected.id) {
            throw new TypeError(`${scenario.objective} does not share the registered comparison contract`);
        }
        const start = [
            scenario.series.arm_angle_rad[0],
            scenario.series.wrist_cock_rad[0],
            scenario.series.arm_angular_velocity_rad_s[0],
            scenario.series.wrist_angular_velocity_rad_s[0],
        ];
        if (Math.abs(scenario.series.time_s[0]) > START_TOLERANCE
            || start.some((value, index) => Math.abs(value - config.initialState[index]) > START_TOLERANCE)) {
            throw new TypeError(`${scenario.objective} does not use the shared initial pose`);
        }
    }
}

function validateObjectiveDominance(scenarios: ForceSourceScenario[]): void {
    const scores = scenarios.map(scenario => scoreForceSourceSeries(scenario.series));
    for (const [index, scenario] of scenarios.entries()) {
        const ownScore = scores[index][scenario.objective];
        const tolerance = SCORE_TOLERANCE * Math.max(1, Math.abs(ownScore));
        if (Math.abs(ownScore - scenario.score) > tolerance) {
            throw new TypeError(`${scenario.objective} score does not match its registered series`);
        }
        const displayedBest = Math.max(...scores.map(score => score[scenario.objective]));
        if (ownScore < displayedBest - tolerance) {
            throw new RangeError(`${scenario.objective} loses its objective to another displayed scenario`);
        }
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
    const scenarios = input.scenarios as unknown as ForceSourceScenario[];
    if (new Set(scenarios.map(item => item.objective)).size !== scenarios.length) {
        throw new TypeError('scenario objectives must be unique');
    }
    validateSharedContract(input, scenarios);
    validateObjectiveDominance(scenarios);
    return input as unknown as ForceSourceArtifact;
}

export function artifactWithScenarios(
    existing: ForceSourceArtifact | null,
    incoming: readonly ForceSourceScenario[],
    config: BrowserOptimizationConfig,
): ForceSourceArtifact {
    const contract = buildOptimizationContract(config);
    const retained = existing?.comparison_contract.id === contract.id
        ? existing.scenarios.filter(item => !incoming.some(next => next.objective === item.objective))
        : [];
    const scenarios = [...retained, ...incoming].map(scenario => ({
        ...scenario,
        comparison_contract_id: contract.id,
    }));
    const artifact: ForceSourceArtifact = {
        schema_version: FORCE_SOURCE_SCHEMA,
        force_attribution_schema: 'force-attribution/v1',
        model: 'Tools pendulum-web shared planar model',
        coordinates: ['shoulder_absolute', 'wrist_relative'],
        initial_pose: {
            arm_angle_rad: config.initialState[0], wrist_cock_rad: config.initialState[1],
            arm_velocity_rad_s: config.initialState[2], wrist_velocity_rad_s: config.initialState[3],
        },
        parameters: config.params,
        search_profile: {
            source: 'browser optimization',
            thoroughness: config.thoroughness,
            candidate_budget: config.constraints.candidateBudget,
            integration_step_s: config.constraints.integrationStepS,
        },
        comparison_contract: contract,
        evaluated_count: scenarios.reduce((sum, scenario) =>
            sum + Number(scenario.provenance?.global_candidate_count ?? 1), 0),
        qualified_count: scenarios.reduce((sum, scenario) =>
            sum + Number(scenario.provenance?.qualified_evaluation_count ?? 1), 0),
        scenarios,
        interpretation_limits: existing?.comparison_contract.id === contract.id
            ? existing.interpretation_limits : [
            'Force-source terms depend on the declared coordinates.',
            'Hand-path impulse is signed force integrated over time; it is not hand-path work or average force over distance.',
            'This synthetic planar model is exploratory, not individualized swing advice.',
        ],
    };
    return parseForceSourceArtifact(artifact);
}
