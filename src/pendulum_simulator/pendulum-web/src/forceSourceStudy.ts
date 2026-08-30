import {
    generalizedForceSources,
    jointVelocities,
    runSimulation,
    type PendulumParams,
    type State,
    type TorqueFunc,
} from './physics';

export const FORCE_SOURCE_SCHEMA = 'force-source-comparison/v1' as const;

export type ForceSourceObjective =
    | 'coriolis_impulse'
    | 'coriolis_energy_transfer'
    | 'centrifugal_impulse'
    | 'centrifugal_energy_transfer'
    | 'clubhead_speed';

export const FORCE_SOURCE_OBJECTIVES: readonly ForceSourceObjective[] = [
    'coriolis_impulse',
    'coriolis_energy_transfer',
    'centrifugal_impulse',
    'centrifugal_energy_transfer',
    'clubhead_speed',
];

export const OBJECTIVE_LABELS: Record<ForceSourceObjective, string> = {
    coriolis_impulse: 'Coriolis impulse',
    coriolis_energy_transfer: 'Coriolis energy transfer',
    centrifugal_impulse: 'Squared-speed impulse',
    centrifugal_energy_transfer: 'Squared-speed energy transfer',
    clubhead_speed: 'Clubhead speed',
};

export interface ForceSourceCandidate {
    shoulder_torque_nm: number;
    wrist_drive_nm: number;
    wrist_restrain_nm: number;
    onset_s: number;
}

export interface RobustnessSummary {
    sample_count: number;
    qualified_count: number;
    qualification_rate: number;
    median_score: number;
    worst_score: number;
    best_score: number;
    score_spread: number;
}

export interface ForceSourceSeries {
    time_s: number[];
    arm_angle_rad: number[];
    wrist_cock_rad: number[];
    arm_angular_velocity_rad_s: number[];
    wrist_angular_velocity_rad_s: number[];
    shoulder_torque_nm: number[];
    wrist_torque_nm: number[];
    clubhead_speed_m_s: number[];
    coriolis_tangent_force_n: number[];
    coriolis_power_w: number[];
    squared_speed_tangent_force_n: number[];
    squared_speed_power_w: number[];
}

export interface ForceSourceScenario {
    objective: ForceSourceObjective;
    score: number;
    candidate: ForceSourceCandidate;
    impact_time_s: number;
    impact_diagnostics?: {
        path_angle_deg: number;
        bottom_reach_fraction: number;
        x_velocity_m_s: number;
        y_velocity_m_s: number;
        arm_angle_deg: number;
        club_angle_deg: number;
    };
    robustness: RobustnessSummary;
    near_optimal_count: number;
    boundary_hits: string[];
    convergence: number[];
    series: ForceSourceSeries;
}

export interface ForceSourceArtifact {
    schema_version: typeof FORCE_SOURCE_SCHEMA;
    force_attribution_schema: string;
    model: string;
    coordinates: string[];
    initial_pose: {
        arm_angle_rad: number;
        wrist_cock_rad: number;
        arm_velocity_rad_s: number;
        wrist_velocity_rad_s: number;
    };
    parameters?: PendulumParams;
    search_profile: Record<string, unknown>;
    evaluated_count: number;
    qualified_count: number;
    qualification_counts?: Record<string, number>;
    scenarios: ForceSourceScenario[];
    interpretation_limits: string[];
    provenance?: Record<string, unknown>;
}

const SERIES_KEYS = [
    'time_s',
    'arm_angle_rad',
    'wrist_cock_rad',
    'arm_angular_velocity_rad_s',
    'wrist_angular_velocity_rad_s',
    'shoulder_torque_nm',
    'wrist_torque_nm',
    'clubhead_speed_m_s',
    'coriolis_tangent_force_n',
    'coriolis_power_w',
    'squared_speed_tangent_force_n',
    'squared_speed_power_w',
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireFinite(value: unknown, path: string): number {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new TypeError(`${path} must be a finite number`);
    }
    return value;
}

function requireFiniteArray(value: unknown, path: string): number[] {
    if (!Array.isArray(value) || value.length < 2) {
        throw new TypeError(`${path} must contain at least two samples`);
    }
    return value.map((entry, index) => requireFinite(entry, `${path}[${index}]`));
}

export function parseForceSourceArtifact(input: unknown): ForceSourceArtifact {
    if (!isRecord(input) || input.schema_version !== FORCE_SOURCE_SCHEMA) {
        throw new TypeError(`Expected ${FORCE_SOURCE_SCHEMA}`);
    }
    if (!Array.isArray(input.scenarios) || input.scenarios.length === 0) {
        throw new TypeError('scenarios must be a non-empty array');
    }
    for (const [scenarioIndex, rawScenario] of input.scenarios.entries()) {
        if (!isRecord(rawScenario) || !FORCE_SOURCE_OBJECTIVES.includes(rawScenario.objective as ForceSourceObjective)) {
            throw new TypeError(`scenarios[${scenarioIndex}].objective is unsupported`);
        }
        if (!isRecord(rawScenario.candidate) || !isRecord(rawScenario.series)) {
            throw new TypeError(`scenarios[${scenarioIndex}] is missing candidate or series`);
        }
        const rawSeries = rawScenario.series;
        requireFinite(rawScenario.score, `scenarios[${scenarioIndex}].score`);
        requireFinite(rawScenario.impact_time_s, `scenarios[${scenarioIndex}].impact_time_s`);
        if (rawScenario.impact_diagnostics !== undefined) {
            if (!isRecord(rawScenario.impact_diagnostics)) {
                throw new TypeError(`scenarios[${scenarioIndex}].impact_diagnostics must be an object`);
            }
            const diagnostics = rawScenario.impact_diagnostics;
            const pathAngle = requireFinite(diagnostics.path_angle_deg, `scenarios[${scenarioIndex}].impact_diagnostics.path_angle_deg`);
            const bottomReach = requireFinite(diagnostics.bottom_reach_fraction, `scenarios[${scenarioIndex}].impact_diagnostics.bottom_reach_fraction`);
            const xVelocity = requireFinite(diagnostics.x_velocity_m_s, `scenarios[${scenarioIndex}].impact_diagnostics.x_velocity_m_s`);
            if (pathAngle > 15 || bottomReach < 0.9 || xVelocity <= 0) {
                throw new RangeError(`scenarios[${scenarioIndex}] does not satisfy golf-like impact qualification`);
            }
        }
        const drive = requireFinite(rawScenario.candidate.wrist_drive_nm, `scenarios[${scenarioIndex}].candidate.wrist_drive_nm`);
        const restrain = requireFinite(rawScenario.candidate.wrist_restrain_nm, `scenarios[${scenarioIndex}].candidate.wrist_restrain_nm`);
        if (drive < 0 || drive > 30 || restrain < 0 || restrain > 30) {
            throw new RangeError(`scenarios[${scenarioIndex}] wrist torque exceeds [0, 30] N m`);
        }
        const lengths = SERIES_KEYS.map(key => requireFiniteArray(rawSeries[key], `scenarios[${scenarioIndex}].series.${key}`).length);
        if (!lengths.every(length => length === lengths[0])) {
            throw new TypeError(`scenarios[${scenarioIndex}] series lengths must match`);
        }
    }
    return input as unknown as ForceSourceArtifact;
}

export type SearchThoroughness = 'quick' | 'thorough';

export interface BrowserOptimizationConfig {
    params: PendulumParams;
    initialState: State;
    objective: ForceSourceObjective;
    wristTorqueLimitNm: number;
    thoroughness: SearchThoroughness;
}

export interface BrowserOptimizationProgress {
    completed: number;
    total: number;
    bestScore: number;
}

interface EvaluatedCandidate {
    candidate: ForceSourceCandidate;
    score: number;
    impactIndex: number;
    scenario: ForceSourceScenario;
}

function range(start: number, end: number, step: number): number[] {
    const result: number[] = [];
    for (let value = start; value <= end + step / 10; value += step) {
        result.push(+value.toFixed(6));
    }
    return result;
}

export function candidateGrid(
    thoroughness: SearchThoroughness,
    wristTorqueLimitNm: number,
): ForceSourceCandidate[] {
    if (!(wristTorqueLimitNm > 0 && wristTorqueLimitNm <= 30)) {
        throw new RangeError('wristTorqueLimitNm must be in (0, 30]');
    }
    const shoulder = [60, 80, 100];
    const wrist = thoroughness === 'quick'
        ? [...new Set([0, wristTorqueLimitNm / 2, wristTorqueLimitNm])]
        : [...new Set([...range(0, wristTorqueLimitNm, 5), wristTorqueLimitNm])]
            .sort((left, right) => left - right);
    const onset = thoroughness === 'quick' ? [0.05, 0.10, 0.15, 0.20] : range(0, 0.30, 0.025);
    const candidates: ForceSourceCandidate[] = [];
    for (const shoulderTorque of shoulder) {
        for (const drive of wrist) {
            for (const restrain of wrist) {
                for (const onsetTime of onset) {
                    candidates.push({
                        shoulder_torque_nm: shoulderTorque,
                        wrist_drive_nm: drive,
                        wrist_restrain_nm: restrain,
                        onset_s: onsetTime,
                    });
                }
            }
        }
    }
    return candidates;
}

function trapezoid(values: number[], time: number[]): number {
    let sum = 0;
    for (let index = 1; index < values.length; index++) {
        sum += 0.5 * (values[index - 1] + values[index]) * (time[index] - time[index - 1]);
    }
    return sum;
}

const degrees = (value: number) => value * Math.PI / 180;

export function golfLikeImpactIndex(states: State[], params: PendulumParams): number | null {
    if (states.length < 2) return null;
    const initialArm = states[0][0];
    const initialClub = states[0][0] + states[0][1];
    const target = 2 * Math.PI * Math.ceil(initialClub / (2 * Math.PI));
    let impactIndex: number | null = null;
    for (let index = 1; index < states.length; index++) {
        const previous = states[index - 1][0] + states[index - 1][1];
        const current = states[index][0] + states[index][1];
        if (previous < target && current >= target) {
            impactIndex = index;
            break;
        }
    }
    if (impactIndex === null) return null;
    const path = states.slice(0, impactIndex + 1);
    const arms = path.map(state => state[0]);
    const wrists = path.map(state => state[1]);
    const clubs = path.map(state => state[0] + state[1]);
    const span = (values: number[]) => Math.max(...values) - Math.min(...values);
    if (
        Math.min(...arms) < initialArm - degrees(15)
        || span(arms) > Math.PI
        || Math.min(...clubs) < initialClub - degrees(45)
        || span(clubs) > 1.5 * Math.PI
        || Math.min(...wrists) < degrees(-135)
        || Math.max(...wrists) > degrees(45)
    ) return null;
    const impact = states[impactIndex];
    const clubAngle = impact[0] + impact[1];
    const clubAngularVelocity = impact[2] + impact[3];
    const xVelocity = params.L1 * Math.cos(impact[0]) * impact[2]
        + params.L2 * Math.cos(clubAngle) * clubAngularVelocity;
    const yVelocity = params.L1 * Math.sin(impact[0]) * impact[2]
        + params.L2 * Math.sin(clubAngle) * clubAngularVelocity;
    const pathAngle = Math.atan2(Math.abs(yVelocity), Math.abs(xVelocity));
    const clubheadY = -params.L1 * Math.cos(impact[0]) - params.L2 * Math.cos(clubAngle);
    if (
        xVelocity <= 0
        || pathAngle > degrees(15)
        || clubheadY > -0.9 * (params.L1 + params.L2)
    ) return null;
    return impactIndex;
}

function evaluateCandidate(
    config: BrowserOptimizationConfig,
    candidate: ForceSourceCandidate,
): EvaluatedCandidate | null {
    const torque: TorqueFunc = time => [
        candidate.shoulder_torque_nm,
        time < candidate.onset_s ? -candidate.wrist_restrain_nm : candidate.wrist_drive_nm,
    ];
    const dt = config.thoroughness === 'thorough' ? 0.002 : 0.003;
    const simulation = runSimulation(config.params, config.initialState, 0.9, torque, dt);
    const impactIndex = golfLikeImpactIndex(simulation.states, config.params);
    if (impactIndex === null) return null;
    const series: ForceSourceSeries = {
        time_s: [], arm_angle_rad: [], wrist_cock_rad: [],
        arm_angular_velocity_rad_s: [], wrist_angular_velocity_rad_s: [],
        shoulder_torque_nm: [], wrist_torque_nm: [], clubhead_speed_m_s: [],
        coriolis_tangent_force_n: [], coriolis_power_w: [],
        squared_speed_tangent_force_n: [], squared_speed_power_w: [],
    };
    for (let index = 0; index <= impactIndex; index++) {
        const time = simulation.t[index];
        const state = simulation.states[index];
        const control = torque(time);
        const sources = generalizedForceSources(state, config.params, control);
        series.time_s.push(time);
        series.arm_angle_rad.push(state[0]);
        series.wrist_cock_rad.push(state[1]);
        series.arm_angular_velocity_rad_s.push(state[2]);
        series.wrist_angular_velocity_rad_s.push(state[3]);
        series.shoulder_torque_nm.push(control[0]);
        series.wrist_torque_nm.push(control[1]);
        series.clubhead_speed_m_s.push(jointVelocities(state, config.params).tipSpeed);
        series.coriolis_tangent_force_n.push(sources.coriolis[0] / config.params.L1);
        series.coriolis_power_w.push(sources.coriolis[0] * state[2] + sources.coriolis[1] * state[3]);
        series.squared_speed_tangent_force_n.push(sources.squaredSpeed[0] / config.params.L1);
        series.squared_speed_power_w.push(sources.squaredSpeed[0] * state[2] + sources.squaredSpeed[1] * state[3]);
    }
    const objectiveScores: Record<ForceSourceObjective, number> = {
        coriolis_impulse: trapezoid(series.coriolis_tangent_force_n.map(Math.abs), series.time_s),
        coriolis_energy_transfer: trapezoid(series.coriolis_power_w, series.time_s),
        centrifugal_impulse: trapezoid(series.squared_speed_tangent_force_n.map(Math.abs), series.time_s),
        centrifugal_energy_transfer: trapezoid(series.squared_speed_power_w, series.time_s),
        clubhead_speed: series.clubhead_speed_m_s[series.clubhead_speed_m_s.length - 1],
    };
    const score = objectiveScores[config.objective];
    const robustness: RobustnessSummary = {
        sample_count: 1, qualified_count: 1, qualification_rate: 1,
        median_score: score, worst_score: score, best_score: score, score_spread: 0,
    };
    const impact = simulation.states[impactIndex];
    const clubAngle = impact[0] + impact[1];
    const clubAngularVelocity = impact[2] + impact[3];
    const xVelocity = config.params.L1 * Math.cos(impact[0]) * impact[2]
        + config.params.L2 * Math.cos(clubAngle) * clubAngularVelocity;
    const yVelocity = config.params.L1 * Math.sin(impact[0]) * impact[2]
        + config.params.L2 * Math.sin(clubAngle) * clubAngularVelocity;
    const clubheadY = -config.params.L1 * Math.cos(impact[0])
        - config.params.L2 * Math.cos(clubAngle);
    return {
        candidate, score, impactIndex,
        scenario: {
            objective: config.objective,
            score,
            candidate,
            impact_time_s: series.time_s[series.time_s.length - 1],
            impact_diagnostics: {
                path_angle_deg: Math.atan2(Math.abs(yVelocity), Math.abs(xVelocity)) * 180 / Math.PI,
                bottom_reach_fraction: -clubheadY / (config.params.L1 + config.params.L2),
                x_velocity_m_s: xVelocity,
                y_velocity_m_s: yVelocity,
                arm_angle_deg: impact[0] * 180 / Math.PI,
                club_angle_deg: clubAngle * 180 / Math.PI,
            },
            robustness,
            near_optimal_count: 1,
            boundary_hits: [],
            convergence: [score],
            series,
        },
    };
}

function refinementNeighbors(
    candidate: ForceSourceCandidate,
    config: BrowserOptimizationConfig,
): ForceSourceCandidate[] {
    const shoulderStep = config.thoroughness === 'thorough' ? 5 : 10;
    const wristStep = config.thoroughness === 'thorough' ? 1 : 2.5;
    const onsetStep = config.thoroughness === 'thorough' ? 0.005 : 0.01;
    const variants = new Map<string, ForceSourceCandidate>();
    const add = (value: ForceSourceCandidate) => variants.set(JSON.stringify(value), value);
    add(candidate);
    const fields: Array<[keyof ForceSourceCandidate, number, number, number]> = [
        ['shoulder_torque_nm', shoulderStep, 60, 100],
        ['wrist_drive_nm', wristStep, 0, config.wristTorqueLimitNm],
        ['wrist_restrain_nm', wristStep, 0, config.wristTorqueLimitNm],
        ['onset_s', onsetStep, 0, 0.30],
    ];
    for (const [field, step, lower, upper] of fields) {
        for (const direction of [-1, 1]) {
            add({ ...candidate, [field]: Math.min(upper, Math.max(lower, candidate[field] + direction * step)) });
        }
    }
    return [...variants.values()];
}

const yieldToBrowser = () => new Promise<void>(resolve => setTimeout(resolve, 0));

export async function optimizeForceSource(
    config: BrowserOptimizationConfig,
    onProgress?: (progress: BrowserOptimizationProgress) => void,
): Promise<ForceSourceScenario> {
    if (!FORCE_SOURCE_OBJECTIVES.includes(config.objective)) {
        throw new RangeError('objective is unsupported');
    }
    const candidates = candidateGrid(config.thoroughness, config.wristTorqueLimitNm);
    let best: EvaluatedCandidate | null = null;
    let completed = 0;
    const total = candidates.length + (config.thoroughness === 'thorough' ? 36 : 18);
    for (const candidate of candidates) {
        const evaluated = evaluateCandidate(config, candidate);
        if (evaluated && (!best || evaluated.score > best.score)) best = evaluated;
        completed += 1;
        if (completed % 24 === 0) {
            onProgress?.({ completed, total, bestScore: best?.score ?? Number.NEGATIVE_INFINITY });
            await yieldToBrowser();
        }
    }
    if (!best) throw new Error('No candidate reached the qualified impact event');
    const rounds = config.thoroughness === 'thorough' ? 4 : 2;
    const convergence = [best.score];
    for (let round = 0; round < rounds; round++) {
        for (const candidate of refinementNeighbors(best.candidate, config)) {
            const evaluated = evaluateCandidate(config, candidate);
            if (evaluated && evaluated.score > best.score) best = evaluated;
            completed += 1;
        }
        convergence.push(best.score);
        onProgress?.({ completed, total, bestScore: best.score });
        await yieldToBrowser();
    }
    best.scenario.convergence = convergence;
    return best.scenario;
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
        search_profile: { source: 'browser optimization' },
        evaluated_count: (existing?.evaluated_count ?? 0) + 1,
        qualified_count: (existing?.qualified_count ?? 0) + 1,
        scenarios,
        interpretation_limits: existing?.interpretation_limits ?? [
            'Force-source terms depend on the declared coordinates.',
            'This synthetic planar model is exploratory, not individualized swing advice.',
        ],
    };
}
