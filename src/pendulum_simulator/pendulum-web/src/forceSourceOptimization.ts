import {
    computeAccelerations,
    generalizedForceSources,
    gripForceAlongHandPath,
    jointVelocities,
    runSimulation,
    type ForceSourceTerms,
    type PendulumParams,
    type State,
    type TorqueFunc,
} from './physics';
import {
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    FORCE_SOURCE_OBJECTIVES,
    type BrowserOptimizationConfig,
    type BrowserOptimizationProgress,
    type ForceSourceCandidate,
    type ForceSourceComparisonContract,
    type ForceSourceConstraints,
    type ForceSourceObjective,
    type ForceSourceScenario,
    type ForceSourceSeries,
    type NumericRange,
    type RobustnessSummary,
} from './forceSourceTypes';

interface EvaluatedCandidate {
    score: number;
    candidate: ForceSourceCandidate;
    scenario: ForceSourceScenario;
}

const radians = (degrees: number) => degrees * Math.PI / 180;
const degrees = (value: number) => value * 180 / Math.PI;
const OBJECTIVE_TOLERANCE = 1e-10;

function contractChecksum(value: string): string {
    let hash = 0x811c9dc5;
    for (let index = 0; index < value.length; index++) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
}

function contractPayload(config: BrowserOptimizationConfig): unknown[] {
    const c = config.constraints;
    const p = config.params;
    return [
        config.initialState,
        [p.m1, p.m2, p.mClub, p.L1, p.L2, p.g, p.b1, p.b2, p.mu1, p.mu2],
        config.thoroughness,
        [
            c.shoulderTorqueNm.min, c.shoulderTorqueNm.max, c.shoulderTorqueNm.step,
            c.wristTorqueLimitNm, c.wristTorqueStepNm,
            c.onsetS.min, c.onsetS.max, c.onsetS.step,
            c.armAngleDeg.min, c.armAngleDeg.max,
            c.wristAngleDeg.min, c.wristAngleDeg.max,
            c.maxImpactPathAngleDeg, c.minBottomReachFraction,
            c.simulationDurationS, c.integrationStepS,
            c.maxArmTravelDeg, c.maxClubTravelDeg,
            c.candidateBudget, c.robustnessTrials,
            c.posePerturbationDeg, c.torquePerturbationFraction,
        ],
    ];
}

/** Build the immutable contract shared by every scenario in a fair comparison. */
export function buildOptimizationContract(
    config: BrowserOptimizationConfig,
): ForceSourceComparisonContract {
    validateBrowserOptimizationConfig(config);
    return {
        id: `force-source-search/v1-${contractChecksum(JSON.stringify(contractPayload(config)))}`,
        thoroughness: config.thoroughness,
        constraints: structuredClone(config.constraints),
    };
}

function validateRange(range: NumericRange, name: string): void {
    if (![range.min, range.max, range.step].every(Number.isFinite)) {
        throw new TypeError(`${name} values must be finite`);
    }
    if (range.min > range.max || range.step <= 0) {
        throw new RangeError(`${name} requires min <= max and step > 0`);
    }
}

export function validateBrowserOptimizationConfig(config: BrowserOptimizationConfig): void {
    const c = config.constraints;
    if (!FORCE_SOURCE_OBJECTIVES.includes(config.objective)) throw new RangeError('objective is unsupported');
    validateRange(c.shoulderTorqueNm, 'shoulder torque');
    validateRange(c.onsetS, 'release onset');
    if (!(c.wristTorqueLimitNm > 0 && c.wristTorqueLimitNm <= 30)) {
        throw new RangeError('wrist torque limit must be in (0, 30] N m');
    }
    if (!(c.wristTorqueStepNm >= 0.1 && c.wristTorqueStepNm <= c.wristTorqueLimitNm)) {
        throw new RangeError('wrist torque step must be between 0.1 N m and the wrist limit');
    }
    const arm = degrees(config.initialState[0]);
    const wrist = degrees(config.initialState[1]);
    if (c.armAngleDeg.min >= c.armAngleDeg.max || c.wristAngleDeg.min >= c.wristAngleDeg.max) throw new RangeError('joint-angle bounds require min < max');
    if (arm < c.armAngleDeg.min || arm > c.armAngleDeg.max) throw new RangeError('initial arm angle violates its bounds');
    if (wrist < c.wristAngleDeg.min || wrist > c.wristAngleDeg.max) throw new RangeError('initial wrist angle violates its bounds');
    if (!(c.maxImpactPathAngleDeg > 0 && c.maxImpactPathAngleDeg <= 30)) throw new RangeError('impact path angle must be in (0, 30] degrees');
    if (!(c.minBottomReachFraction >= 0.8 && c.minBottomReachFraction <= 1)) throw new RangeError('bottom reach must be in [0.8, 1]');
    if (!(c.integrationStepS > 0 && c.integrationStepS < c.simulationDurationS)) throw new RangeError('integration step must be inside the simulation duration');
    if (!(c.maxArmTravelDeg > 0 && c.maxClubTravelDeg > 0)) throw new RangeError('travel limits must be positive');
    if (!(c.posePerturbationDeg >= 0 && c.torquePerturbationFraction >= 0 && c.torquePerturbationFraction <= 0.25)) throw new RangeError('robustness perturbations are outside supported bounds');
    if (!Number.isInteger(c.candidateBudget) || c.candidateBudget < 8 || c.candidateBudget > 10000) throw new RangeError('candidate budget must be an integer in [8, 10000]');
    if (!Number.isInteger(c.robustnessTrials) || c.robustnessTrials < 1 || c.robustnessTrials > 101) throw new RangeError('robustness trials must be an integer in [1, 101]');
}

function halton(index: number, base: number): number {
    let result = 0;
    let fraction = 1;
    for (let value = index; value > 0; value = Math.floor(value / base)) {
        fraction /= base;
        result += fraction * (value % base);
    }
    return result;
}

function quantized(unit: number, range: NumericRange): number {
    const raw = range.min + unit * (range.max - range.min);
    const steps = Math.round((raw - range.min) / range.step);
    return Math.min(range.max, Math.max(range.min, +(range.min + steps * range.step).toFixed(8)));
}

function sampledCandidate(index: number, constraints: ForceSourceConstraints): ForceSourceCandidate {
    const wristRange = { min: 0, max: constraints.wristTorqueLimitNm, step: constraints.wristTorqueStepNm };
    return {
        shoulder_torque_nm: quantized(halton(index, 2), constraints.shoulderTorqueNm),
        wrist_drive_nm: quantized(halton(index, 3), wristRange),
        wrist_restrain_nm: quantized(halton(index, 5), wristRange),
        onset_s: quantized(halton(index, 7), constraints.onsetS),
    };
}

export function buildCandidateSet(config: BrowserOptimizationConfig): ForceSourceCandidate[] {
    validateBrowserOptimizationConfig(config);
    const c = config.constraints;
    const bounds: ForceSourceCandidate[] = [
        { shoulder_torque_nm: c.shoulderTorqueNm.min, wrist_drive_nm: 0, wrist_restrain_nm: 0, onset_s: c.onsetS.min },
        { shoulder_torque_nm: c.shoulderTorqueNm.max, wrist_drive_nm: c.wristTorqueLimitNm, wrist_restrain_nm: c.wristTorqueLimitNm, onset_s: c.onsetS.max },
    ];
    return Array.from({ length: c.candidateBudget }, (_, index) => bounds[index] ?? sampledCandidate(index - 1, c));
}

function span(values: number[]): number {
    return Math.max(...values) - Math.min(...values);
}

export function golfLikeImpactIndex(
    states: State[],
    params: PendulumParams,
    constraints: ForceSourceConstraints = DEFAULT_OPTIMIZATION_CONSTRAINTS,
): number | null {
    if (states.length < 2) return null;
    const initialArm = states[0][0];
    const initialClub = states[0][0] + states[0][1];
    const target = 2 * Math.PI * Math.ceil(initialClub / (2 * Math.PI));
    const impact = states.findIndex((state, index) => index > 0
        && states[index - 1][0] + states[index - 1][1] < target
        && state[0] + state[1] >= target);
    if (impact < 1) return null;
    const path = states.slice(0, impact + 1);
    const arms = path.map(state => state[0]);
    const wrists = path.map(state => state[1]);
    const clubs = path.map(state => state[0] + state[1]);
    if (Math.min(...arms) < radians(constraints.armAngleDeg.min)
        || Math.max(...arms) > radians(constraints.armAngleDeg.max)
        || Math.min(...arms) < initialArm - radians(15)
        || span(arms) > radians(constraints.maxArmTravelDeg)
        || Math.min(...clubs) < initialClub - radians(45)
        || span(clubs) > radians(constraints.maxClubTravelDeg)
        || Math.min(...wrists) < radians(constraints.wristAngleDeg.min)
        || Math.max(...wrists) > radians(constraints.wristAngleDeg.max)) return null;
    return qualifiesImpact(path[impact], params, constraints) ? impact : null;
}

function impactVelocity(state: State, params: PendulumParams): [number, number] {
    const clubAngle = state[0] + state[1];
    const clubRate = state[2] + state[3];
    return [
        params.L1 * Math.cos(state[0]) * state[2] + params.L2 * Math.cos(clubAngle) * clubRate,
        params.L1 * Math.sin(state[0]) * state[2] + params.L2 * Math.sin(clubAngle) * clubRate,
    ];
}

function qualifiesImpact(state: State, params: PendulumParams, c: ForceSourceConstraints): boolean {
    const [vx, vy] = impactVelocity(state, params);
    const clubAngle = state[0] + state[1];
    const clubheadY = -params.L1 * Math.cos(state[0]) - params.L2 * Math.cos(clubAngle);
    return vx > 0
        && Math.atan2(Math.abs(vy), Math.abs(vx)) <= radians(c.maxImpactPathAngleDeg)
        && -clubheadY / (params.L1 + params.L2) >= c.minBottomReachFraction;
}

function emptySeries(): ForceSourceSeries {
    return {
        time_s: [], arm_angle_rad: [], wrist_cock_rad: [],
        arm_angular_velocity_rad_s: [], wrist_angular_velocity_rad_s: [],
        shoulder_torque_nm: [], wrist_torque_nm: [], clubhead_speed_m_s: [],
        coriolis_tangent_force_n: [], coriolis_power_w: [],
        squared_speed_tangent_force_n: [], squared_speed_power_w: [],
        hand_path_tangent_force_n: [],
    };
}

/** Energy crossing from the proximal arm into the distal coordinate. */
export function forceSourceTransferPowers(
    sources: ForceSourceTerms,
    state: State,
): { coriolis_to_distal_w: number; centrifugal_to_distal_w: number } {
    return {
        coriolis_to_distal_w: -sources.coriolis[0] * state[2],
        centrifugal_to_distal_w: sources.squaredSpeed[1] * state[3],
    };
}

function appendSample(series: ForceSourceSeries, state: State, time: number, config: BrowserOptimizationConfig, torque: TorqueFunc): void {
    const control = torque(time);
    const sources = generalizedForceSources(state, config.params, control);
    const acceleration = computeAccelerations(state, time, config.params, torque);
    const hand = gripForceAlongHandPath(state, acceleration, config.params);
    const transfer = forceSourceTransferPowers(sources, state);
    series.time_s.push(time); series.arm_angle_rad.push(state[0]); series.wrist_cock_rad.push(state[1]);
    series.arm_angular_velocity_rad_s.push(state[2]); series.wrist_angular_velocity_rad_s.push(state[3]);
    series.shoulder_torque_nm.push(control[0]); series.wrist_torque_nm.push(control[1]);
    series.clubhead_speed_m_s.push(jointVelocities(state, config.params).tipSpeed);
    series.coriolis_tangent_force_n.push(sources.coriolis[0] / config.params.L1);
    series.coriolis_power_w.push(transfer.coriolis_to_distal_w);
    series.squared_speed_tangent_force_n.push(sources.squaredSpeed[0] / config.params.L1);
    series.squared_speed_power_w.push(transfer.centrifugal_to_distal_w);
    series.hand_path_tangent_force_n?.push(hand.tangentForceN ?? 0);
}

function trapezoid(values: number[], time: number[]): number {
    return values.slice(1).reduce((sum, value, index) =>
        sum + 0.5 * (values[index] + value) * (time[index + 1] - time[index]), 0);
}

export function scoreForceSourceSeries(
    series: ForceSourceSeries,
): Record<ForceSourceObjective, number> {
    return {
        coriolis_impulse: trapezoid(series.coriolis_tangent_force_n.map(Math.abs), series.time_s),
        coriolis_energy_transfer: trapezoid(series.coriolis_power_w, series.time_s),
        centrifugal_impulse: trapezoid(series.squared_speed_tangent_force_n.map(Math.abs), series.time_s),
        centrifugal_energy_transfer: trapezoid(series.squared_speed_power_w, series.time_s),
        clubhead_speed: series.clubhead_speed_m_s[series.clubhead_speed_m_s.length - 1] ?? 0,
        hand_path_impulse: trapezoid(series.hand_path_tangent_force_n ?? [], series.time_s),
    };
}

function impactDiagnostics(state: State, params: PendulumParams) {
    const [xVelocity, yVelocity] = impactVelocity(state, params);
    const clubAngle = state[0] + state[1];
    const clubheadY = -params.L1 * Math.cos(state[0]) - params.L2 * Math.cos(clubAngle);
    return {
        path_angle_deg: degrees(Math.atan2(Math.abs(yVelocity), Math.abs(xVelocity))),
        bottom_reach_fraction: -clubheadY / (params.L1 + params.L2),
        x_velocity_m_s: xVelocity, y_velocity_m_s: yVelocity,
        arm_angle_deg: degrees(state[0]), club_angle_deg: degrees(clubAngle),
    };
}

function evaluateCandidate(config: BrowserOptimizationConfig, candidate: ForceSourceCandidate): EvaluatedCandidate | null {
    const torque: TorqueFunc = time => [candidate.shoulder_torque_nm,
        time < candidate.onset_s ? -candidate.wrist_restrain_nm : candidate.wrist_drive_nm];
    const c = config.constraints;
    const simulation = runSimulation(config.params, config.initialState, c.simulationDurationS, torque, c.integrationStepS);
    const impactIndex = golfLikeImpactIndex(simulation.states, config.params, c);
    if (impactIndex === null) return null;
    const series = emptySeries();
    for (let index = 0; index <= impactIndex; index++) appendSample(series, simulation.states[index], simulation.t[index], config, torque);
    const score = scoreForceSourceSeries(series)[config.objective];
    const robustness = summarizeRobustness([score]);
    return { score, candidate, scenario: {
        objective: config.objective, score, candidate,
        comparison_contract_id: buildOptimizationContract(config).id,
        impact_time_s: series.time_s[series.time_s.length - 1] ?? 0,
        impact_diagnostics: impactDiagnostics(simulation.states[impactIndex], config.params),
        robustness, near_optimal_count: 1, boundary_hits: [], convergence: [score], series,
    } };
}

export function summarizeRobustness(scores: Array<number | null>): RobustnessSummary {
    const qualified = scores.filter((score): score is number => score !== null).sort((a, b) => a - b);
    const middle = Math.floor(qualified.length / 2);
    const median = qualified.length === 0 ? 0 : qualified.length % 2 ? qualified[middle] : (qualified[middle - 1] + qualified[middle]) / 2;
    const worst = qualified[0] ?? 0;
    const best = qualified[qualified.length - 1] ?? 0;
    return {
        sample_count: scores.length, qualified_count: qualified.length,
        qualification_rate: scores.length ? qualified.length / scores.length : 0,
        median_score: median, worst_score: worst, best_score: best, score_spread: best - worst,
    };
}

function refinementNeighbors(candidate: ForceSourceCandidate, config: BrowserOptimizationConfig): ForceSourceCandidate[] {
    const c = config.constraints;
    const fields: Array<[keyof ForceSourceCandidate, number, number, number]> = [
        ['shoulder_torque_nm', c.shoulderTorqueNm.step, c.shoulderTorqueNm.min, c.shoulderTorqueNm.max],
        ['wrist_drive_nm', c.wristTorqueStepNm, 0, c.wristTorqueLimitNm],
        ['wrist_restrain_nm', c.wristTorqueStepNm, 0, c.wristTorqueLimitNm],
        ['onset_s', c.onsetS.step, c.onsetS.min, c.onsetS.max],
    ];
    return [candidate, ...fields.flatMap(([field, step, min, max]) => [-1, 1].map(direction => ({
        ...candidate, [field]: Math.min(max, Math.max(min, +(candidate[field] + direction * step).toFixed(8))),
    })))];
}

function boundaryHits(candidate: ForceSourceCandidate, constraints: ForceSourceConstraints): string[] {
    const checks: Array<[string, number, number, number]> = [
        ['shoulder_torque_nm', candidate.shoulder_torque_nm, constraints.shoulderTorqueNm.min, constraints.shoulderTorqueNm.max],
        ['wrist_drive_nm', candidate.wrist_drive_nm, 0, constraints.wristTorqueLimitNm],
        ['wrist_restrain_nm', candidate.wrist_restrain_nm, 0, constraints.wristTorqueLimitNm],
        ['onset_s', candidate.onset_s, constraints.onsetS.min, constraints.onsetS.max],
    ];
    return checks.flatMap(([name, value, min, max]) => {
        if (Math.abs(value - min) <= 1e-8) return [`${name}:lower`];
        if (Math.abs(value - max) <= 1e-8) return [`${name}:upper`];
        return [];
    });
}

function perturbedConfig(config: BrowserOptimizationConfig, index: number): BrowserOptimizationConfig {
    const c = config.constraints;
    const armDelta = radians(c.posePerturbationDeg * (2 * halton(index + 1, 2) - 1));
    const wristDelta = radians(c.posePerturbationDeg * (2 * halton(index + 1, 3) - 1));
    return { ...config, initialState: [config.initialState[0] + armDelta, config.initialState[1] + wristDelta, config.initialState[2], config.initialState[3]] };
}

function perturbedCandidate(candidate: ForceSourceCandidate, config: BrowserOptimizationConfig, index: number): ForceSourceCandidate {
    const fraction = config.constraints.torquePerturbationFraction * (2 * halton(index + 1, 5) - 1);
    const scale = 1 + fraction;
    return {
        ...candidate,
        shoulder_torque_nm: Math.min(config.constraints.shoulderTorqueNm.max, Math.max(config.constraints.shoulderTorqueNm.min, candidate.shoulder_torque_nm * scale)),
        wrist_drive_nm: Math.min(config.constraints.wristTorqueLimitNm, candidate.wrist_drive_nm * scale),
        wrist_restrain_nm: Math.min(config.constraints.wristTorqueLimitNm, candidate.wrist_restrain_nm * scale),
    };
}

async function evaluateRobustness(best: EvaluatedCandidate, config: BrowserOptimizationConfig): Promise<RobustnessSummary> {
    const scores: Array<number | null> = [best.score];
    for (let index = 1; index < config.constraints.robustnessTrials; index++) {
        scores.push(evaluateCandidate(perturbedConfig(config, index), perturbedCandidate(best.candidate, config, index))?.score ?? null);
        if (index % 3 === 0) await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    return summarizeRobustness(scores);
}

export async function optimizeForceSource(
    config: BrowserOptimizationConfig,
    onProgress?: (progress: BrowserOptimizationProgress) => void,
): Promise<ForceSourceScenario> {
    const candidates = buildCandidateSet(config);
    let best: EvaluatedCandidate | null = null;
    const qualifiedScores: number[] = [];
    let completed = 0;
    for (const candidate of candidates) {
        const evaluated = evaluateCandidate(config, candidate);
        if (evaluated) qualifiedScores.push(evaluated.score);
        if (evaluated && (!best || evaluated.score > best.score)) best = evaluated;
        completed += 1;
        if (completed % 12 === 0) {
            onProgress?.({ completed, total: candidates.length, bestScore: best?.score ?? -Infinity, objective: config.objective });
            await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
    }
    if (!best) throw new Error('No candidate reached the qualified, non-looping golf impact event');
    const rounds = config.thoroughness === 'quick' ? 1 : config.thoroughness === 'thorough' ? 3 : 6;
    const convergence = [best.score];
    for (let round = 0; round < rounds; round++) {
        for (const candidate of refinementNeighbors(best.candidate, config)) {
            const evaluated = evaluateCandidate(config, candidate);
            if (evaluated) qualifiedScores.push(evaluated.score);
            if (evaluated && evaluated.score > best.score) best = evaluated;
        }
        convergence.push(best.score);
        await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    best.scenario.convergence = convergence;
    best.scenario.robustness = await evaluateRobustness(best, config);
    const nearOptimalFloor = best.score - 0.01 * Math.max(Math.abs(best.score), 1);
    best.scenario.near_optimal_count = qualifiedScores.filter(score => score >= nearOptimalFloor).length;
    best.scenario.boundary_hits = boundaryHits(best.candidate, config.constraints);
    best.scenario.provenance = {
        runtime: 'Tools pendulum-web deterministic search',
        thoroughness: config.thoroughness,
        global_candidate_count: candidates.length,
        qualified_evaluation_count: qualifiedScores.length,
        robustness_trial_count: config.constraints.robustnessTrials,
        integration_step_s: config.constraints.integrationStepS,
    };
    return best.scenario;
}

function candidateKey(candidate: ForceSourceCandidate): string {
    return [
        candidate.shoulder_torque_nm,
        candidate.wrist_drive_nm,
        candidate.wrist_restrain_nm,
        candidate.onset_s,
    ].join('|');
}

async function finalizedCrossWinner(
    evaluated: EvaluatedCandidate,
    original: ForceSourceScenario,
    config: BrowserOptimizationConfig,
    scores: number[],
): Promise<ForceSourceScenario> {
    const scenario = evaluated.scenario;
    scenario.convergence = [...original.convergence, evaluated.score];
    scenario.robustness = await evaluateRobustness(evaluated, config);
    const floor = evaluated.score - 0.01 * Math.max(Math.abs(evaluated.score), 1);
    scenario.near_optimal_count = scores.filter(score => score >= floor).length;
    scenario.boundary_hits = boundaryHits(evaluated.candidate, config.constraints);
    scenario.provenance = {
        ...original.provenance,
        comparison_contract_id: buildOptimizationContract(config).id,
        cross_objective_certified: true,
        cross_objective_candidate_count: scores.length,
    };
    return scenario;
}

/**
 * Optimize requested objectives and certify every winner against every displayed
 * candidate. This guarantees that an objective cannot lose to another row in
 * the same comparison, while making no claim of a mathematical global optimum.
 */
export async function optimizeForceSourceComparison(
    config: Omit<BrowserOptimizationConfig, 'objective'>,
    objectives: readonly ForceSourceObjective[] = FORCE_SOURCE_OBJECTIVES,
    seedScenarios: readonly ForceSourceScenario[] = [],
    onProgress?: (progress: BrowserOptimizationProgress) => void,
): Promise<ForceSourceScenario[]> {
    if (objectives.length === 0) throw new RangeError('at least one objective is required');
    const contractId = buildOptimizationContract({ ...config, objective: objectives[0] }).id;
    const retained = seedScenarios.filter(item => item.comparison_contract_id === contractId);
    const byObjective = new Map(retained.map(item => [item.objective, item]));
    for (const objective of objectives) {
        const scenario = await optimizeForceSource(
            { ...config, objective },
            progress => onProgress?.({ ...progress, objective }),
        );
        byObjective.set(objective, scenario);
    }

    const candidates = [...new Map(
        [...byObjective.values()].map(item => [candidateKey(item.candidate), item.candidate]),
    ).values()];
    for (const [objective, original] of byObjective) {
        const objectiveConfig: BrowserOptimizationConfig = { ...config, objective };
        const evaluated = candidates
            .map(candidate => evaluateCandidate(objectiveConfig, candidate))
            .filter((item): item is EvaluatedCandidate => item !== null);
        if (evaluated.length === 0) continue;
        const best = evaluated.reduce((winner, item) => item.score > winner.score ? item : winner);
        const scores = evaluated.map(item => item.score);
        if (best.score > original.score + OBJECTIVE_TOLERANCE) {
            byObjective.set(objective, await finalizedCrossWinner(best, original, objectiveConfig, scores));
        } else {
            original.provenance = {
                ...original.provenance,
                comparison_contract_id: contractId,
                cross_objective_certified: true,
                cross_objective_candidate_count: scores.length,
            };
        }
    }
    return FORCE_SOURCE_OBJECTIVES
        .filter(objective => byObjective.has(objective))
        .map(objective => byObjective.get(objective) as ForceSourceScenario);
}
