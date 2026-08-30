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
    type TorquePolynomialCoefficients,
    type TorqueProfileDiagnostics,
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
            c.profileDurationS.min, c.profileDurationS.max, c.profileDurationS.step,
            c.maxTorqueSlewNmS, c.transitionTorqueNm, c.minWristTransitionS,
            c.targetClubheadSpeedMps, c.eliteCandidateCount,
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
        id: `force-source-search/v2-${contractChecksum(JSON.stringify(contractPayload(config)))}`,
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
    validateRange(c.profileDurationS, 'profile duration');
    if (c.shoulderTorqueNm.min > 0 || c.shoulderTorqueNm.max < 0) {
        throw new RangeError('shoulder torque bounds must contain zero for continuous profile endpoints');
    }
    if (!(c.wristTorqueLimitNm > 0 && c.wristTorqueLimitNm <= 30)) {
        throw new RangeError('wrist torque limit must be in (0, 30] N m');
    }
    if (!(c.wristTorqueStepNm >= 0.05 && c.wristTorqueStepNm <= c.wristTorqueLimitNm)) {
        throw new RangeError('wrist torque step must be between 0.05 N m and the wrist limit');
    }
    if (!(c.profileDurationS.min > 0 && c.profileDurationS.max <= c.simulationDurationS)) throw new RangeError('profile duration must be positive and inside the simulation duration');
    if (!(c.maxTorqueSlewNmS > 0)) throw new RangeError('torque slew limit must be positive');
    if (!(c.transitionTorqueNm > 0 && c.transitionTorqueNm < c.wristTorqueLimitNm)) throw new RangeError('transition torque must be inside the wrist-torque bounds');
    if (!(c.minWristTransitionS > 0 && c.minWristTransitionS < c.profileDurationS.max)) throw new RangeError('wrist transition duration must be positive and shorter than the profile');
    if (!(c.targetClubheadSpeedMps > 0)) throw new RangeError('clubhead speed target must be positive');
    if (!Number.isInteger(c.eliteCandidateCount) || c.eliteCandidateCount < 1 || c.eliteCandidateCount > 64) throw new RangeError('elite candidate count must be an integer in [1, 64]');
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

const BINOMIAL_6 = [1, 6, 15, 20, 15, 6, 1] as const;

/** Evaluate a degree-6 Bernstein polynomial at normalized phase [0, 1]. */
export function bernsteinTorque(
    coefficients: TorquePolynomialCoefficients,
    phase: number,
): number {
    const value = Math.min(1, Math.max(0, phase));
    const complement = 1 - value;
    return coefficients.reduce((sum, coefficient, index) => sum
        + coefficient * BINOMIAL_6[index] * value ** index * complement ** (6 - index), 0);
}

/** Create the continuous two-joint torque function represented by a candidate. */
export function candidateTorqueFunction(candidate: ForceSourceCandidate): TorqueFunc {
    return time => {
        if (time < 0 || time > candidate.profile_duration_s) return [0, 0];
        const phase = time / candidate.profile_duration_s;
        return [
            bernsteinTorque(candidate.shoulder_coefficients_nm, phase),
            bernsteinTorque(candidate.wrist_coefficients_nm, phase),
        ];
    };
}

function derivativeBound(coefficients: TorquePolynomialCoefficients, duration: number): number {
    return 6 * Math.max(...coefficients.slice(1).map((value, index) =>
        Math.abs(value - coefficients[index]))) / duration;
}

/** Summarize shape, continuity, slew, and wrist-transition properties. */
export function profileDiagnostics(
    candidate: ForceSourceCandidate,
    transitionTorqueNm: number,
): TorqueProfileDiagnostics {
    const sampleCount = 1001;
    const torque = candidateTorqueFunction(candidate);
    const samples = Array.from({ length: sampleCount }, (_, index) => ({
        time: candidate.profile_duration_s * index / (sampleCount - 1),
        value: torque(candidate.profile_duration_s * index / (sampleCount - 1)),
    }));
    const shoulder = samples.map(sample => sample.value[0]);
    const wrist = samples.map(sample => sample.value[1]);
    const signs = wrist.map(value => value > 1e-7 ? 1 : value < -1e-7 ? -1 : 0);
    let previousSign = 0;
    let reversalCount = 0;
    let reversalIndex: number | null = null;
    for (const [index, sign] of signs.entries()) {
        if (sign === 0) continue;
        if (previousSign < 0 && sign > 0) {
            reversalCount += 1;
            reversalIndex ??= index;
        } else if (previousSign > 0 && sign < 0) {
            reversalCount += 1;
            reversalIndex ??= index;
        }
        previousSign = sign;
    }
    let transitionSamples = 0;
    if (reversalIndex !== null) {
        let left = reversalIndex;
        let right = reversalIndex;
        while (left > 0 && Math.abs(wrist[left - 1]) <= transitionTorqueNm) left -= 1;
        while (right < wrist.length - 1 && Math.abs(wrist[right + 1]) <= transitionTorqueNm) right += 1;
        transitionSamples = right - left;
    }
    const rms = (values: number[]) => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
    return {
        peak_shoulder_torque_nm: Math.max(...shoulder.map(Math.abs)),
        peak_wrist_torque_nm: Math.max(...wrist.map(Math.abs)),
        rms_shoulder_torque_nm: rms(shoulder),
        rms_wrist_torque_nm: rms(wrist),
        peak_shoulder_slew_nm_s: derivativeBound(candidate.shoulder_coefficients_nm, candidate.profile_duration_s),
        peak_wrist_slew_nm_s: derivativeBound(candidate.wrist_coefficients_nm, candidate.profile_duration_s),
        wrist_reversal_count: reversalCount,
        wrist_reversal_time_s: reversalIndex === null ? null : samples[reversalIndex].time,
        wrist_transition_duration_s: candidate.profile_duration_s * transitionSamples / (sampleCount - 1),
    };
}

function quantizeValue(value: number, range: NumericRange): number {
    return quantized((value - range.min) / (range.max - range.min || 1), range);
}

function boundedCoefficients(
    raw: TorquePolynomialCoefficients,
    range: NumericRange,
    duration: number,
    maxSlew: number,
    zeroStart = true,
): TorquePolynomialCoefficients {
    const gridDelta = Math.max(range.step, Math.floor(maxSlew * duration / (6 * range.step)) * range.step);
    const values = raw.map(value => quantizeValue(Math.min(range.max, Math.max(range.min, value)), range));
    if (zeroStart) values[0] = 0;
    values[6] = 0;
    for (let pass = 0; pass < 3; pass += 1) {
        for (let index = 1; index < values.length; index += 1) {
            values[index] = quantizeValue(Math.min(values[index - 1] + gridDelta,
                Math.max(values[index - 1] - gridDelta, values[index])), range);
        }
        for (let index = values.length - 2; index >= 0; index -= 1) {
            values[index] = quantizeValue(Math.min(values[index + 1] + gridDelta,
                Math.max(values[index + 1] - gridDelta, values[index])), range);
        }
        if (zeroStart) values[0] = 0;
        values[6] = 0;
    }
    return values as TorquePolynomialCoefficients;
}

function candidateIsShapeQualified(candidate: ForceSourceCandidate, constraints: ForceSourceConstraints): boolean {
    const diagnostics = profileDiagnostics(candidate, constraints.transitionTorqueNm);
    return diagnostics.peak_shoulder_slew_nm_s <= constraints.maxTorqueSlewNmS + 1e-8
        && diagnostics.peak_wrist_slew_nm_s <= constraints.maxTorqueSlewNmS + 1e-8
        && diagnostics.wrist_reversal_count === 1
        && diagnostics.wrist_transition_duration_s >= constraints.minWristTransitionS - 1e-3;
}

function sampledCandidate(index: number, constraints: ForceSourceConstraints): ForceSourceCandidate {
    const duration = quantized(halton(index, 31), constraints.profileDurationS);
    const shoulderRaw = [2, 3, 5, 7, 11, 37, 0].map(base => base === 0 ? 0
        : constraints.shoulderTorqueNm.min
            + halton(index, base) * (constraints.shoulderTorqueNm.max - constraints.shoulderTorqueNm.min)) as TorquePolynomialCoefficients;
    const limit = constraints.wristTorqueLimitNm;
    const wristRaw: TorquePolynomialCoefficients = [
        0,
        -limit * (0.15 + 0.85 * halton(index, 13)),
        -limit * (0.1 + 0.9 * halton(index, 17)),
        constraints.transitionTorqueNm * (2 * halton(index, 19) - 1),
        limit * (0.1 + 0.9 * halton(index, 23)),
        limit * (0.15 + 0.85 * halton(index, 29)),
        0,
    ];
    const wristRange = { min: -limit, max: limit, step: constraints.wristTorqueStepNm };
    return {
        basis: 'bernstein_6',
        profile_duration_s: duration,
        shoulder_coefficients_nm: boundedCoefficients(
            shoulderRaw, constraints.shoulderTorqueNm, duration, constraints.maxTorqueSlewNmS, false,
        ),
        wrist_coefficients_nm: boundedCoefficients(
            wristRaw, wristRange, duration, constraints.maxTorqueSlewNmS,
        ),
    };
}

function seededCandidates(constraints: ForceSourceConstraints): ForceSourceCandidate[] {
    const limit = constraints.wristTorqueLimitNm;
    const wristRange = { min: -limit, max: limit, step: constraints.wristTorqueStepNm };
    const wristShapes: TorquePolynomialCoefficients[] = [
        [0, -0.1 * limit, limit, limit, limit, limit, 0],
        [0, -0.3 * limit, 0.4 * limit, limit, limit, 0.8 * limit, 0],
        [0, -0.55 * limit, -0.5 * limit, 0, 0.65 * limit, limit, 0],
        [0, -0.7 * limit, -0.7 * limit, -0.1 * limit, 0.1 * limit, limit, 0],
    ];
    const shoulderLevels = [60, 80, 100, 120, 140, constraints.shoulderTorqueNm.max]
        .filter((value, index, values) => value >= constraints.shoulderTorqueNm.min
            && value <= constraints.shoulderTorqueNm.max && values.indexOf(value) === index);
    const durations = [0.4, 0.5, 0.6, constraints.profileDurationS.max]
        .map(value => quantizeValue(value, constraints.profileDurationS))
        .filter((value, index, values) => values.indexOf(value) === index);
    return shoulderLevels.flatMap(shoulderLevel => durations.flatMap(duration => {
        const endpointNeighbor = Math.min(shoulderLevel,
            Math.floor(constraints.maxTorqueSlewNmS * duration
                / (6 * constraints.shoulderTorqueNm.step)) * constraints.shoulderTorqueNm.step);
        const brake = Math.max(constraints.shoulderTorqueNm.min, -0.5 * shoulderLevel);
        const shoulderShapes: TorquePolynomialCoefficients[] = [
            [shoulderLevel, shoulderLevel, shoulderLevel, shoulderLevel,
                shoulderLevel, endpointNeighbor, 0],
            [shoulderLevel, shoulderLevel, shoulderLevel, 0.6 * shoulderLevel,
                0, brake * 0.5, 0],
            [shoulderLevel, shoulderLevel, 0.8 * shoulderLevel, 0,
                brake, brake * 0.5, 0],
            [0, 0.5 * shoulderLevel, shoulderLevel, shoulderLevel,
                0.5 * shoulderLevel, 0, 0],
        ];
        return shoulderShapes.flatMap(shoulderRaw => {
            const shoulder = boundedCoefficients(
                shoulderRaw, constraints.shoulderTorqueNm, duration,
                constraints.maxTorqueSlewNmS, false,
            );
            return wristShapes.map(raw => ({
                basis: 'bernstein_6' as const,
                profile_duration_s: duration,
                shoulder_coefficients_nm: shoulder,
                wrist_coefficients_nm: boundedCoefficients(
                    raw, wristRange, duration, constraints.maxTorqueSlewNmS,
                ),
            }));
        });
    })).filter(candidate => candidateIsShapeQualified(candidate, constraints));
}

export function buildCandidateSet(config: BrowserOptimizationConfig): ForceSourceCandidate[] {
    validateBrowserOptimizationConfig(config);
    const c = config.constraints;
    const candidates = seededCandidates(c).slice(0, c.candidateBudget);
    for (let index = 1; candidates.length < c.candidateBudget && index < c.candidateBudget * 100; index += 1) {
        const candidate = sampledCandidate(index, c);
        if (candidateIsShapeQualified(candidate, c)) candidates.push(candidate);
    }
    if (candidates.length !== c.candidateBudget) {
        throw new Error('Unable to construct enough continuous torque profiles under the selected slew and transition constraints');
    }
    return candidates;
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
    if (!candidateIsShapeQualified(candidate, config.constraints)) return null;
    const torque = candidateTorqueFunction(candidate);
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

function coefficientNeighbor(
    candidate: ForceSourceCandidate,
    joint: 'shoulder' | 'wrist',
    index: number,
    direction: -1 | 1,
    config: BrowserOptimizationConfig,
): ForceSourceCandidate | null {
    const c = config.constraints;
    const field = joint === 'shoulder' ? 'shoulder_coefficients_nm' : 'wrist_coefficients_nm';
    const step = joint === 'shoulder' ? c.shoulderTorqueNm.step : c.wristTorqueStepNm;
    const min = joint === 'shoulder' ? c.shoulderTorqueNm.min : -c.wristTorqueLimitNm;
    const max = joint === 'shoulder' ? c.shoulderTorqueNm.max : c.wristTorqueLimitNm;
    const coefficients = [...candidate[field]] as TorquePolynomialCoefficients;
    coefficients[index] = Math.min(max, Math.max(min, +(coefficients[index] + direction * step).toFixed(8)));
    const next = { ...candidate, [field]: coefficients };
    return candidateIsShapeQualified(next, c) ? next : null;
}

function refinementNeighbors(candidate: ForceSourceCandidate, config: BrowserOptimizationConfig): ForceSourceCandidate[] {
    const candidates: Array<ForceSourceCandidate | null> = [candidate];
    for (const joint of ['shoulder', 'wrist'] as const) {
        for (let index = 1; index <= 5; index += 1) {
            candidates.push(coefficientNeighbor(candidate, joint, index, -1, config));
            candidates.push(coefficientNeighbor(candidate, joint, index, 1, config));
        }
    }
    const c = config.constraints;
    for (const direction of [-1, 1] as const) {
        const profile_duration_s = Math.min(c.profileDurationS.max, Math.max(c.profileDurationS.min,
            +(candidate.profile_duration_s + direction * c.profileDurationS.step).toFixed(8)));
        const next = { ...candidate, profile_duration_s };
        candidates.push(candidateIsShapeQualified(next, c) ? next : null);
    }
    return candidates.filter((item): item is ForceSourceCandidate => item !== null);
}

function boundaryHits(candidate: ForceSourceCandidate, constraints: ForceSourceConstraints): string[] {
    const checks: Array<[string, number, number, number]> = [
        ...candidate.shoulder_coefficients_nm.map((value, index) => [
            `shoulder_coefficients_nm[${index}]`, value,
            constraints.shoulderTorqueNm.min, constraints.shoulderTorqueNm.max,
        ] as [string, number, number, number]),
        ...candidate.wrist_coefficients_nm.map((value, index) => [
            `wrist_coefficients_nm[${index}]`, value,
            -constraints.wristTorqueLimitNm, constraints.wristTorqueLimitNm,
        ] as [string, number, number, number]),
        ['profile_duration_s', candidate.profile_duration_s,
            constraints.profileDurationS.min, constraints.profileDurationS.max],
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
        shoulder_coefficients_nm: candidate.shoulder_coefficients_nm.map(value =>
            Math.min(config.constraints.shoulderTorqueNm.max,
                Math.max(config.constraints.shoulderTorqueNm.min, value * scale))) as TorquePolynomialCoefficients,
        wrist_coefficients_nm: candidate.wrist_coefficients_nm.map(value =>
            Math.min(config.constraints.wristTorqueLimitNm,
                Math.max(-config.constraints.wristTorqueLimitNm, value * scale))) as TorquePolynomialCoefficients,
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
    let elite: EvaluatedCandidate[] = [];
    const qualifiedScores: number[] = [];
    let completed = 0;
    for (const candidate of candidates) {
        const evaluated = evaluateCandidate(config, candidate);
        if (evaluated) qualifiedScores.push(evaluated.score);
        if (evaluated) {
            elite = [...elite, evaluated]
                .sort((left, right) => right.score - left.score)
                .slice(0, config.constraints.eliteCandidateCount);
            if (!best || evaluated.score > best.score) best = evaluated;
        }
        completed += 1;
        if (completed % 12 === 0) {
            onProgress?.({ completed, total: candidates.length, bestScore: best?.score ?? -Infinity, objective: config.objective });
            await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
    }
    if (!best) throw new Error('No candidate reached the qualified, non-looping golf impact event');
    const rounds = config.thoroughness === 'quick' ? 2 : config.thoroughness === 'thorough' ? 6 : 12;
    const convergence = [best.score];
    for (let round = 0; round < rounds; round++) {
        const neighborhood = [...new Map(elite.flatMap(item => refinementNeighbors(item.candidate, config))
            .map(candidate => [candidateKey(candidate), candidate])).values()];
        const refined: EvaluatedCandidate[] = [];
        for (const candidate of neighborhood) {
            const evaluated = evaluateCandidate(config, candidate);
            if (evaluated) {
                refined.push(evaluated);
                qualifiedScores.push(evaluated.score);
            }
        }
        elite = [...elite, ...refined]
            .sort((left, right) => right.score - left.score)
            .filter((item, index, values) => values.findIndex(other =>
                candidateKey(other.candidate) === candidateKey(item.candidate)) === index)
            .slice(0, config.constraints.eliteCandidateCount);
        best = elite[0] ?? best;
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
        candidate.basis,
        candidate.profile_duration_s,
        ...candidate.shoulder_coefficients_nm,
        ...candidate.wrist_coefficients_nm,
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
