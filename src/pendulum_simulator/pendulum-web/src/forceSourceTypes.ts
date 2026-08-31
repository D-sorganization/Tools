import type { PendulumParams, State } from './physics';

export const FORCE_SOURCE_SCHEMA = 'force-source-comparison/v4' as const;
export const TORQUE_POLYNOMIAL_DEGREE = 6 as const;

export const FORCE_SOURCE_OBJECTIVES = [
    'coriolis_impulse',
    'coriolis_energy_transfer',
    'centrifugal_impulse',
    'centrifugal_energy_transfer',
    'clubhead_speed',
    'hand_path_impulse',
] as const;

export type ForceSourceObjective = typeof FORCE_SOURCE_OBJECTIVES[number];
export type SearchThoroughness = 'quick' | 'thorough' | 'research';
export type ForceSourceStudyMode = 'common_bounds' | 'equal_effort' | 'equal_speed';

export const OBJECTIVE_LABELS: Record<ForceSourceObjective, string> = {
    coriolis_impulse: 'Coriolis impulse',
    coriolis_energy_transfer: 'Coriolis energy transfer',
    centrifugal_impulse: 'Centrifugal (squared-speed) impulse',
    centrifugal_energy_transfer: 'Centrifugal (squared-speed) energy transfer',
    clubhead_speed: 'Clubhead speed',
    hand_path_impulse: 'Hand-path force impulse',
};

export interface NumericRange { min: number; max: number; step: number }
export interface AngleBounds { min: number; max: number }

export interface ForceSourceConstraints {
    shoulderTorqueNm: NumericRange;
    wristTorqueLimitNm: number;
    wristTorqueStepNm: number;
    profileDurationS: NumericRange;
    maxTorqueSlewNmS: number;
    transitionTorqueNm: number;
    minWristTransitionS: number;
    studyMode: ForceSourceStudyMode;
    targetClubheadSpeedMps: number;
    speedToleranceMps: number;
    maxPositiveActuatorWorkJ: number;
    maxSquaredTorqueEffortNm2S: number;
    minimumRobustQualificationRate: number;
    eliteCandidateCount: number;
    armAngleDeg: AngleBounds;
    wristAngleDeg: AngleBounds;
    maxImpactPathAngleDeg: number;
    minBottomReachFraction: number;
    simulationDurationS: number;
    integrationStepS: number;
    maxArmTravelDeg: number;
    maxClubTravelDeg: number;
    candidateBudget: number;
    robustnessTrials: number;
    posePerturbationDeg: number;
    torquePerturbationFraction: number;
}

export const DEFAULT_OPTIMIZATION_CONSTRAINTS: ForceSourceConstraints = {
    shoulderTorqueNm: { min: -250, max: 250, step: 5 },
    wristTorqueLimitNm: 30,
    wristTorqueStepNm: 0.5,
    profileDurationS: { min: 0.3, max: 0.7, step: 0.01 },
    maxTorqueSlewNmS: 2500,
    transitionTorqueNm: 3,
    minWristTransitionS: 0.012,
    studyMode: 'equal_speed',
    targetClubheadSpeedMps: 52.3,
    speedToleranceMps: 0.75,
    maxPositiveActuatorWorkJ: 525,
    maxSquaredTorqueEffortNm2S: 7500,
    minimumRobustQualificationRate: 0.6,
    eliteCandidateCount: 8,
    armAngleDeg: { min: -180, max: 45 },
    wristAngleDeg: { min: -135, max: 110 },
    maxImpactPathAngleDeg: 15,
    minBottomReachFraction: 0.9,
    simulationDurationS: 0.9,
    integrationStepS: 0.002,
    maxArmTravelDeg: 180,
    maxClubTravelDeg: 270,
    candidateBudget: 160,
    robustnessTrials: 9,
    posePerturbationDeg: 1,
    torquePerturbationFraction: 0.03,
};

/** Registered golf-like address for the bundled force-source comparison. */
export const DEFAULT_FORCE_SOURCE_INITIAL_STATE: State = [-2.2, -1.57, 0, 0];

export interface BrowserOptimizationConfig {
    params: PendulumParams;
    initialState: State;
    objective: ForceSourceObjective;
    thoroughness: SearchThoroughness;
    constraints: ForceSourceConstraints;
}

export interface BrowserOptimizationProgress {
    completed: number;
    total: number;
    bestScore: number;
    objective?: ForceSourceObjective;
}

export type TorquePolynomialCoefficients = [number, number, number, number, number, number, number];

export interface ForceSourceCandidate {
    basis: 'bernstein_6';
    profile_duration_s: number;
    shoulder_coefficients_nm: TorquePolynomialCoefficients;
    wrist_coefficients_nm: TorquePolynomialCoefficients;
}

export interface TorqueProfileDiagnostics {
    peak_shoulder_torque_nm: number;
    peak_wrist_torque_nm: number;
    rms_shoulder_torque_nm: number;
    rms_wrist_torque_nm: number;
    peak_shoulder_slew_nm_s: number;
    peak_wrist_slew_nm_s: number;
    wrist_reversal_count: number;
    wrist_reversal_time_s: number | null;
    wrist_transition_duration_s: number;
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
    hand_path_tangent_force_n?: number[];
    shoulder_actuator_power_w: number[];
    wrist_actuator_power_w: number[];
    total_actuator_power_w: number[];
    cumulative_positive_actuator_work_j: number[];
    cumulative_net_actuator_work_j: number[];
}

export interface ActuatorEffortMetrics {
    shoulder_net_work_j: number;
    wrist_net_work_j: number;
    total_net_work_j: number;
    total_positive_work_j: number;
    total_negative_work_j: number;
    absolute_torque_impulse_nm_s: number;
    squared_torque_effort_nm2_s: number;
    peak_shoulder_power_w: number;
    peak_wrist_power_w: number;
    peak_total_power_w: number;
}

export interface ImpactDiagnostics {
    path_angle_deg: number;
    bottom_reach_fraction: number;
    x_velocity_m_s: number;
    y_velocity_m_s: number;
    arm_angle_deg: number;
    club_angle_deg: number;
}

export interface ForceSourceScenario {
    objective: ForceSourceObjective;
    score: number;
    candidate: ForceSourceCandidate;
    profile_id: string;
    effort: ActuatorEffortMetrics;
    impact_time_s: number;
    impact_diagnostics?: ImpactDiagnostics;
    robustness: RobustnessSummary;
    near_optimal_count: number;
    boundary_hits: string[];
    convergence: number[];
    series: ForceSourceSeries;
    comparison_contract_id: string;
    provenance?: Record<string, unknown>;
}

export interface ForceSourceComparisonContract {
    id: string;
    thoroughness: SearchThoroughness;
    constraints: ForceSourceConstraints;
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
    comparison_contract: ForceSourceComparisonContract;
    evaluated_count: number;
    qualified_count: number;
    qualification_counts?: Record<string, number>;
    scenarios: ForceSourceScenario[];
    interpretation_limits: string[];
    provenance?: Record<string, unknown>;
}
