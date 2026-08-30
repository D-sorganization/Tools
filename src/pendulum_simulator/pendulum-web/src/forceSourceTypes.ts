import type { PendulumParams, State } from './physics';

export const FORCE_SOURCE_SCHEMA = 'force-source-comparison/v2' as const;

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
    onsetS: NumericRange;
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
    shoulderTorqueNm: { min: 60, max: 120, step: 10 },
    wristTorqueLimitNm: 30,
    wristTorqueStepNm: 1,
    onsetS: { min: 0.04, max: 0.28, step: 0.01 },
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
    hand_path_tangent_force_n?: number[];
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
