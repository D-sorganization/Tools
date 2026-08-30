import {
    buildCandidateSet,
    DEFAULT_FORCE_SOURCE_INITIAL_STATE,
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    evaluateForceSourceCandidate,
    type BrowserOptimizationConfig,
} from '../src/forceSourceStudy';
import { PRESETS } from '../src/presets';

const config: BrowserOptimizationConfig = {
    params: PRESETS[0].params,
    initialState: DEFAULT_FORCE_SOURCE_INITIAL_STATE,
    objective: 'clubhead_speed',
    thoroughness: 'research',
    constraints: {
        ...DEFAULT_OPTIMIZATION_CONSTRAINTS,
        studyMode: 'common_bounds',
        candidateBudget: 2048,
        robustnessTrials: 1,
        integrationStepS: 0.001,
    },
};

const scenarios = buildCandidateSet(config)
    .map(candidate => evaluateForceSourceCandidate(config, candidate))
    .filter(item => item !== null);
const target = config.constraints.targetClubheadSpeedMps;
const tolerance = config.constraints.speedToleranceMps;
const inSpeedBand = scenarios.filter(item => {
    const value = item.series.clubhead_speed_m_s[item.series.clubhead_speed_m_s.length - 1] ?? 0;
    return value >= target && value <= target + tolerance;
});
const inEffortCaps = scenarios.filter(item =>
    item.effort.total_positive_work_j <= config.constraints.maxPositiveActuatorWorkJ
    && item.effort.squared_torque_effort_nm2_s <= config.constraints.maxSquaredTorqueEffortNm2S);
const fullyFair = inSpeedBand.filter(item => inEffortCaps.includes(item));
const speed = (item: typeof scenarios[number]) => item.series.clubhead_speed_m_s[
    item.series.clubhead_speed_m_s.length - 1
] ?? 0;
const extrema = <T>(values: T[], value: (item: T) => number) => values.length === 0 ? null : {
    min: Math.min(...values.map(value)),
    max: Math.max(...values.map(value)),
};

process.stdout.write(`${JSON.stringify({
    global_candidates: config.constraints.candidateBudget,
    golf_qualified: scenarios.length,
    speed_range_m_s: extrema(scenarios, speed),
    speed_band: { target, tolerance, count: inSpeedBand.length },
    effort_caps: {
        positive_work_j: config.constraints.maxPositiveActuatorWorkJ,
        squared_torque_nm2_s: config.constraints.maxSquaredTorqueEffortNm2S,
        count: inEffortCaps.length,
    },
    fair_count: fullyFair.length,
    speed_band_positive_work_j: extrema(inSpeedBand, item => item.effort.total_positive_work_j),
    speed_band_squared_effort_nm2_s: extrema(inSpeedBand, item => item.effort.squared_torque_effort_nm2_s),
    closest: [...scenarios].sort((left, right) => Math.abs(speed(left) - target)
        - Math.abs(speed(right) - target)).slice(0, 10).map(item => ({
        speed_m_s: speed(item),
        positive_work_j: item.effort.total_positive_work_j,
        squared_effort_nm2_s: item.effort.squared_torque_effort_nm2_s,
        profile_id: item.profile_id,
    })),
}, null, 2)}\n`);
