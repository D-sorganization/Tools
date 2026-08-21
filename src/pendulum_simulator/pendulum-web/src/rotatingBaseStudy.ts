import rawStudy from "../../../shared/python/swing_sim/rotating_base/resources/rotating_base_torso_velocity_study_v1.json";

export const ROTATING_BASE_SOURCE_REVISION =
  "967c40f54cc03f8cae89cde09268d62771d220fe";
export const ROTATING_BASE_STUDY_SHA256 =
  "e6a55e6cf91e51f21fe3eb8bcb07b990a7798f18abcaf5ca73f5214cb6c5f9ec";
export const ROTATING_BASE_MODEL_TIER =
  "planar_rotating_base_two_hand_compliant_club";

export const ROTATING_BASE_BOUNDARIES = {
  coordinate_semantics: "nonanatomical_model_coordinate",
  human_validation: "unavailable",
  coaching_recommendation: "unsupported",
} as const;

export type TorsoProfile = "accelerate" | "constant_rate" | "decelerate";
export type MatchingRule = "relative_club_rate" | "absolute_club_rate";

export interface RotatingBaseCaseMetrics {
  initial_club_rate_rad_s: number;
  final_torso_rate_rad_s: number;
  impact_speed_m_s: number;
  clubhead_speed_gain_m_s: number;
  contact_work_on_club_j: number;
  braking_grip_work_j: number;
  force_couple_work_j: number;
  negative_along_path_impulse_ns: number;
  bilateral_wrist_work_j: number;
  total_control_work_j: number;
  distal_energy_gain_j: number;
  peak_grip_force_n: number;
  maximum_constraint_residual_m: number;
  maximum_velocity_constraint_residual_m_s: number;
  maximum_contact_power_identity_residual_w: number;
  work_energy_closure_j: number;
}

export interface RotatingBaseCase extends RotatingBaseCaseMetrics {
  case_index: number;
  torso_profile: TorsoProfile;
  matching_rule: MatchingRule;
  initial_torso_rate_rad_s: number;
  valid: boolean;
  exclusion_reasons: string[];
}

export interface KillswitchChannel {
  pre_branch_state_max_abs_difference: number;
  delivery_speed_difference_m_s: number;
  post_branch_contact_work_difference_j: number;
}

export interface RotatingBaseStudy {
  schema_version: string;
  study_id: string;
  model_tier: string;
  attempted_case_count: number;
  valid_case_count: number;
  matching_rules: Record<MatchingRule, string>;
  cases: RotatingBaseCase[];
  same_state_killswitch: {
    branch_time_s: number;
    pre_branch_state_max_abs_difference: number;
    channels: Record<"torso" | "bilateral_arm" | "bilateral_wrist", KillswitchChannel>;
  };
  claims: {
    universal_high_torso_velocity_strategy: string;
    human_coaching_strategy: string;
  };
  limitations: string[];
}

const PROFILES = new Set<TorsoProfile>([
  "accelerate",
  "constant_rate",
  "decelerate",
]);
const MATCHING_RULES = new Set<MatchingRule>([
  "relative_club_rate",
  "absolute_club_rate",
]);
const TORSO_RATES = new Set([1.5, 3.5, 5.5]);
const METRIC_KEYS: Array<keyof RotatingBaseCaseMetrics> = [
  "initial_club_rate_rad_s",
  "final_torso_rate_rad_s",
  "impact_speed_m_s",
  "clubhead_speed_gain_m_s",
  "contact_work_on_club_j",
  "braking_grip_work_j",
  "force_couple_work_j",
  "negative_along_path_impulse_ns",
  "bilateral_wrist_work_j",
  "total_control_work_j",
  "distal_energy_gain_j",
  "peak_grip_force_n",
  "maximum_constraint_residual_m",
  "maximum_velocity_constraint_residual_m_s",
  "maximum_contact_power_identity_residual_w",
  "work_energy_closure_j",
];

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be finite`);
  }
  return value;
}

function stringArray(value: unknown, name: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new TypeError(`${name} must be a string array`);
  }
  return value;
}

function validateCase(value: unknown, index: number): RotatingBaseCase {
  const item = record(value, `case ${index}`);
  if (item.case_index !== index) throw new RangeError("case indices must be contiguous");
  if (!PROFILES.has(item.torso_profile as TorsoProfile)) {
    throw new RangeError("case torso profile is outside the registered design");
  }
  if (!MATCHING_RULES.has(item.matching_rule as MatchingRule)) {
    throw new RangeError("case matching rule is outside the registered design");
  }
  if (typeof item.valid !== "boolean") throw new TypeError("case validity must be Boolean");
  const reasons = stringArray(item.exclusion_reasons, "case exclusion reasons");
  if (item.valid === (reasons.length > 0)) {
    throw new RangeError("case validity and exclusion reasons disagree");
  }
  for (const key of METRIC_KEYS) finite(item[key], String(key));
  const torsoRate = finite(item.initial_torso_rate_rad_s, "initial_torso_rate_rad_s");
  if (!TORSO_RATES.has(torsoRate)) {
    throw new RangeError("initial torso rate is outside the registered design");
  }
  return item as unknown as RotatingBaseCase;
}

function validateChannel(value: unknown, name: string): KillswitchChannel {
  const channel = record(value, name);
  finite(channel.pre_branch_state_max_abs_difference, `${name} pre-branch closure`);
  finite(channel.delivery_speed_difference_m_s, `${name} delivery-speed difference`);
  finite(
    channel.post_branch_contact_work_difference_j,
    `${name} contact-work difference`,
  );
  return channel as unknown as KillswitchChannel;
}

export function validateRotatingBaseStudy(value: unknown): RotatingBaseStudy {
  const study = record(value, "rotating-base study");
  if (study.schema_version !== "rotating-base-torso-velocity-study-v1") {
    throw new RangeError("rotating-base schema version is unqualified");
  }
  if (study.study_id !== "registered-rotating-base-two-hand-torso-velocity-grid") {
    throw new RangeError("rotating-base study identifier is unqualified");
  }
  if (study.model_tier !== ROTATING_BASE_MODEL_TIER) {
    throw new RangeError("rotating-base model tier is unqualified");
  }
  if (study.attempted_case_count !== 18 || !Array.isArray(study.cases)) {
    throw new RangeError("the complete 18-case design must be retained");
  }
  const cases = study.cases.map(validateCase);
  const designKeys = new Set(
    cases.map(
      (item) =>
        `${item.matching_rule}/${item.torso_profile}/${item.initial_torso_rate_rad_s}`,
    ),
  );
  if (designKeys.size !== 18) {
    throw new RangeError("registered case combinations must be unique and complete");
  }
  if (study.valid_case_count !== cases.filter((item) => item.valid).length) {
    throw new RangeError("valid-case count does not match retained rows");
  }
  const rules = record(study.matching_rules, "matching rules");
  if (Object.keys(rules).join(",") !== "relative_club_rate,absolute_club_rate") {
    throw new RangeError("matching rules do not match the registered order");
  }
  const killswitch = record(study.same_state_killswitch, "same-state killswitch");
  const channels = record(killswitch.channels, "killswitch channels");
  if (Object.keys(channels).join(",") !== "torso,bilateral_arm,bilateral_wrist") {
    throw new RangeError("killswitch channels do not match the registered order");
  }
  for (const name of ["torso", "bilateral_arm", "bilateral_wrist"]) {
    validateChannel(channels[name], name);
  }
  finite(killswitch.branch_time_s, "killswitch branch time");
  finite(killswitch.pre_branch_state_max_abs_difference, "killswitch pre-branch closure");
  const claims = record(study.claims, "claims");
  if (
    claims.universal_high_torso_velocity_strategy !== "not_supported" &&
    claims.universal_high_torso_velocity_strategy !== "rejected"
  ) {
    throw new RangeError("universal torso-velocity strategy must remain unsupported");
  }
  if (claims.human_coaching_strategy !== "unsupported") {
    throw new RangeError("human coaching must remain unsupported");
  }
  const limitations = stringArray(study.limitations, "study limitations");
  if (limitations.length === 0) throw new RangeError("study limitations must be retained");
  return { ...(study as unknown as RotatingBaseStudy), cases };
}

export const ROTATING_BASE_STUDY = validateRotatingBaseStudy(rawStudy);

export function registeredCase(
  profile: TorsoProfile,
  matchingRule: MatchingRule,
  torsoRateRadS: number,
): RotatingBaseCase {
  const selected = ROTATING_BASE_STUDY.cases.find(
    (item) =>
      item.torso_profile === profile &&
      item.matching_rule === matchingRule &&
      item.initial_torso_rate_rad_s === torsoRateRadS,
  );
  if (!selected) throw new RangeError("selection is outside the registered design");
  return selected;
}
