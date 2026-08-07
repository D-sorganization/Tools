/** Canonical, versioned launch-monitor-style ball-flight result contract. */

export const FLIGHT_METRIC_IDS = [
  "initial_velocity", "ball_speed", "vertical_launch_angle", "launch_direction",
  "spin_vector", "total_spin", "spin_axis_tilt", "landing_position",
  "landing_velocity", "carry_distance", "carry_offline", "apex_height",
  "flight_time", "landing_angle", "curve", "terminal_speed",
  "terminal_direction", "target_residual", "target_downrange_residual",
  "target_lateral_residual", "total_distance", "roll_distance", "bounce_count",
  "final_offline",
] as const;
export type FlightMetricId = typeof FLIGHT_METRIC_IDS[number];
export type ValueStatus = "input" | "directly_simulated" | "derived" |
  "model_dependent" | "estimated" | "optimized" | "unsupported" | "unavailable";
export type SignRule = "nonnegative" | "positive_right" | "positive_up" |
  "positive_down" | "vector_components" | "signed";
export type ComparabilityStatus = "native" | "definition_aligned" | "not_comparable";

export interface ComparisonCoverageTs {
  readonly conventionId: string;
  readonly status: ComparabilityStatus;
  readonly reasonCode: string;
  readonly sourceUrl: string;
}
export interface FlightMetricDefinitionTs {
  readonly metricId: FlightMetricId;
  readonly label: string;
  readonly definition: string;
  readonly unit: string;
  readonly defaultStatus: ValueStatus;
  readonly frameId: string;
  readonly signRule: SignRule;
  readonly referenceEvent: string;
  readonly geometryContract: string;
  readonly provenance: string;
  readonly availabilityRule: string;
  readonly solverObjective: boolean;
  readonly coverage: readonly ComparisonCoverageTs[];
}
export interface FlightMetricCatalogTs {
  readonly schemaVersion: "ball-flight-metrics/v1";
  readonly definitions: readonly FlightMetricDefinitionTs[];
  definition(metricId: string): FlightMetricDefinitionTs;
}

interface Identity {
  label: string; definition: string; unit: string; defaultStatus: ValueStatus;
  signRule: SignRule; referenceEvent: string; geometryContract: string;
  availabilityRule: string; solverObjective?: boolean;
}

const FRAME = "target_frame:x_downrange,y_up,z_right";
const APP_SOURCE = "https://github.com/D-sorganization/Tools/blob/main/docs/specs/BALL_FLIGHT_RESULT_CONTRACT.md";
const TRACKMAN_SOURCE = "https://www.trackman.com/blog/golf/40-trackman-parameters";
const FORESIGHT_SOURCE = "https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results";

const identity = (
  label: string, definition: string, unit: string, defaultStatus: ValueStatus,
  signRule: SignRule, referenceEvent: string, geometryContract: string,
  availabilityRule: string, solverObjective = false,
): Identity => ({
  label, definition, unit, defaultStatus, signRule, referenceEvent,
  geometryContract, availabilityRule, solverObjective,
});

const IDENTITIES: Record<FlightMetricId, Identity> = {
  initial_velocity: identity("Initial Velocity", "Ball-center velocity immediately after separation.", "m/s", "input", "vector_components", "launch", "vector(v_x,v_y,v_z)", "initial trajectory point"),
  ball_speed: identity("Ball Speed", "Magnitude of initial ball velocity.", "m/s", "derived", "nonnegative", "launch", "norm(initial_velocity)", "initial trajectory point", true),
  vertical_launch_angle: identity("Vertical Launch Angle", "Elevation of initial velocity above horizontal.", "deg", "derived", "positive_up", "launch", "atan2(v_y,hypot(v_x,v_z))", "nonzero initial speed", true),
  launch_direction: identity("Launch Direction", "Horizontal heading of initial velocity from the target line.", "deg", "derived", "positive_right", "launch", "atan2(v_z,v_x)", "nonzero horizontal speed", true),
  spin_vector: identity("Spin Vector", "Post-separation angular-velocity vector in target-frame components.", "rpm", "input", "vector_components", "launch", "vector(omega_x,omega_y,omega_z)", "post-impact spin state"),
  total_spin: identity("Total Spin", "Magnitude of the post-separation spin vector.", "rpm", "derived", "nonnegative", "launch", "norm(spin_vector)", "post-impact spin state", true),
  spin_axis_tilt: identity("Spin Axis Tilt", "Signed tilt of the spin axis from +right toward +up.", "deg", "derived", "positive_up", "launch", "atan2(omega_y,omega_z)", "nonzero spin", true),
  landing_position: identity("Landing Position", "Linearly interpolated first descending ground crossing.", "m", "derived", "vector_components", "landing", "interpolate(position,y=0)", "ground crossing"),
  landing_velocity: identity("Landing Velocity", "Velocity interpolated at the first descending ground crossing.", "m/s", "derived", "vector_components", "landing", "interpolate(velocity,y=0)", "ground crossing"),
  carry_distance: identity("Carry Distance", "Horizontal distance from launch to first ground contact; excludes bounce and roll.", "m", "derived", "nonnegative", "landing", "hypot(delta_x,delta_z)", "ground crossing", true),
  carry_offline: identity("Carry Offline", "Signed right/left displacement at first ground contact.", "m", "derived", "positive_right", "landing", "landing_z-launch_z", "ground crossing", true),
  apex_height: identity("Apex Height", "Maximum sampled height above the launch ground plane.", "m", "derived", "positive_up", "airborne", "max(position_y)", "trajectory samples", true),
  flight_time: identity("Flight Time", "Elapsed time from launch to first ground contact.", "s", "derived", "nonnegative", "landing", "landing_time-launch_time", "ground crossing", true),
  landing_angle: identity("Landing Angle", "Downward angle of landing velocity below horizontal.", "deg", "derived", "positive_down", "landing", "atan2(-v_y,hypot(v_x,v_z))", "ground crossing and horizontal speed", true),
  curve: identity("Curve", "Largest signed lateral departure from the initial vertical launch plane; not landing offline.", "m", "derived", "positive_right", "airborne", "signed_max_abs(-sin(heading)*x+cos(heading)*z)", "ground crossing and horizontal speed", true),
  terminal_speed: identity("Terminal Speed", "Magnitude of velocity at first ground contact.", "m/s", "derived", "nonnegative", "landing", "norm(landing_velocity)", "ground crossing", true),
  terminal_direction: identity("Terminal Direction", "Horizontal heading of velocity at first ground contact.", "deg", "derived", "positive_right", "landing", "atan2(v_z,v_x)", "ground crossing and horizontal speed", true),
  target_residual: identity("Target Residual", "Three-dimensional miss distance from landing point to target.", "m", "derived", "nonnegative", "landing", "norm(landing_position-target)", "ground crossing and target", true),
  target_downrange_residual: identity("Target Downrange Residual", "Landing minus target downrange position.", "m", "derived", "signed", "landing", "landing_x-target_x", "ground crossing and target", true),
  target_lateral_residual: identity("Target Lateral Residual", "Landing minus target lateral position; positive right.", "m", "derived", "positive_right", "landing", "landing_z-target_z", "ground crossing and target", true),
  total_distance: identity("Total Distance", "Horizontal distance after a qualified bounce-and-roll model terminates.", "m", "model_dependent", "nonnegative", "ground_stop", "qualified_ground_model(total_distance)", "qualified ground model", true),
  roll_distance: identity("Roll Distance", "Ground-travel distance reported by a qualified ground model.", "m", "model_dependent", "nonnegative", "ground_stop", "qualified_ground_model(roll_distance)", "qualified ground model", true),
  bounce_count: identity("Bounce Count", "Number of bounce events resolved by a qualified ground model.", "count", "model_dependent", "nonnegative", "ground_stop", "qualified_ground_model(bounce_count)", "qualified ground model"),
  final_offline: identity("Final Offline", "Signed lateral position after qualified ground motion terminates.", "m", "model_dependent", "positive_right", "ground_stop", "qualified_ground_model(final_offline)", "qualified ground model", true),
};

const PUBLICLY_ALIGNED = new Set<FlightMetricId>([
  "ball_speed", "vertical_launch_angle", "launch_direction", "total_spin",
  "spin_axis_tilt", "carry_distance", "carry_offline", "apex_height",
  "flight_time", "landing_angle", "curve", "total_distance", "roll_distance",
]);

const coverage = (metricId: FlightMetricId): readonly ComparisonCoverageTs[] => {
  const aligned = PUBLICLY_ALIGNED.has(metricId);
  const status = aligned ? "definition_aligned" : "not_comparable";
  const reasonCode = aligned ? "modeled_not_measured" : "public_definition_not_established";
  return Object.freeze([
    Object.freeze({ conventionId: "app_native", status: "native", reasonCode: "canonical_app_definition", sourceUrl: APP_SOURCE }),
    Object.freeze({ conventionId: "trackman_comparable", status, reasonCode, sourceUrl: TRACKMAN_SOURCE }),
    Object.freeze({ conventionId: "foresight_comparable", status, reasonCode, sourceUrl: FORESIGHT_SOURCE }),
  ]);
};

const makeCatalog = (definitions: readonly FlightMetricDefinitionTs[]): FlightMetricCatalogTs => {
  const ordered = Object.freeze([...definitions].sort((left, right) =>
    left.metricId.localeCompare(right.metricId)));
  return Object.freeze({
    schemaVersion: "ball-flight-metrics/v1" as const,
    definitions: ordered,
    definition: (metricId: string) => {
      const found = ordered.find((item) => item.metricId === metricId);
      if (!found) throw new RangeError(`unknown ball-flight metric: ${metricId}`);
      return found;
    },
  });
};

let cached: FlightMetricCatalogTs | null = null;
export const flightMetricCatalog = (): FlightMetricCatalogTs => {
  cached ??= makeCatalog(FLIGHT_METRIC_IDS.map((metricId) => Object.freeze({
    metricId, ...IDENTITIES[metricId], frameId: FRAME, provenance: APP_SOURCE,
    solverObjective: IDENTITIES[metricId].solverObjective ?? false,
    coverage: coverage(metricId),
  })));
  return cached;
};

const toWire = (item: FlightMetricDefinitionTs) => ({
  availability_rule: item.availabilityRule,
  coverage: item.coverage.map((cell) => ({
    convention_id: cell.conventionId, reason_code: cell.reasonCode,
    source_url: cell.sourceUrl, status: cell.status,
  })),
  default_status: item.defaultStatus, definition: item.definition,
  frame_id: item.frameId, geometry_contract: item.geometryContract,
  label: item.label, metric_id: item.metricId, provenance: item.provenance,
  reference_event: item.referenceEvent, sign_rule: item.signRule,
  solver_objective: item.solverObjective, unit: item.unit,
});
const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
  : value && typeof value === "object" ? Object.fromEntries(Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => [key, stable(item)])) : value;
export const stableFlightMetricCatalogJson = (catalog: FlightMetricCatalogTs): string =>
  JSON.stringify(stable({
    definitions: catalog.definitions.map(toWire), schema_version: catalog.schemaVersion,
  }));

const exactKeys = (value: Record<string, unknown>, expected: readonly string[], label: string) => {
  if (Object.keys(value).sort().join("|") !== [...expected].sort().join("|")) {
    throw new RangeError(`${label} fields do not match v1 schema`);
  }
};
const record = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};
const text = (value: unknown, label: string): string => {
  if (typeof value !== "string" || value.trim() === "") throw new RangeError(`${label} must be nonempty`);
  return value;
};
const oneOf = <T extends string>(value: unknown, allowed: readonly T[], label: string): T => {
  const candidate = text(value, label);
  if (!allowed.includes(candidate as T)) throw new RangeError(`invalid ${label}: ${candidate}`);
  return candidate as T;
};
const httpsUrl = (value: unknown, label: string): string => {
  const candidate = text(value, label);
  let parsed: URL;
  try { parsed = new URL(candidate); } catch { throw new RangeError(`${label} must be absolute HTTPS`); }
  if (parsed.protocol !== "https:" || !parsed.hostname) throw new RangeError(`${label} must be absolute HTTPS`);
  return candidate;
};
const parseCoverage = (value: unknown): ComparisonCoverageTs => {
  const item = record(value, "coverage");
  exactKeys(item, ["convention_id", "reason_code", "source_url", "status"], "coverage");
  const status = oneOf(item.status,
    ["native", "definition_aligned", "not_comparable"] as const, "coverage status");
  return Object.freeze({
    conventionId: text(item.convention_id, "convention_id"), status,
    reasonCode: text(item.reason_code, "reason_code"),
    sourceUrl: httpsUrl(item.source_url, "source_url"),
  });
};
const parseDefinition = (value: unknown): FlightMetricDefinitionTs => {
  const item = record(value, "metric definition");
  exactKeys(item, ["availability_rule", "coverage", "default_status", "definition",
    "frame_id", "geometry_contract", "label", "metric_id", "provenance",
    "reference_event", "sign_rule", "solver_objective", "unit"], "metric definition");
  const metricId = text(item.metric_id, "metric_id") as FlightMetricId;
  if (!FLIGHT_METRIC_IDS.includes(metricId)) throw new RangeError(`unknown metric_id: ${metricId}`);
  if (!Array.isArray(item.coverage) || item.coverage.length !== 3) throw new RangeError("coverage must contain three entries");
  if (typeof item.solver_objective !== "boolean") throw new RangeError("solver_objective must be boolean");
  const parsedCoverage = item.coverage.map(parseCoverage);
  const conventions = new Set(parsedCoverage.map((cell) => cell.conventionId));
  if (!["app_native", "trackman_comparable", "foresight_comparable"]
    .every((convention) => conventions.has(convention))) {
    throw new RangeError("coverage must contain each supported convention exactly once");
  }
  return Object.freeze({
    metricId, label: text(item.label, "label"), definition: text(item.definition, "definition"),
    unit: text(item.unit, "unit"), defaultStatus: oneOf(item.default_status,
      ["input", "directly_simulated", "derived", "model_dependent", "estimated",
        "optimized", "unsupported", "unavailable"] as const, "default_status"),
    frameId: text(item.frame_id, "frame_id"), signRule: oneOf(item.sign_rule,
      ["nonnegative", "positive_right", "positive_up", "positive_down",
        "vector_components", "signed"] as const, "sign_rule"),
    referenceEvent: text(item.reference_event, "reference_event"),
    geometryContract: text(item.geometry_contract, "geometry_contract"),
    provenance: text(item.provenance, "provenance"),
    availabilityRule: text(item.availability_rule, "availability_rule"),
    solverObjective: item.solver_objective, coverage: Object.freeze(parsedCoverage),
  });
};

export function parseFlightMetricCatalog(payload: unknown): FlightMetricCatalogTs {
  const root = record(payload, "catalog");
  exactKeys(root, ["definitions", "schema_version"], "catalog");
  if (root.schema_version !== "ball-flight-metrics/v1") throw new RangeError("unsupported schema_version");
  if (!Array.isArray(root.definitions)) throw new RangeError("definitions must be an array");
  const catalog = makeCatalog(root.definitions.map(parseDefinition));
  if (catalog.definitions.length !== FLIGHT_METRIC_IDS.length ||
      new Set(catalog.definitions.map((item) => item.metricId)).size !== FLIGHT_METRIC_IDS.length) {
    throw new RangeError("catalog must contain each canonical metric exactly once");
  }
  return catalog;
}
