import type { FlightPoint } from "../model/flight";
import {
  directLaunch, type FlightExplorationTs, type WindComparisonTs,
} from "../model/flightExplorer";
import { BALL_POSITION, MPH_PER_MPS } from "../model/simulation";
import type { LaunchDirectionConvention } from "../model/launchDirection";
import { WIND_SCHEMA_VERSION, type WindScenario } from "../model/wind";
import {
  flightSampleSource,
  MAX_FLIGHT_VELOCITY_MPS,
  planFlightSamples,
  type FlightSamplePlan,
} from "../model/flightSampleInspector";

export interface FlightStudyContext {
  readonly entryMode: "direct";
  readonly ballSpeedMph: number;
  readonly launchAngleDeg: number;
  readonly launchDirectionDeg: number;
  readonly spinRpm: number;
  readonly spinAxisTiltDeg: number;
  readonly directionConvention: LaunchDirectionConvention;
  readonly windScenario: WindScenario | null;
  readonly model: "waterloo_penner";
  readonly kernelRevision: "web-rk4-10ms-sampled-v1";
}

export interface AcceptedFlightStudy {
  readonly generation: number;
  readonly context: FlightStudyContext;
  readonly contextLabel: string;
  readonly exploration: FlightExplorationTs;
  readonly calmComparison: FlightExplorationTs | null;
  readonly comparison: WindComparisonTs | null;
  readonly plan: FlightSamplePlan;
}

function snapshotPoint(point: FlightPoint): FlightPoint {
  if (!Number.isFinite(point.time) ||
      point.position.some((value) => !Number.isFinite(value)) ||
      point.velocity.some((value) => !Number.isFinite(value))) {
    throw new RangeError("flight point evidence must be finite");
  }
  if (point.velocity.some((value) => Math.abs(value) > MAX_FLIGHT_VELOCITY_MPS)) {
    throw new RangeError("flight velocity exceeds the explorer contract");
  }
  return Object.freeze({
    time: point.time,
    position: Object.freeze([...point.position]) as unknown as [number, number, number],
    velocity: Object.freeze([...point.velocity]) as unknown as [number, number, number],
  });
}

function snapshotExploration(exploration: FlightExplorationTs): FlightExplorationTs {
  if (typeof exploration !== "object" || exploration === null ||
      !Array.isArray(exploration.points) || exploration.points.length < 2 ||
      exploration.points.length > 1_002 || exploration.points.some((point) =>
        typeof point !== "object" || point === null ||
        !Array.isArray(point.position) || point.position.length !== 3 ||
        !Array.isArray(point.velocity) || point.velocity.length !== 3)) {
    throw new RangeError("flight point evidence must contain 2..1002 aligned samples");
  }
  const metricKeys = [
    "ballSpeedMph", "launchAngleDeg", "launchDirectionDeg", "launchAzimuthDeg",
    "spinRpm", "carryM", "maxHeightM", "flightTimeS", "landingAngleDeg", "lateralM",
  ] as const;
  if (typeof exploration.metrics !== "object" || exploration.metrics === null ||
      Object.keys(exploration.metrics).sort().join() !== [...metricKeys].sort().join() ||
      metricKeys.some((key) => typeof exploration.metrics[key] !== "number" ||
        !Number.isFinite(exploration.metrics[key]))) {
    throw new RangeError("flight summary evidence must be finite");
  }
  if (typeof exploration.execution !== "object" || exploration.execution === null ||
      Object.keys(exploration.execution).sort().join() !==
        ["kernelRevision", "launch", "model", "windScenario"].sort().join() ||
      typeof exploration.execution.launch !== "object" || exploration.execution.launch === null ||
      !Array.isArray(exploration.execution.launch.spinAxis) ||
      exploration.execution.launch.spinAxis.length !== 3 ||
      !(exploration.execution.windScenario === null ||
        (typeof exploration.execution.windScenario === "object" &&
          exploration.execution.windScenario !== null))) {
    throw new RangeError("flight execution provenance is incomplete");
  }
  const metrics = Object.freeze({ ...exploration.metrics });
  const launch = exploration.execution.launch;
  return Object.freeze({
    points: Object.freeze(exploration.points.map(snapshotPoint)) as FlightPoint[],
    metrics,
    execution: Object.freeze({
      model: exploration.execution?.model,
      kernelRevision: exploration.execution?.kernelRevision,
      windScenario: snapshotWind(exploration.execution?.windScenario ?? null),
      launch: Object.freeze({
        ...launch,
        spinAxis: Object.freeze([...launch.spinAxis]) as [number, number, number],
        windScenario: snapshotWind(launch.windScenario ?? null) ?? undefined,
      }),
    }) as FlightExplorationTs["execution"],
  });
}

function requireClose(actual: number, expected: number, field: string): void {
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    throw new RangeError(`${field} must be finite`);
  }
  const tolerance = 1e-7 * Math.max(1, Math.abs(actual), Math.abs(expected));
  if (Math.abs(actual - expected) > tolerance) {
    throw new RangeError(`${field} disagrees with exact trajectory evidence`);
  }
}

function validateCoherence(exploration: FlightExplorationTs, plan: FlightSamplePlan): void {
  const first = plan.rawSample(0);
  const last = plan.rawSample(plan.rawCount - 1);
  requireClose(first.timeS, 0, "first sample time");
  requireClose(first.downrangeM, BALL_POSITION[0], "launch downrange position");
  requireClose(first.heightM, BALL_POSITION[1], "launch height");
  requireClose(first.rightM, BALL_POSITION[2], "launch lateral position");
  requireClose(last.heightM, BALL_POSITION[1], "landing height");
  if (plan.samples.some((sample) => sample.heightM < BALL_POSITION[1] - 1e-7)) {
    throw new RangeError("flight evidence falls below the canonical ground plane");
  }
  requireClose(exploration.metrics.flightTimeS, last.timeS - first.timeS, "flight time");
  requireClose(
    exploration.metrics.carryM,
    Math.hypot(last.downrangeM - first.downrangeM, last.rightM - first.rightM),
    "carry",
  );
  requireClose(
    exploration.metrics.lateralM, last.rightM - first.rightM, "lateral landing offset",
  );
  const sampledHeight = Math.max(...plan.samples.map((sample) => sample.heightM - first.heightM));
  if (exploration.metrics.maxHeightM + 1e-7 < sampledHeight) {
    throw new RangeError("maximum height is below sampled trajectory evidence");
  }
  if (exploration.metrics.maxHeightM < 0 || exploration.metrics.maxHeightM > 10_000) {
    throw new RangeError("maximum height exceeds the explorer evidence envelope");
  }
  const maximumSampleGap = Math.max(...plan.samples.slice(1).map((sample, index) =>
    sample.timeS - plan.samples[index].timeS));
  const hiddenApexAllowance = maximumSampleGap * MAX_FLIGHT_VELOCITY_MPS;
  if (exploration.metrics.maxHeightM > sampledHeight + hiddenApexAllowance + 1e-7) {
    throw new RangeError("maximum height exceeds the bounded sampled-apex allowance");
  }
}

function snapshotWind(scenario: WindScenario | null): WindScenario | null {
  if (scenario === null) return null;
  if (scenario.schemaVersion !== WIND_SCHEMA_VERSION ||
      !Array.isArray(scenario.baseVelocityMps) || scenario.baseVelocityMps.length !== 3 ||
      !Array.isArray(scenario.gusts) || scenario.gusts.length > 128 ||
      typeof scenario.provenance !== "string" || scenario.gusts.some((gust) =>
        typeof gust !== "object" || gust === null || !Array.isArray(gust.peakVelocityMps) ||
        gust.peakVelocityMps.length !== 3)) throw new RangeError("wind scenario is malformed");
  const finiteValues = [
    ...scenario.baseVelocityMps, scenario.shearFractionPer10m,
    scenario.turbulenceIntensityMps, scenario.seed,
  ];
  if (finiteValues.some((value) => typeof value !== "number" || !Number.isFinite(value)) ||
      !Number.isSafeInteger(scenario.seed) || !scenario.provenance.trim() ||
      scenario.shearFractionPer10m < 0 || scenario.turbulenceIntensityMps < 0) {
    throw new RangeError("wind scenario is malformed");
  }
  const gusts = Object.freeze(scenario.gusts.map((gust) => Object.freeze({
    startTimeS: gust.startTimeS,
    durationS: gust.durationS,
    peakVelocityMps: Object.freeze([...gust.peakVelocityMps]) as [number, number, number],
  })));
  if (gusts.some((gust) => !Number.isFinite(gust.startTimeS) || gust.startTimeS < 0 ||
      !Number.isFinite(gust.durationS) || gust.durationS <= 0 ||
      gust.peakVelocityMps.some((value) => !Number.isFinite(value)))) {
    throw new RangeError("wind scenario is malformed");
  }
  return Object.freeze({
    ...scenario,
    baseVelocityMps: Object.freeze([...scenario.baseVelocityMps]) as [number, number, number],
    gusts,
  });
}

function validateContext(context: FlightStudyContext, exploration: FlightExplorationTs): void {
  const values = [
    context.ballSpeedMph, context.launchAngleDeg, context.launchDirectionDeg,
    context.spinRpm, context.spinAxisTiltDeg,
  ];
  if (values.some((value) => !Number.isFinite(value)) || context.ballSpeedMph < 1 ||
      context.ballSpeedMph > 250 || Math.abs(context.launchAngleDeg) > 89 ||
      Math.abs(context.launchDirectionDeg) > 45 || context.spinRpm < 0 ||
      context.spinRpm > 15_000 || Math.abs(context.spinAxisTiltDeg) > 60 ||
      !["app_native", "trackman_comparable"].includes(context.directionConvention) ||
      context.entryMode !== "direct" || context.model !== "waterloo_penner" ||
      context.kernelRevision !== "web-rk4-10ms-sampled-v1") {
    throw new RangeError("accepted flight context is outside the direct-entry domain");
  }
  requireClose(exploration.metrics.ballSpeedMph, context.ballSpeedMph, "ball speed context");
  requireClose(exploration.metrics.launchAngleDeg, context.launchAngleDeg, "launch angle context");
  requireClose(
    exploration.metrics.launchDirectionDeg, context.launchDirectionDeg, "direction context",
  );
  requireClose(exploration.metrics.spinRpm, context.spinRpm, "spin context");
  if (exploration.execution?.model !== context.model ||
      exploration.execution.kernelRevision !== context.kernelRevision ||
      JSON.stringify(snapshotWind(exploration.execution.windScenario)) !==
        JSON.stringify(snapshotWind(context.windScenario))) {
    throw new RangeError("flight execution provenance disagrees with accepted context");
  }
  const expectedLaunch = directLaunch({
    ballSpeedMph: context.ballSpeedMph,
    launchAngleDeg: context.launchAngleDeg,
    launchDirectionDeg: context.launchDirectionDeg,
    spinRpm: context.spinRpm,
    spinAxisTiltDeg: context.spinAxisTiltDeg,
    launchDirectionConvention: context.directionConvention,
  });
  const expectedWithWind = { ...expectedLaunch, windScenario: context.windScenario ?? undefined };
  if (JSON.stringify(exploration.execution.launch) !== JSON.stringify(expectedWithWind)) {
    throw new RangeError("flight launch fingerprint disagrees with accepted context");
  }
  requireClose(
    exploration.metrics.launchAzimuthDeg,
    exploration.metrics.launchDirectionDeg,
    "launch direction alias",
  );
  const velocity = exploration.points[0].velocity;
  const horizontal = Math.hypot(velocity[0], velocity[2]);
  requireClose(Math.hypot(...velocity) * MPH_PER_MPS, context.ballSpeedMph, "raw launch speed");
  requireClose(
    Math.atan2(velocity[1], horizontal) * 180 / Math.PI,
    context.launchAngleDeg,
    "raw launch angle",
  );
  requireClose(
    Math.atan2(velocity[2], velocity[0]) * 180 / Math.PI,
    context.launchDirectionDeg,
    "raw launch direction",
  );
  const landingVelocity = exploration.points[exploration.points.length - 1].velocity;
  const landingHorizontal = Math.hypot(landingVelocity[0], landingVelocity[2]);
  const landingAngle = landingHorizontal > 0.1
    ? Math.atan2(-landingVelocity[1], landingHorizontal) * 180 / Math.PI : 90;
  requireClose(exploration.metrics.landingAngleDeg, landingAngle, "landing angle");
}

export function flightContextLabel(context: FlightStudyContext): string {
  return [
    `direct ${context.ballSpeedMph.toFixed(2)} mph`,
    `launch ${context.launchAngleDeg.toFixed(2)} deg`,
    `direction ${context.launchDirectionDeg.toFixed(2)} deg (${context.directionConvention})`,
    `spin ${context.spinRpm.toFixed(0)} rpm`,
    `axis ${context.spinAxisTiltDeg.toFixed(2)} deg`,
    context.windScenario === null ? "calm" :
      `wind [${context.windScenario.baseVelocityMps.map((value) => value.toFixed(3)).join(", ")}] m/s (${context.windScenario.provenance})`,
    `model ${context.model}`, `kernel ${context.kernelRevision}`,
  ].join("; ");
}

export function buildAcceptedFlightStudy(
  generation: number,
  context: FlightStudyContext,
  explorationInput: FlightExplorationTs,
  comparisonInput: WindComparisonTs | null,
): AcceptedFlightStudy {
  if (!Number.isSafeInteger(generation) || generation < 1) {
    throw new RangeError("accepted flight generation must be a positive safe integer");
  }
  const exploration = snapshotExploration(explorationInput);
  const plan = planFlightSamples(flightSampleSource(exploration));
  validateCoherence(exploration, plan);
  validateContext(context, exploration);
  const windScenario = snapshotWind(context.windScenario);
  const calmComparison = comparisonInput ? snapshotExploration(comparisonInput.calm) : null;
  if (comparisonInput) {
    const wind = snapshotExploration(comparisonInput.wind);
    if (JSON.stringify(wind) !== JSON.stringify(exploration)) {
      throw new RangeError("wind comparison primary disagrees with accepted exploration");
    }
    if (windScenario === null || JSON.stringify(snapshotWind(comparisonInput.scenario)) !==
        JSON.stringify(windScenario)) {
      throw new RangeError("wind comparison scenario disagrees with accepted context");
    }
    const calmPlan = planFlightSamples(flightSampleSource(calmComparison as FlightExplorationTs));
    validateCoherence(calmComparison as FlightExplorationTs, calmPlan);
    validateContext(
      { ...context, windScenario: null }, calmComparison as FlightExplorationTs,
    );
    const deltaKeys = ["carryM", "maxHeightM", "flightTimeS", "landingAngleDeg", "lateralM"] as const;
    if (typeof comparisonInput.deltas !== "object" || comparisonInput.deltas === null ||
        Object.keys(comparisonInput.deltas).sort().join() !== [...deltaKeys].sort().join()) {
      throw new RangeError("wind deltas must contain the exact comparison metrics");
    }
    for (const key of deltaKeys) {
      const delta = comparisonInput.deltas[key];
      if (!Number.isFinite(delta)) throw new RangeError("wind deltas must be finite");
      requireClose(delta, exploration.metrics[key] - comparisonInput.calm.metrics[key], `wind ${key}`);
    }
  } else if (windScenario !== null) {
    throw new RangeError("enabled wind context requires a cohesive comparison");
  }
  const comparison = comparisonInput ? Object.freeze({
    calm: calmComparison as FlightExplorationTs,
    wind: exploration,
    deltas: Object.freeze({ ...comparisonInput.deltas }),
    scenario: windScenario as WindScenario,
  }) : null;
  return Object.freeze({
    generation,
    context: Object.freeze({ ...context, windScenario }),
    contextLabel: flightContextLabel({ ...context, windScenario }),
    exploration, calmComparison, comparison, plan,
  });
}
