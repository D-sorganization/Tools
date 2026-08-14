/** Swept rigid-wedge clearance against the web app's y=0 ground plane. */

import type { ClubSpec } from "./club";
import type { ImpactScenario } from "./impact";
import { add, cross, dot, norm, scale, sub, type Vec3 } from "./impactPhysics";
import { applyRotation, slerpRotation, type Mat3 } from "./rotation";
import type { SimulationRunTs } from "./simulation";
import {
  representativeWedgeForClub, wedgeContactCandidates, wedgeFaceContactPointM,
  type RepresentativeWedge, type WedgeContactCandidate, type WedgeContactFeature,
} from "./wedgeGroundGeometry";

export const WEDGE_GROUND_CLEARANCE_FORMAT = "upstreamdrift.wedge-ground-clearance/v1";
const SUBDIVISIONS = 8;
const BISECTION_ITERATIONS = 48;
const TIME_TOLERANCE_S = 1e-9;
const LIMITATIONS = "Rigid geometric contact against a static plane only; no turf deformation, soil force, divot depth, friction impulse, or injury inference is modeled.";

export type ContactSequence = "ball_first" | "ground_first" | "simultaneous" |
  "ball_only" | "ground_only_miss" | "no_contact_miss";

interface RegisteredSample {
  t: number;
  position: Vec3;
  velocity: Vec3;
  angularVelocity: Vec3;
  rotation: Mat3;
}

export interface WedgeClearanceSampleTs {
  timeS: number;
  minimumClearanceM: number;
  feature: WedgeContactFeature;
  worldPointM: Vec3;
}

export interface WedgeGroundContactEventTs {
  timeS: number;
  feature: WedgeContactFeature;
  worldPointM: Vec3;
  normalVelocityMps: number;
  tangentialVelocityMps: Vec3;
  poseHeadToGround: [number[], number[], number[], number[]];
}

export interface WedgeGroundClearancePayloadTs {
  format: typeof WEDGE_GROUND_CLEARANCE_FORMAT;
  frameId: "ground_frame:x_target,y_up,z_right";
  units: { angle: "deg"; angularVelocity: "rad/s"; length: "m"; time: "s"; velocity: "m/s" };
  sequence: ContactSequence;
  ballContactTimeS: number | null;
  firstGroundContact: WedgeGroundContactEventTs | null;
  metrics: {
    bounceUtilizationMarginDeg: number | null;
    deliveredBounceDegAtBall: number | null;
    groundAfterBallTimeMarginS: number | null;
    leadingEdgeClearanceAtBallM: number | null;
    minimumPreBallClearanceM: number | null;
    pathProjectedEffectiveBounceDegAtBall: number | null;
    referenceAoaDegAtBall: number | null;
    soleEntryMarginM: number | null;
  };
  lowPoint: { feature: WedgeContactFeature; timeS: number; worldPointM: Vec3 };
  envelope: WedgeClearanceSampleTs[];
  geometryBasis: string;
  provenance: string;
  limitations: string;
}

function registeredSweep(
  run: SimulationRunTs, scenario: ImpactScenario, wedge: RepresentativeWedge,
): RegisteredSample[] {
  const facePoint = wedgeFaceContactPointM(
    wedge, scenario.impactOffsetToeMm / 1000, scenario.impactOffsetHighMm / 1000,
  );
  const localShift = sub([
    scenario.comToFaceMm / 1000,
    scenario.impactOffsetHighMm / 1000,
    scenario.impactOffsetToeMm / 1000,
  ], facePoint);
  return run.swing.map((sample) => {
    const worldShift = applyRotation(sample.rotation, localShift);
    return {
      t: sample.t,
      position: add(sample.position, worldShift),
      velocity: add(sample.velocity, cross(sample.angularVelocity, worldShift)),
      angularVelocity: sample.angularVelocity,
      rotation: sample.rotation,
    };
  });
}

function interpolate(samples: RegisteredSample[], timeS: number): RegisteredSample {
  if (timeS <= samples[0].t) return samples[0];
  if (timeS >= samples[samples.length - 1].t) return samples[samples.length - 1];
  let upper = 1;
  while (samples[upper].t < timeS) upper += 1;
  const first = samples[upper - 1];
  const second = samples[upper];
  const alpha = (timeS - first.t) / (second.t - first.t);
  const blend = (a: Vec3, b: Vec3): Vec3 => add(a, scale(sub(b, a), alpha));
  return {
    t: timeS,
    position: blend(first.position, second.position),
    velocity: blend(first.velocity, second.velocity),
    angularVelocity: blend(first.angularVelocity, second.angularVelocity),
    rotation: slerpRotation(first.rotation, second.rotation, alpha),
  };
}

const worldPoint = (sample: RegisteredSample, candidate: WedgeContactCandidate): Vec3 =>
  add(sample.position, applyRotation(sample.rotation, candidate.localPointM));

function candidateState(
  samples: RegisteredSample[], candidates: WedgeContactCandidate[], timeS: number,
) {
  const sample = interpolate(samples, timeS);
  const points = candidates.map((candidate) => worldPoint(sample, candidate));
  return { sample, points, clearances: points.map((point) => point[1]) };
}

function sweepTimes(samples: RegisteredSample[]): number[] {
  const times: number[] = [];
  for (let index = 0; index < samples.length - 1; index += 1) {
    const start = samples[index].t;
    const step = (samples[index + 1].t - start) / SUBDIVISIONS;
    for (let subdivision = 0; subdivision < SUBDIVISIONS; subdivision += 1) {
      times.push(start + subdivision * step);
    }
  }
  times.push(samples[samples.length - 1].t);
  return times;
}

function crossingTime(
  samples: RegisteredSample[], candidate: WedgeContactCandidate, lower: number, upper: number,
): number {
  for (let iteration = 0; iteration < BISECTION_ITERATIONS; iteration += 1) {
    const midpoint = (lower + upper) / 2;
    if (worldPoint(interpolate(samples, midpoint), candidate)[1] > 0) lower = midpoint;
    else upper = midpoint;
  }
  return upper;
}

function firstContact(
  samples: RegisteredSample[], candidates: WedgeContactCandidate[], times: number[],
  clearanceRows: number[][],
): WedgeGroundContactEventTs | null {
  let timeS: number | null = null;
  let candidateIndex = -1;
  const initialMinimum = Math.min(...clearanceRows[0]);
  if (initialMinimum <= 1e-10) {
    timeS = times[0];
    candidateIndex = clearanceRows[0].indexOf(initialMinimum);
  } else {
    for (let row = 1; row < times.length; row += 1) {
      for (let index = 0; index < candidates.length; index += 1) {
        if (clearanceRows[row - 1][index] > 0 && clearanceRows[row][index] <= 0) {
          const crossing = crossingTime(samples, candidates[index], times[row - 1], times[row]);
          if (timeS === null || crossing < timeS) {
            timeS = crossing;
            candidateIndex = index;
          }
        }
      }
    }
  }
  if (timeS === null) return null;
  const sample = interpolate(samples, timeS);
  const point = worldPoint(sample, candidates[candidateIndex]);
  const velocity = add(sample.velocity, cross(sample.angularVelocity, sub(point, sample.position)));
  return {
    timeS,
    feature: candidates[candidateIndex].feature,
    worldPointM: point,
    normalVelocityMps: velocity[1],
    tangentialVelocityMps: [velocity[0], 0, velocity[2]],
    poseHeadToGround: [
      [...sample.rotation[0], sample.position[0]],
      [...sample.rotation[1], sample.position[1]],
      [...sample.rotation[2], sample.position[2]],
      [0, 0, 0, 1],
    ],
  };
}

function contactSequence(ballTime: number | null, event: WedgeGroundContactEventTs | null): ContactSequence {
  if (ballTime === null) return event === null ? "no_contact_miss" : "ground_only_miss";
  if (event === null) return "ball_only";
  const difference = event.timeS - ballTime;
  if (Math.abs(difference) <= TIME_TOLERANCE_S) return "simultaneous";
  return difference > 0 ? "ball_first" : "ground_first";
}

function ballMetrics(
  wedge: RepresentativeWedge, samples: RegisteredSample[], candidates: WedgeContactCandidate[],
  envelope: WedgeClearanceSampleTs[], ballTime: number | null,
) {
  const unavailable = {
    bounceUtilizationMarginDeg: null, deliveredBounceDegAtBall: null,
    leadingEdgeClearanceAtBallM: null, minimumPreBallClearanceM: null,
    pathProjectedEffectiveBounceDegAtBall: null, referenceAoaDegAtBall: null,
    soleEntryMarginM: null,
  };
  if (ballTime === null) return unavailable;
  const state = candidateState(samples, candidates, ballTime);
  const leading = state.clearances.filter((_, index) => candidates[index].feature.startsWith("leading_edge"));
  const sole = state.clearances.filter((_, index) => !candidates[index].feature.startsWith("leading_edge"));
  const minimumPreBall = Math.min(
    ...envelope.filter((entry) => entry.timeS <= ballTime).map((entry) => entry.minimumClearanceM),
    ...state.clearances,
  );
  const bounce = wedge.bounceDeg * Math.PI / 180;
  const worldSole = applyRotation(state.sample.rotation, [
    -wedge.soleWidthM * Math.cos(bounce), wedge.soleWidthM * Math.sin(bounce), 0,
  ]);
  const delivered = Math.atan2(worldSole[1], Math.hypot(worldSole[0], worldSole[2])) * 180 / Math.PI;
  const verticalVelocity = state.sample.velocity[1];
  const horizontalVelocity: Vec3 = [state.sample.velocity[0], 0, state.sample.velocity[2]];
  const horizontalSpeed = norm(horizontalVelocity);
  if (horizontalSpeed <= 1e-12) return {
    ...unavailable, deliveredBounceDegAtBall: delivered,
    leadingEdgeClearanceAtBallM: Math.min(...leading), minimumPreBallClearanceM: minimumPreBall,
    soleEntryMarginM: Math.min(...sole),
  };
  const path = scale(horizontalVelocity, 1 / horizontalSpeed);
  const trailingAlongPath = dot([worldSole[0], 0, worldSole[2]], scale(path, -1));
  const effective = Math.atan2(worldSole[1], trailingAlongPath) * 180 / Math.PI;
  const aoa = Math.atan2(verticalVelocity, horizontalSpeed) * 180 / Math.PI;
  return {
    bounceUtilizationMarginDeg: effective + aoa,
    deliveredBounceDegAtBall: delivered,
    leadingEdgeClearanceAtBallM: Math.min(...leading),
    minimumPreBallClearanceM: minimumPreBall,
    pathProjectedEffectiveBounceDegAtBall: effective,
    referenceAoaDegAtBall: aoa,
    soleEntryMarginM: Math.min(...sole),
  };
}

/** Build the canonical visualization payload; non-wedges deliberately return null. */
export function wedgeGroundClearance(
  run: SimulationRunTs, scenario: ImpactScenario, club: ClubSpec,
): WedgeGroundClearancePayloadTs | null {
  if (run.swing.length < 2) throw new RangeError("run must retain at least two swing samples");
  const wedge = representativeWedgeForClub(club);
  if (wedge === null) return null;
  const samples = registeredSweep(run, scenario, wedge);
  const candidates = wedgeContactCandidates(wedge);
  const times = sweepTimes(samples);
  const states = times.map((time) => candidateState(samples, candidates, time));
  const envelope = states.map((state, index) => {
    const minimum = Math.min(...state.clearances);
    const candidateIndex = state.clearances.indexOf(minimum);
    return {
      timeS: times[index], minimumClearanceM: minimum,
      feature: candidates[candidateIndex].feature, worldPointM: state.points[candidateIndex],
    };
  });
  const event = firstContact(samples, candidates, times, states.map((state) => state.clearances));
  const lowPoint = envelope.reduce((lowest, current) =>
    current.minimumClearanceM < lowest.minimumClearanceM ? current : lowest);
  const ballTime = run.impactTimeS;
  const metrics = ballMetrics(wedge, samples, candidates, envelope, ballTime);
  return {
    format: WEDGE_GROUND_CLEARANCE_FORMAT,
    frameId: "ground_frame:x_target,y_up,z_right",
    units: { angle: "deg", angularVelocity: "rad/s", length: "m", time: "s", velocity: "m/s" },
    sequence: contactSequence(ballTime, event),
    ballContactTimeS: ballTime,
    firstGroundContact: event,
    metrics: {
      ...metrics,
      groundAfterBallTimeMarginS: ballTime === null || event === null ? null : event.timeS - ballTime,
    },
    lowPoint: { feature: lowPoint.feature, timeS: lowPoint.timeS, worldPointM: lowPoint.worldPointM },
    envelope,
    geometryBasis: "canonical_wedge_face_contact_registration",
    provenance: `${wedge.geometryBasis}. ${wedge.uncertaintyNote}`,
    limitations: `${run.impactOutcome.geometryLimitations} ${LIMITATIONS}`,
  };
}
