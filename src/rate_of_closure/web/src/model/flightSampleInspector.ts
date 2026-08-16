import type { FlightExplorationTs } from "./flightExplorer";

export const MAX_FLIGHT_SAMPLES = 1_002;
export const MAX_FLIGHT_TIME_S = 10.001;
export const MAX_FLIGHT_POSITION_M = 10_000;
export const MAX_FLIGHT_VELOCITY_MPS = 1_000;
export const DEFAULT_FLIGHT_HIT_RADIUS_PX = 12;

export type FlightPhase = "launch" | "ascent" | "apex" | "descent" | "landing";
export type FlightNavigation = "previous" | "next" | "home" | "end" | "clear";
export type FlightCohort = "current";

export interface FlightSampleSource {
  readonly timesS: readonly number[];
  readonly positionsM: readonly (readonly [number, number, number])[];
}

export interface FlightDisplaySample {
  readonly rawIndex: number;
  readonly timeS: number;
  readonly downrangeM: number;
  readonly heightM: number;
  readonly rightM: number;
  readonly phase: FlightPhase;
}

export interface FlightSamplePlan {
  readonly source: FlightSampleSource;
  readonly samples: readonly FlightDisplaySample[];
  readonly rawCount: number;
  readonly apexRawIndex: number;
  readonly rawSample: (index: number) => FlightDisplaySample;
}

export interface FlightSampleSelection {
  readonly cohort: FlightCohort;
  readonly rawIndex: number;
}

function finite(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${field} must be a finite number`);
  }
  return value;
}

export function flightSampleSource(exploration: FlightExplorationTs): FlightSampleSource {
  if (typeof exploration !== "object" || exploration === null || !Array.isArray(exploration.points)) {
    throw new RangeError("flight sample source must be a flight exploration");
  }
  if (exploration.points.length < 2 || exploration.points.length > MAX_FLIGHT_SAMPLES) {
    throw new RangeError(`flight evidence must contain 2..${MAX_FLIGHT_SAMPLES} samples`);
  }
  if (exploration.points.some((point) => typeof point !== "object" || point === null ||
      !Array.isArray(point.position) || point.position.length !== 3)) {
    throw new RangeError("flight positions must have shape (N, 3)");
  }
  return {
    timesS: exploration.points.map((point) => point.time),
    positionsM: exploration.points.map((point) => Object.freeze([...point.position]) as
      readonly [number, number, number]),
  };
}

export function planFlightSamples(input: FlightSampleSource): FlightSamplePlan {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new RangeError("flight planner requires aligned sample evidence");
  }
  if (!Array.isArray(input.timesS) || !Array.isArray(input.positionsM)) {
    throw new RangeError("flight planner requires aligned sample evidence");
  }
  const count = input.timesS.length;
  if (count < 2 || count > MAX_FLIGHT_SAMPLES) {
    throw new RangeError(`flight evidence must contain 2..${MAX_FLIGHT_SAMPLES} samples`);
  }
  if (input.positionsM.length !== count ||
      input.positionsM.some((row) => !Array.isArray(row) || row.length !== 3)) {
    throw new RangeError("flight positions must have shape (N, 3)");
  }
  const times = Object.freeze(input.timesS.map((value) => finite(value, "timesS")));
  const positions = Object.freeze(input.positionsM.map((row) => {
    if (!Array.isArray(row) || row.length !== 3) {
      throw new RangeError("flight positions must have shape (N, 3)");
    }
    return Object.freeze(row.map((value) => finite(value, "positionsM"))) as unknown as
      readonly [number, number, number];
  }));
  if (times[0] < 0 || times.slice(1).some((value, index) => value <= times[index])) {
    throw new RangeError("flight sample times must be nonnegative and strictly increasing");
  }
  if (times[times.length - 1] > MAX_FLIGHT_TIME_S) {
    throw new RangeError("flight sample time exceeds the explorer contract");
  }
  if (positions.some((row) => row.some((value) => Math.abs(value) > MAX_FLIGHT_POSITION_M))) {
    throw new RangeError("flight position exceeds the explorer contract");
  }
  let apexRawIndex = 0;
  positions.forEach((position, index) => {
    if (position[1] > positions[apexRawIndex][1]) apexRawIndex = index;
  });
  const last = times.length - 1;
  const samples = Object.freeze(times.map((timeS, rawIndex): FlightDisplaySample => {
    let phase: FlightPhase;
    if (rawIndex === 0) phase = "launch";
    else if (rawIndex === last) phase = "landing";
    else if (rawIndex === apexRawIndex) phase = "apex";
    else if (rawIndex < apexRawIndex) phase = "ascent";
    else phase = "descent";
    return Object.freeze({
      rawIndex, timeS, downrangeM: positions[rawIndex][0],
      heightM: positions[rawIndex][1], rightM: positions[rawIndex][2], phase,
    });
  }));
  const source = Object.freeze({ timesS: times, positionsM: positions });
  const rawSample = (index: number): FlightDisplaySample => {
    if (!Number.isSafeInteger(index) || index < 0 || index >= samples.length) {
      throw new RangeError("raw sample index is outside the accepted flight");
    }
    return samples[index];
  };
  return Object.freeze({ source, samples, rawCount: samples.length, apexRawIndex, rawSample });
}

export function navigateFlightSamples(
  plan: FlightSamplePlan, current: number | null, command: FlightNavigation,
): number | null {
  if (!["previous", "next", "home", "end", "clear"].includes(command)) {
    throw new RangeError("unknown flight sample navigation command");
  }
  if (command === "clear") return null;
  if (command === "home") return 0;
  if (command === "end") return plan.rawCount - 1;
  if (!Number.isSafeInteger(current) || current === null || current < 0 || current >= plan.rawCount) {
    return command === "next" ? 0 : plan.rawCount - 1;
  }
  return command === "next" ? Math.min(current + 1, plan.rawCount - 1) : Math.max(current - 1, 0);
}

export function nearestFlightSample(
  plan: FlightSamplePlan,
  projected: readonly (readonly [FlightCohort, number, number, number])[],
  pointerPx: readonly [number, number],
  hitRadiusPx = DEFAULT_FLIGHT_HIT_RADIUS_PX,
): FlightSampleSelection | null {
  const [pointerX, pointerY] = pointerPx.map((value) => finite(value, "pointer"));
  const radius = finite(hitRadiusPx, "hit radius");
  if (radius <= 0 || radius > 100) throw new RangeError("hit radius must be a positive pixel distance");
  let nearest: readonly [number, number] | null = null;
  const seen = new Set<number>();
  for (const [cohort, rawIndex, rawX, rawY] of projected) {
    if (cohort !== "current") throw new RangeError("calm comparison samples are not selectable");
    if (!Number.isSafeInteger(rawIndex) || rawIndex < 0) {
      throw new RangeError("projected raw index must be a nonnegative integer");
    }
    if (rawIndex >= plan.rawCount || seen.has(rawIndex)) {
      throw new RangeError("projected raw indices must be unique and in range");
    }
    seen.add(rawIndex);
    const x = finite(rawX, "projected x");
    const y = finite(rawY, "projected y");
    const candidate: readonly [number, number] = [Math.hypot(x - pointerX, y - pointerY), rawIndex];
    if (nearest === null || candidate[0] < nearest[0] ||
        (candidate[0] === nearest[0] && candidate[1] < nearest[1])) nearest = candidate;
  }
  if (seen.size !== plan.rawCount) {
    throw new RangeError("projected samples must cover the complete primary plan");
  }
  return nearest !== null && nearest[0] <= radius
    ? Object.freeze({ cohort: "current", rawIndex: nearest[1] }) : null;
}
