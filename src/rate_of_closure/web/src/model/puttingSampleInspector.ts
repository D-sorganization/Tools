import type { PuttResult } from "./putting";

export const MAX_PUTTING_RAW_SAMPLES = 30_001;
export const MAX_PUTTING_DISPLAY_SAMPLES = 1_024;
export const DEFAULT_PUTTING_HIT_RADIUS_PX = 12;

export type PuttingPhase = "skid" | "pure-roll";
export type PuttingNavigation = "previous" | "next" | "home" | "end" | "clear";

export function puttingContextLabel(
  putter: { name: string; headMassKg: number; loftDeg: number; cor: number },
  speed: number, stimp: number, grade: number, aspect: number, hole: number,
): string {
  return [
    `putter ${putter.name} (${putter.headMassKg.toFixed(3)} kg, ${putter.loftDeg.toFixed(1)} deg, COR ${putter.cor.toFixed(2)})`,
    `resolved speed ${speed.toFixed(3)} m/s`, `stimp ${stimp.toFixed(2)} ft`,
    `grade ${grade.toFixed(2)}%`, `aspect ${aspect.toFixed(1)} deg`,
    `hole ${hole.toFixed(2)} m`, "kernel RK4-2ms-v1",
  ].join("; ");
}

export interface PuttingSampleSource {
  readonly path_x_m: readonly number[];
  readonly path_y_m: readonly number[];
  readonly speeds_mps: readonly number[];
  readonly times_s: readonly number[];
  readonly skid_end_index: number;
}

export interface PuttingDisplaySample {
  readonly rawIndex: number;
  readonly timeS: number;
  readonly cumulativeDistanceM: number;
  readonly xM: number;
  readonly yM: number;
  readonly speedMps: number;
  readonly phase: PuttingPhase;
}

export interface PuttingSamplePlan {
  readonly source: PuttingSampleSource;
  readonly rawCount: number;
  readonly displayedCount: number;
  readonly skidEndIndex: number;
  readonly cumulativeDistanceM: readonly number[];
  readonly samples: readonly PuttingDisplaySample[];
  readonly displayedRawIndices: readonly number[];
  readonly skidPolylineIndices: readonly number[];
  readonly pureRollPolylineIndices: readonly number[];
  readonly rawSample: (rawIndex: number) => PuttingDisplaySample;
}

export function puttingSampleSource(result: PuttResult): PuttingSampleSource {
  return {
    path_x_m: result.pathXM,
    path_y_m: result.pathYM,
    speeds_mps: result.speedsMps,
    times_s: result.timesS,
    skid_end_index: result.skidEndIndex,
  };
}

export function snapshotPuttingResult(result: PuttResult): PuttResult {
  const copy = {
    ...result,
    pathXM: Object.freeze([...result.pathXM]), pathYM: Object.freeze([...result.pathYM]),
    speedsMps: Object.freeze([...result.speedsMps]), timesS: Object.freeze([...result.timesS]),
  };
  return Object.freeze(copy) as PuttResult;
}

function finiteArray(values: unknown, field: string): readonly number[] {
  if (!Array.isArray(values) || values.some((value) =>
    typeof value !== "number" || !Number.isFinite(value))) {
    throw new RangeError(`${field} must contain finite numbers`);
  }
  return Object.freeze([...values]);
}

function validatedSource(source: PuttingSampleSource): PuttingSampleSource {
  if (typeof source !== "object" || source === null || Array.isArray(source)) {
    throw new RangeError("putting planner requires aligned sample evidence");
  }
  const pathXM = finiteArray(source.path_x_m, "path_x_m");
  const pathYM = finiteArray(source.path_y_m, "path_y_m");
  const speeds = finiteArray(source.speeds_mps, "speeds_mps");
  const times = finiteArray(source.times_s, "times_s");
  const count = times.length;
  if (count < 1 || count > MAX_PUTTING_RAW_SAMPLES) {
    throw new RangeError(`putting evidence must contain 1..${MAX_PUTTING_RAW_SAMPLES} samples`);
  }
  if ([pathXM, pathYM, speeds].some((values) => values.length !== count)) {
    throw new RangeError("putting sample arrays must have equal lengths");
  }
  if (speeds.some((speed) => speed < 0)) {
    throw new RangeError("putting sample speeds must be nonnegative");
  }
  if (times[0] < 0 || times.slice(1).some((time, index) => time <= times[index])) {
    throw new RangeError("putting sample times must be nonnegative and strictly increasing");
  }
  if (!Number.isSafeInteger(source.skid_end_index) ||
      source.skid_end_index < 0 || source.skid_end_index >= count) {
    throw new RangeError("skid_end_index must identify the first pure-roll sample");
  }
  return Object.freeze({
    path_x_m: pathXM, path_y_m: pathYM, speeds_mps: speeds, times_s: times,
    skid_end_index: source.skid_end_index,
  });
}

function stableExtrema(values: readonly number[]): readonly [number, number] {
  let minimum = 0; let maximum = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < values[minimum]) minimum = index;
    if (values[index] > values[maximum]) maximum = index;
  }
  return [minimum, maximum];
}

function displayIndices(source: PuttingSampleSource, cap: number): readonly number[] {
  const count = source.times_s.length;
  const mandatory = new Set([0, count - 1, source.skid_end_index]);
  [source.path_x_m, source.path_y_m, source.speeds_mps]
    .forEach((values) => stableExtrema(values).forEach((index) => mandatory.add(index)));
  if (mandatory.size > cap) {
    throw new RangeError("putting display cap cannot retain all scientific landmarks");
  }
  if (count <= cap) return Object.freeze(Array.from({ length: count }, (_, index) => index));
  const available = Array.from({ length: count }, (_, index) => index)
    .filter((index) => !mandatory.has(index));
  const needed = cap - mandatory.size;
  if (needed === 1) mandatory.add(available[Math.floor(available.length / 2)]);
  else if (needed > 1) {
    for (let position = 0; position < needed; position += 1) {
      const numerator = 2 * position * (available.length - 1) + (needed - 1);
      mandatory.add(available[Math.floor(numerator / (2 * (needed - 1)))]);
    }
  }
  return Object.freeze([...mandatory].sort((left, right) => left - right));
}

export function planPuttingSamples(
  input: PuttingSampleSource, cap = MAX_PUTTING_DISPLAY_SAMPLES,
): PuttingSamplePlan {
  if (!Number.isSafeInteger(cap) || cap < 3 || cap > MAX_PUTTING_DISPLAY_SAMPLES) {
    throw new RangeError("putting display cap must be an integer from 3 through 1024");
  }
  const source = validatedSource(input);
  const cumulative = new Array<number>(source.times_s.length).fill(0);
  for (let index = 1; index < cumulative.length; index += 1) {
    const step = Math.hypot(
      source.path_x_m[index] - source.path_x_m[index - 1],
      source.path_y_m[index] - source.path_y_m[index - 1],
    );
    cumulative[index] = cumulative[index - 1] + step;
    if (!Number.isFinite(step) || !Number.isFinite(cumulative[index])) {
      throw new RangeError("putting cumulative distance must remain finite");
    }
  }
  const cumulativeDistanceM = Object.freeze(cumulative);
  const envelope = [
    Math.max(...source.path_x_m) + 0.3 - (Math.min(...source.path_x_m) - 0.3),
    2 * Math.max(0.3, ...source.path_y_m.map(Math.abs)),
    Math.max(...source.speeds_mps) * 1.08,
  ];
  if (envelope.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("putting display envelope must remain finite and positive");
  }
  const rawSample = (rawIndex: number): PuttingDisplaySample => {
    if (!Number.isSafeInteger(rawIndex) || rawIndex < 0 || rawIndex >= source.times_s.length) {
      throw new RangeError("raw sample index is outside the accepted result");
    }
    return Object.freeze({
      rawIndex, timeS: source.times_s[rawIndex],
      cumulativeDistanceM: cumulativeDistanceM[rawIndex],
      xM: source.path_x_m[rawIndex], yM: source.path_y_m[rawIndex],
      speedMps: source.speeds_mps[rawIndex],
      phase: rawIndex < source.skid_end_index ? "skid" : "pure-roll",
    });
  };
  const displayedRawIndices = displayIndices(source, cap);
  const samples = Object.freeze(displayedRawIndices.map(rawSample));
  const skidPolylineIndices = Object.freeze(source.skid_end_index === 0 ? [] :
    displayedRawIndices.filter((index) => index <= source.skid_end_index));
  const pureRollPolylineIndices = Object.freeze(
    displayedRawIndices.filter((index) => index >= source.skid_end_index),
  );
  return Object.freeze({
    source, rawCount: source.times_s.length, displayedCount: samples.length,
    skidEndIndex: source.skid_end_index, cumulativeDistanceM, samples,
    displayedRawIndices, skidPolylineIndices, pureRollPolylineIndices, rawSample,
  });
}

export function validatePuttingResultSummary(result: PuttResult, plan: PuttingSamplePlan): void {
  const finite = [result.skidDistanceM, result.totalDistanceM, result.timeS, result.breakM];
  if (finite.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new RangeError("putting result summary must contain finite scalars");
  }
  if (typeof result.holed !== "boolean") throw new RangeError("holed must be boolean");
  [result.speedAtHoleMps, result.marginMps, result.missDistanceM].forEach((value) => {
    if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
      throw new RangeError("putting nullable result scalars must be finite or null");
    }
    if (value !== null && value < 0) throw new RangeError("putting nullable summaries must be nonnegative");
  });
  if ([result.totalDistanceM, result.skidDistanceM, result.timeS].some((value) => value < 0)) {
    throw new RangeError("putting distance and time summaries must be nonnegative");
  }
  const close = (left: number, right: number) =>
    Math.abs(left - right) <= 1e-9 * Math.max(1, Math.abs(left), Math.abs(right));
  const last = plan.rawCount - 1;
  const matches = close(result.totalDistanceM, plan.cumulativeDistanceM[last])
    && close(result.skidDistanceM, plan.cumulativeDistanceM[plan.skidEndIndex])
    && close(result.timeS, plan.source.times_s[last])
    && close(result.breakM, plan.source.path_y_m[last]);
  if (!matches) throw new RangeError("putting summary must match exact raw sample evidence");
  const coherent = result.holed
    ? result.speedAtHoleMps !== null && result.marginMps !== null && result.missDistanceM === null
    : result.marginMps === null && result.missDistanceM !== null;
  if (!coherent) throw new RangeError("putting capture summaries are internally inconsistent");
}

export function navigatePuttingSamples(
  plan: PuttingSamplePlan, currentRawIndex: number | null, command: PuttingNavigation,
): number | null {
  if (!["previous", "next", "home", "end", "clear"].includes(command)) {
    throw new RangeError("unknown putting sample navigation command");
  }
  const indices = plan.displayedRawIndices;
  if (command === "clear" || indices.length === 0) return null;
  if (command === "home") return indices[0];
  if (command === "end") return indices[indices.length - 1];
  const position = indices.indexOf(currentRawIndex ?? -1);
  if (position < 0) return command === "next" ? indices[0] : indices[indices.length - 1];
  return command === "next"
    ? indices[Math.min(position + 1, indices.length - 1)]
    : indices[Math.max(position - 1, 0)];
}

export function nearestPuttingSample(
  projected: readonly (readonly [number, number, number])[],
  pointerPx: readonly [number, number],
  hitRadiusPx = DEFAULT_PUTTING_HIT_RADIUS_PX,
): number | null {
  if (pointerPx.some((value) => !Number.isFinite(value))) {
    throw new RangeError("pointer must contain two finite pixel coordinates");
  }
  if (!Number.isFinite(hitRadiusPx) || hitRadiusPx <= 0 || hitRadiusPx > 100) {
    throw new RangeError("hit radius must be a finite positive pixel distance");
  }
  let nearest: readonly [number, number] | null = null;
  for (const [rawIndex, x, y] of projected) {
    if (!Number.isSafeInteger(rawIndex) || !Number.isFinite(x) || !Number.isFinite(y)) {
      throw new RangeError("projected samples must contain finite pixel coordinates");
    }
    const distanceSquared = (x - pointerPx[0]) ** 2 + (y - pointerPx[1]) ** 2;
    const candidate: readonly [number, number] = [distanceSquared, rawIndex];
    if (nearest === null || candidate[0] < nearest[0] ||
        (candidate[0] === nearest[0] && candidate[1] < nearest[1])) nearest = candidate;
  }
  return nearest !== null && nearest[0] <= hitRadiusPx ** 2 ? nearest[1] : null;
}
