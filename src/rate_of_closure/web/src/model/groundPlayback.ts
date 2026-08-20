/** Phase-safe playback semantics for strict imported ground results. */

import type {
  FlightToGroundResult,
  GroundPhase,
  GroundTrajectoryPoint,
  GroundVec3,
} from "./flightGroundTypes";
import type { GroundRegionalExecutionResult } from "./groundRegionalExecution";

export interface GroundPlaybackFrame {
  readonly timeS: number;
  readonly elapsedS: number;
  readonly positionM: GroundVec3;
  readonly phase: GroundPhase;
  readonly lowerIndex: number;
  readonly interpolationFraction: number;
  readonly isTerminal: boolean;
}

const VALID_PHASES: readonly GroundPhase[] = ["impact", "bounce", "skid", "roll", "rest"];
export const GROUND_PLAYBACK_MAX_POINTS = 100_000;
export const GROUND_PLAYBACK_VISUAL_POINT_BUDGET = 2_048;
export const GROUND_PLAYBACK_VISIBLE_ROW_BUDGET = 256;

/** Find the final sample at or before a target in logarithmic time. */
export const lowerPlaybackIndex = (
  times: readonly number[],
  target: number,
): number => {
  let low = 0;
  let high = times.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (times[middle] <= target) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
};

/** Select the exact adjacent sample in logarithmic time. */
export const adjacentPlaybackTime = (
  times: readonly number[],
  currentTimeS: number,
  direction: -1 | 1,
): number => {
  if (times.length === 0) throw new RangeError("playback timeline is empty");
  const first = times[0];
  const last = times[times.length - 1];
  const tolerance = 1e-12;
  if (direction === 1) {
    if (currentTimeS < first - tolerance) return first;
    if (currentTimeS >= last - tolerance) return last;
    const lower = lowerPlaybackIndex(times, currentTimeS + tolerance);
    return times[Math.min(lower + 1, times.length - 1)];
  }
  if (currentTimeS <= first + tolerance) return first;
  if (currentTimeS > last + tolerance) return last;
  return times[lowerPlaybackIndex(times, currentTimeS - tolerance)];
};

const evenlySelected = (values: readonly number[], limit: number): number[] => {
  if (limit <= 0 || values.length === 0) return [];
  if (limit === 1) return [values[0]];
  const last = values.length - 1;
  return Array.from({ length: limit }, (_, index) =>
    values[Math.round(index * last / (limit - 1))]);
};

/** Select a deterministic bounded path with phase and event landmarks. */
export const selectGroundPlaybackPath = (
  result: FlightToGroundResult,
  maxPoints = GROUND_PLAYBACK_VISUAL_POINT_BUDGET,
): readonly GroundTrajectoryPoint[] => {
  if (!Number.isInteger(maxPoints) || maxPoints < 2) {
    throw new RangeError("maxPoints must be an integer of at least two");
  }
  const points = result.trajectory;
  if (points.length <= maxPoints) return points;
  const phaseIndices = new Set<number>([0, points.length - 1]);
  points.slice(1).forEach((point, offset) => {
    const index = offset + 1;
    if (point.phase !== points[index - 1].phase) {
      phaseIndices.add(index - 1);
      phaseIndices.add(index);
    }
  });
  const priority = [...phaseIndices].sort((left, right) => left - right);
  if (priority.length >= maxPoints) {
    return evenlySelected(priority, maxPoints).map((index) => points[index]);
  }
  const times = points.map(({ time_s }) => time_s);
  const eventIndices = new Set<number>();
  result.events.forEach(({ time_s }) => {
    const lower = lowerPlaybackIndex(times, time_s);
    eventIndices.add(lower);
    if (lower + 1 < points.length && times[lower] < time_s) {
      eventIndices.add(lower + 1);
    }
  });
  const selected = new Set(priority);
  const events = [...eventIndices]
    .filter((index) => !selected.has(index))
    .sort((left, right) => left - right);
  evenlySelected(events, Math.min(maxPoints - selected.size, events.length))
    .forEach((index) => selected.add(index));
  const context = Array.from({ length: points.length }, (_, index) => index)
    .filter((index) => !selected.has(index));
  evenlySelected(context, Math.min(maxPoints - selected.size, context.length))
    .forEach((index) => selected.add(index));
  return [...selected].sort((left, right) => left - right)
    .map((index) => points[index]);
};

export interface GroundEvidenceWindow<T> {
  readonly rows: readonly T[];
  readonly disclosure: string | null;
}

/** Return a bounded evidence-table window without modifying the source result. */
export const groundEvidenceWindow = <T>(
  values: readonly T[],
): GroundEvidenceWindow<T> => {
  const rows = values.slice(0, GROUND_PLAYBACK_VISIBLE_ROW_BUDGET);
  const disclosure = values.length > rows.length
    ? `Showing first ${rows.length} of ${values.length} validated rows; full result retained.`
    : null;
  return Object.freeze({ rows: Object.freeze(rows), disclosure });
};

export class GroundPlaybackTimeline {
  readonly result: FlightToGroundResult;
  private readonly times: readonly number[];

  constructor(result: FlightToGroundResult) {
    const playableStatus = result.status === "complete" || result.status === "partial";
    if (!playableStatus || result.trajectory.length === 0 || result.summary === null) {
      throw new RangeError("playback requires a complete or partial ground result");
    }
    this.result = result;
    this.times = result.trajectory.map(({ time_s }) => time_s);
  }

  get startTimeS(): number { return this.times[0]; }
  get endTimeS(): number { return this.times[this.times.length - 1]; }
  get durationS(): number { return this.endTimeS - this.startTimeS; }
  get isComplete(): boolean { return this.result.status === "complete"; }
  get carryPositionM(): GroundVec3 { return this.result.trajectory[0].position_m; }
  get endpointPositionM(): GroundVec3 {
    return this.result.trajectory[this.result.trajectory.length - 1].position_m;
  }
  get endLabel(): string {
    if (!this.isComplete) return "Observed end";
    return this.result.termination.reason === "rest" ? "Rest" : "End / left surface";
  }

  phaseTime(phase: GroundPhase): number | null {
    if (!VALID_PHASES.includes(phase)) throw new RangeError(`unknown ground phase: ${phase}`);
    return this.result.trajectory.find((point) => point.phase === phase)?.time_s ?? null;
  }

  stepTime(currentTimeS: number, direction: -1 | 1): number {
    this.validateTime(currentTimeS);
    return adjacentPlaybackTime(this.times, currentTimeS, direction);
  }

  frameAt(timeS: number): GroundPlaybackFrame {
    this.validateTime(timeS);
    const clamped = Math.min(Math.max(timeS, this.startTimeS), this.endTimeS);
    const lowerIndex = lowerPlaybackIndex(this.times, clamped);
    const lower = this.result.trajectory[lowerIndex];
    const upper = this.result.trajectory[lowerIndex + 1];
    if (upper === undefined || lower.phase !== upper.phase) {
      return this.frame(lowerIndex, clamped, lower.position_m, 0);
    }
    const fraction = (clamped - lower.time_s) / (upper.time_s - lower.time_s);
    const position = lower.position_m.map((value, index) =>
      value + fraction * (upper.position_m[index] - value),
    ) as unknown as GroundVec3;
    return this.frame(lowerIndex, clamped, position, fraction);
  }

  private validateTime(value: number): void {
    if (!Number.isFinite(value)) throw new RangeError("playback time must be finite");
  }

  private frame(
    lowerIndex: number,
    timeS: number,
    positionM: GroundVec3,
    interpolationFraction: number,
  ): GroundPlaybackFrame {
    return Object.freeze({
      timeS,
      elapsedS: timeS - this.startTimeS,
      positionM,
      phase: this.result.trajectory[lowerIndex].phase,
      lowerIndex,
      interpolationFraction,
      isTerminal: timeS >= this.endTimeS,
    });
  }
}

/** Adapt validated regional evidence without executing or reconstructing physics. */
export const timelineFromRegionalExecution = (
  execution: GroundRegionalExecutionResult,
): GroundPlaybackTimeline => {
  const playable = execution.status === "complete" || execution.status === "partial";
  if (!playable || execution.ground_result === null) {
    throw new RangeError("regional execution requires a playable ground result");
  }
  if (execution.ground_result.trajectory.length > GROUND_PLAYBACK_MAX_POINTS) {
    throw new RangeError("ground result trajectory exceeds the import point limit");
  }
  return new GroundPlaybackTimeline(execution.ground_result);
};
