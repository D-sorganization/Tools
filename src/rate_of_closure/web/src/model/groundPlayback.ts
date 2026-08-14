/** Phase-safe playback semantics for strict imported ground results. */

import type {
  FlightToGroundResult,
  GroundPhase,
  GroundVec3,
} from "./flightGroundTypes";

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
    if (direction === 1) {
      return this.times.find((time) => time > currentTimeS + 1e-12) ?? this.endTimeS;
    }
    return [...this.times].reverse().find((time) => time < currentTimeS - 1e-12)
      ?? this.startTimeS;
  }

  frameAt(timeS: number): GroundPlaybackFrame {
    this.validateTime(timeS);
    const clamped = Math.min(Math.max(timeS, this.startTimeS), this.endTimeS);
    let lowerIndex = 0;
    while (lowerIndex + 1 < this.times.length && this.times[lowerIndex + 1] <= clamped) {
      lowerIndex += 1;
    }
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
