import { describe, expect, it } from "vitest";

import {
  candidateParetoFrontier,
  summarizeChipTrials,
  type ChipCandidateScoreTs,
  type ChipTrialRecordTs,
} from "./chipForgiveness";

const records: ChipTrialRecordTs[] = [
  { trialIndex: 0, cohort: "ball_first", loss: 1, constraintViolated: false, metrics: {} },
  { trialIndex: 1, cohort: "ball_first", loss: 2, constraintViolated: false, metrics: {} },
  { trialIndex: 2, cohort: "ground_first", loss: 4, constraintViolated: true, metrics: {} },
  { trialIndex: 3, cohort: "numerical_failure", loss: 9, constraintViolated: true, metrics: {} },
];

describe("chip forgiveness decision analysis", () => {
  it("keeps failures in probability, expected-loss, and CVaR denominators", () => {
    const summary = summarizeChipTrials(records, {
      seed: 41,
      cvarTailFraction: 0.5,
      bootstrapSamples: 512,
      turfCalibrationStatus: "illustrative",
    });

    expect(summary.sampleCount).toBe(4);
    expect(summary.cohorts.ball_first.probability).toBeCloseTo(0.5);
    expect(summary.cohorts.numerical_failure.probability).toBeCloseTo(0.25);
    expect(summary.expectedLoss).toBeCloseTo(4);
    expect(summary.cvarLoss).toBeCloseTo(6.5);
    expect(summary.constraintViolationRate).toBeCloseTo(0.5);
    expect(summary.supportsTurfRankings).toBe(false);
  });

  it("reports bounded Wilson confidence intervals", () => {
    const trials = Array.from({ length: 10 }, (_, index): ChipTrialRecordTs => ({
      trialIndex: index,
      cohort: index < 7 ? "ball_first" : "ground_first",
      loss: index,
      constraintViolated: false,
      metrics: {},
    }));

    const interval = summarizeChipTrials(trials, {
      seed: 2,
      bootstrapSamples: 128,
    }).cohorts.ball_first;

    expect(interval.ciLow).toBeCloseTo(0.396778, 6);
    expect(interval.ciHigh).toBeCloseTo(0.892209, 6);
  });

  it("replays bootstrap and convergence evidence deterministically", () => {
    const first = summarizeChipTrials(records, { seed: 7, bootstrapSamples: 128 });
    const second = summarizeChipTrials(records, { seed: 7, bootstrapSamples: 128 });

    expect(first.expectedLossCi).toEqual(second.expectedLossCi);
    expect(first.expectedLossCi[0]).toBeCloseTo(1.75, 12);
    expect(first.expectedLossCi[1]).toBeCloseTo(7.6625, 12);
    expect(first.convergence).toEqual(second.convergence);
  });

  it("retains nondominated mean, tail, and clean-contact tradeoffs", () => {
    const scores: ChipCandidateScoreTs[] = [
      { candidateId: "dominated", expectedLoss: 2, cvarLoss: 5, cleanProbability: 0.6 },
      { candidateId: "low-tail", expectedLoss: 1.3, cvarLoss: 2, cleanProbability: 0.8 },
      { candidateId: "low-mean", expectedLoss: 1, cvarLoss: 4, cleanProbability: 0.7 },
    ];

    expect(candidateParetoFrontier(scores).map((score) => score.candidateId)).toEqual([
      "low-mean",
      "low-tail",
    ]);
  });

  it("rejects sparse trial indices and non-finite losses", () => {
    expect(() => summarizeChipTrials([
      { ...records[0], loss: Number.NaN },
    ])).toThrow(/finite/i);
    expect(() => summarizeChipTrials([
      records[0], { ...records[1], trialIndex: 2 },
    ])).toThrow(/canonical trial order/i);
    expect(() => summarizeChipTrials([
      { ...records[0], metrics: { carry_m: Number.NaN } },
    ])).toThrow(/metric carry_m must be finite/i);
  });
});
