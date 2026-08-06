import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { summarizeChipTrials, type ChipTrialRecordTs } from "../model/chipForgiveness";
import { ChipForgivenessPanel } from "./ChipForgivenessPanel";

describe("ChipForgivenessPanel", () => {
  it("makes all-trial risk, cohort confidence, and limitations visible", () => {
    const records: ChipTrialRecordTs[] = [
      {
        trialIndex: 0,
        cohort: "ball_first",
        loss: 1,
        constraintViolated: false,
        metrics: { carry_m: 27, peak_turf_penetration_m: null },
      },
      {
        trialIndex: 1,
        cohort: "numerical_failure",
        loss: 16,
        constraintViolated: true,
        metrics: { carry_m: null, peak_turf_penetration_m: null },
      },
    ];
    const summary = summarizeChipTrials(records, {
      bootstrapSamples: 64,
      turfCalibrationStatus: "illustrative",
    });

    render(<ChipForgivenessPanel summary={summary} limitations="Reduced turf diagnostic." />);

    expect(screen.getByRole("heading", { name: /conditional chip-shot forgiveness/i })).toBeInTheDocument();
    expect(screen.getByText(/expected loss/i)).toBeInTheDocument();
    expect(screen.getByText(/worst-tail cvar/i)).toBeInTheDocument();
    expect(screen.getByText(/numerical failure/i)).toBeInTheDocument();
    expect(screen.getByText(/turf ranking disabled/i)).toBeInTheDocument();
    expect(screen.getByText(/reduced turf diagnostic/i)).toBeInTheDocument();
    expect(screen.getAllByText("1 / 1").length).toBeGreaterThan(0);
  });
});
