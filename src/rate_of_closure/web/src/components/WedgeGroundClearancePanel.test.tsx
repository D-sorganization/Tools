import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { getClub } from "../model/club";
import { DEFAULT_SCENARIO, solve } from "../model/impact";
import { runSimulation, type SimulationInput } from "../model/simulation";
import { wedgeGroundClearance } from "../model/wedgeGroundClearance";
import { WedgeGroundClearancePanel } from "./WedgeGroundClearancePanel";

const scenario = { ...DEFAULT_SCENARIO, clubheadSpeedMph: 30, lieAngleDeg: 64,
  omegaPlaneDps: 0, omegaShaftDps: 1307, comToFaceMm: 20 };
const input: SimulationInput = {
  sourceKind: "manual", clubheadSpeedMph: 30, omegaDps: solve(scenario).omegaDps,
  loftDeg: 46, impactOffsetToeMm: 0, impactOffsetHighMm: 0,
  planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
  impactTimeS: 0.03, swingDurationS: 1.5,
};

describe("WedgeGroundClearancePanel", () => {
  it("makes sequence, clearance, provenance, and limits visible", () => {
    const result = wedgeGroundClearance(
      runSimulation(input), scenario, getClub("Pitching Wedge"),
    );
    render(<WedgeGroundClearancePanel result={result} />);

    expect(screen.getByLabelText("Wedge Ground-Clearance Engineering Readout")).toBeInTheDocument();
    expect(screen.getByLabelText("Wedge contact sequence")).toHaveTextContent("Ball First");
    expect(screen.getAllByText("1.07 mm")).toHaveLength(2);
    expect(screen.getByText(/not a measured or manufacturer-specific grind/i)).toBeInTheDocument();
    expect(screen.getByText(/no turf deformation/i)).toBeInTheDocument();
  });

  it("does not present wedge-only claims for other club families", () => {
    const result = wedgeGroundClearance(
      runSimulation(input), scenario, getClub("Driver 10.5°"),
    );
    const { container } = render(<WedgeGroundClearancePanel result={result} />);
    expect(container).toBeEmptyDOMElement();
  });
});
