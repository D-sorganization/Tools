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
    const sim = runSimulation(input);
    const club = getClub("Pitching Wedge");
    const result = wedgeGroundClearance(sim, scenario, club);
    render(<WedgeGroundClearancePanel result={result} />);

    expect(screen.getByLabelText("Wedge Ground-Clearance Engineering Readout")).toBeInTheDocument();
    expect(screen.getByLabelText("Wedge contact sequence")).toHaveTextContent("Ball First");
    expect(screen.getAllByText("1.07 mm")).toHaveLength(2);
    expect(screen.getByText(/not a measured or manufacturer-specific grind/i)).toBeInTheDocument();
    expect(screen.getByText(/no turf deformation/i)).toBeInTheDocument();
  });

  it("does not present wedge-only claims for other club families", () => {
    const sim = runSimulation(input);
    const result = wedgeGroundClearance(
      sim, scenario, getClub("Driver 10.5°"),
    );
    const { container } = render(<WedgeGroundClearancePanel result={result} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders synchronized delivery cards, waterfall table, and accessible explainers when run is provided", () => {
    const sim = runSimulation(input);
    const club = getClub("Pitching Wedge");
    const result = wedgeGroundClearance(sim, scenario, club);
    render(
      <WedgeGroundClearancePanel
        result={result}
        run={sim}
        scenario={scenario}
        club={club}
      />,
    );

    // Delivery cards
    expect(screen.getByText("Contact Attack Angle")).toBeInTheDocument();
    expect(screen.getByText("AoA Without Shaft Rotation")).toBeInTheDocument();
    expect(screen.getByText("Shaft Rotation AoA Delta")).toBeInTheDocument();
    expect(screen.getByText("Delivered Dynamic Loft")).toBeInTheDocument();
    expect(screen.getByText("Delivered Dynamic Lie")).toBeInTheDocument();
    expect(screen.getByText("Dynamic Face Angle")).toBeInTheDocument();
    expect(screen.getByText("Delivery Low Point")).toBeInTheDocument();
    expect(screen.getByText("LE Vertical Rate")).toBeInTheDocument();

    // Waterfall section & table
    expect(
      screen.getByLabelText("Linear-Velocity Contribution Waterfall"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /linear-velocity contribution waterfall/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/1\. Shaft-Axis Translation \(v_axis\)/)).toBeInTheDocument();
    expect(screen.getByText(/\+ 2\. Shaft-Rotation Velocity \(v_shaft\)/)).toBeInTheDocument();
    expect(screen.getByText(/= Total Contact Velocity \(v_contact\)/)).toBeInTheDocument();

    // Non-additive AoA disclaimer note
    expect(
      screen.getByText(/Linear velocity components are strictly additive in 3D Euclidean space/i),
    ).toBeInTheDocument();

    // Explainers details are present and clickable
    const explainers = screen.getAllByText("Click for Definition");
    expect(explainers.length).toBeGreaterThan(10);
  });
});


