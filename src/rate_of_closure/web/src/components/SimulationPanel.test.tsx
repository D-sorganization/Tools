import { cleanup, fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { SimulationPanel } from "./SimulationPanel";
import {
  defaultSpatialTarget,
  installSimulationPanelTestEnvironment,
  renderSimulationPanel as renderPanel,
} from "./simulationPanelTestSupport";

beforeAll(installSimulationPanelTestEnvironment);

afterEach(() => {
  cleanup();
  if (typeof window.localStorage.clear === "function") window.localStorage.clear();
});

function displayedBallSpeed(): number {
  const text = screen.getByRole("button", { name: /Ball Speed/ }).textContent ?? "";
  const match = text.match(/([0-9]+(?:\.[0-9]+)?)\s+mph/);
  if (!match) throw new Error(`Ball Speed row had no mph value: ${text}`);
  return Number(match[1]);
}

function displayedLaunchAngle(): number {
  const text =
    screen.getByRole("button", { name: /Launch Angle/ }).textContent ?? "";
  const match = text.match(/([0-9]+(?:\.[0-9]+)?)\s+°/);
  if (!match) throw new Error(`Launch Angle row had no degree value: ${text}`);
  return Number(match[1]);
}

describe("SimulationPanel impact club", () => {
  it("shows a selectable, explained screw-axis glyph for club and joints", () => {
    renderPanel(getClub("Driver 10.5°"));
    expect(screen.getByRole("checkbox", { name: "Screw Axis" })).toBeChecked();
    expect(screen.getByRole("combobox", { name: "Screw Motion Entity" }))
      .toHaveValue("club");
    expect(screen.getByRole("note", { name: "Screw Motion Explanation" }))
      .toHaveTextContent(/finite screw.*Rate 2281\.5 deg\/s/i);

    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    const selector = screen.getByRole("combobox", { name: "Screw Motion Entity" });
    expect(selector).toHaveTextContent("Shoulder Joint");
    expect(selector).toHaveTextContent("Wrist Joint");
    fireEvent.change(selector, { target: { value: "joint.shoulder" } });
    expect(screen.getByRole("note", { name: "Screw Motion Explanation" }))
      .toHaveTextContent(/Shoulder Joint.*contribution residual/i);
    expect(screen.getByRole("note", { name: "Screw Motion Explanation" }))
      .toHaveTextContent("total = orbital + axial");
  });

  it("applies club defaults, preserves explicit overrides, and can restore the default", () => {
    const driver = getClub("Driver 10.5°");
    const view = renderPanel(driver);
    expect(screen.getByRole("radio", { name: "Tee" })).toBeChecked();
    expect(screen.getByRole("textbox", { name: "Tee Height" })).toHaveValue("38.1");

    fireEvent.click(screen.getByRole("radio", { name: "Ground" }));
    expect(screen.getByRole("textbox", { name: "Tee Height" })).toBeDisabled();
    expect(screen.getByText(/Ground mode.*effective tee height is 0 mm/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Tee" }));
    expect(screen.getByRole("textbox", { name: "Tee Height" })).toHaveValue("38.1");

    view.rerender(
      <SimulationPanel
        scenario={{ ...DEFAULT_SCENARIO, impactOffsetToeMm: 20 }}
        loftDeg={34}
        clubSpec={getClub("7-Iron")}
        onScenarioChange={() => undefined}
        spatialTarget={defaultSpatialTarget}
        onSpatialTargetChange={() => undefined}
      />,
    );
    expect(screen.getByRole("radio", { name: "Tee" })).toBeChecked();
    expect(screen.getByText(/Explicit Override/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use Club Default" }));
    expect(screen.getByText(/Club Default/i)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Ground" })).toBeChecked();
  });

  it("uses whole-field decimal editing and shows actionable negative validation", () => {
    renderPanel(getClub("Driver 10.5°"));
    const field = screen.getByRole("textbox", { name: "Tee Height" });
    fireEvent.focus(field);
    expect((field as HTMLInputElement).selectionStart).toBe(0);
    expect((field as HTMLInputElement).selectionEnd).toBe("38.1".length);
    fireEvent.change(field, { target: { value: "-2" } });
    fireEvent.blur(field);
    expect(screen.getByRole("alert")).toHaveTextContent(/Tee height.*finite.*non-negative/i);
    expect(screen.getByRole("textbox", { name: "Tee Height" })).toHaveValue("38.1");
  });

  it("defaults non-drivers to Ground and imports old exports as Ground", async () => {
    renderPanel(getClub("7-Iron"));
    expect(screen.getByRole("radio", { name: "Ground" })).toBeChecked();
    const file = new File([
      JSON.stringify({
        format: "rate_of_closure.simulation_run.web/2",
        parameters: { sourceKind: "manual" },
      }),
    ], "old-run.json", { type: "application/json" });
    fireEvent.change(screen.getByLabelText("Import Simulation Settings JSON"), {
      target: { files: [file] },
    });
    await screen.findByText(/Imported Ground ball setup/i);
    expect(screen.getByRole("radio", { name: "Ground" })).toBeChecked();
  });

  it("announces when impact physics falls back to the default driver", () => {
    renderPanel(null);
    expect(
      screen.getByRole("note", { name: "Impact club physics" }),
    ).toHaveTextContent(/default driver/i);
  });

  it("marks results stale as soon as a simulation input changes", () => {
    renderPanel(getClub("Driver 10.5°"));
    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    expect(screen.getByText("Inputs changed — run required")).toBeInTheDocument();
  });

  it("marks plane orientation non-applicable for manual and enables it for pendulums", () => {
    renderPanel(getClub("Driver 10.5°"));
    const yaw = screen.getByRole("textbox", { name: "Plane Yaw deg" });
    const side = screen.getByRole("textbox", { name: "Plane Side Tilt deg" });
    const forward = screen.getByRole("textbox", { name: "Plane Forward Tilt deg" });
    expect(yaw).toBeDisabled();
    expect(side).toBeDisabled();
    expect(forward).toBeDisabled();
    expect(screen.getByText(/Not applicable to Manual Constant-Twist Delivery/i))
      .toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    expect(yaw).toBeEnabled();
    expect(side).toBeEnabled();
    expect(forward).toBeEnabled();
    expect(screen.getByText(/Applies to articulated pendulum swing sources/i))
      .toBeInTheDocument();
  });

  it("resets prescribed torque atomically when leaving double pendulum", () => {
    renderPanel(getClub("Driver 10.5°"));
    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Torque execution mode" }), {
      target: { value: "prescribed" },
    });
    expect(screen.getByRole("status", { name: "Torque execution status" }))
      .toHaveTextContent(/prescribed/i);
    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "manual" },
    });
    expect(screen.getByRole("status", { name: "Torque execution status" }))
      .toHaveTextContent(/passive/i);
    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    expect(screen.queryByText(/Run failed/)).not.toBeInTheDocument();
  });

  it("reports a fixed-ball miss without launch values or an editable impact time", () => {
    renderPanel(getClub("Driver 10.5°"));
    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    fireEvent.change(screen.getByLabelText("Contact Policy"), {
      target: { value: "fixed_ball_contact" },
    });
    expect(screen.getByRole("slider", { name: "Impact Time" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    expect(
      screen.getByText("Completed — no club–ball impact"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Launch and flight values are intentionally absent/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ball Speed/ })).toHaveTextContent(
      "—",
    );
    const timeline = screen.getByRole("slider", { name: "Playback timeline" });
    const play = screen.getByRole("button", { name: "Play" });
    const jump = screen.getByRole("button", { name: "Jump to Closest Approach" });
    expect(timeline).toBeEnabled();
    expect(play).toBeEnabled();
    fireEvent.click(jump);
    expect(timeline).toHaveValue(jump.getAttribute("data-event-time"));
    fireEvent.click(play);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
  });

  it("jumps exactly to impact for a completed hit", () => {
    renderPanel(getClub("Driver 10.5°"));
    const jump = screen.getByRole("button", { name: "Jump to Impact" });
    const timeline = screen.getByRole("slider", { name: "Playback timeline" });

    fireEvent.click(jump);

    expect(timeline).toHaveValue(jump.getAttribute("data-event-time"));
    expect(screen.getByRole("complementary", {
      name: "Impact Kinematics Engineering Readout",
    })).toHaveTextContent("Shaft AoA Contribution");
    expect(screen.getByRole("complementary", {
      name: "Impact Kinematics Engineering Readout",
    })).toHaveTextContent("Geometry Basis");
    expect(screen.getByRole("complementary", {
      name: "Impact Kinematics Engineering Readout",
    })).toHaveTextContent("1307.0 °/s");
    expect(screen.getByRole("region", { name: "Interactive Impact Scene" }))
      .toBeInTheDocument();
    expect(screen.getByRole("img", {
      name: /Rotatable 3D wedge impact scene/i,
    })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("button", { name: "Export Vector SVG" }))
      .toBeEnabled();
    expect(screen.getByRole("checkbox", { name: "Face-Center Normal" }))
      .toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Face-Center Travel" }))
      .toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Spin-Loft Sector" }))
      .toBeChecked();
    expect(screen.getByText(/Face-Center Spin Loft:/).closest("p")).toHaveTextContent(
      /exact 3D.*planar approximation.*residual/i,
    );
    fireEvent.click(screen.getByText("Contact-Point AoA"));
    expect(screen.getByText(/atan2\(v_contact/)).toBeVisible();
  });

  it("persists independently toggleable D-plane layers", () => {
    const first = renderPanel(getClub("Driver 10.5°"));
    const spinSector = screen.getByRole("checkbox", { name: "Spin-Loft Sector" });
    const faceTravel = screen.getByRole("checkbox", { name: "Face-Center Travel" });
    fireEvent.click(spinSector);
    fireEvent.click(faceTravel);
    expect(spinSector).not.toBeChecked();
    expect(faceTravel).not.toBeChecked();
    first.unmount();

    renderPanel(getClub("Driver 10.5°"));
    expect(screen.getByRole("checkbox", { name: "Spin-Loft Sector" }))
      .not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Face-Center Travel" }))
      .not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Face-Center Normal" }))
      .toBeChecked();
  });

  it("passes the selected club mass and MOI into the simulation", () => {
    const driver = getClub("Driver 10.5°");
    const lightLowMoi = {
      ...driver,
      name: "Light Test Head",
      headMassKg: 0.15,
      moiAboutShaftKgM2: 2.0e-4,
    };
    const heavyHighMoi = {
      ...driver,
      name: "Heavy Test Head",
      headMassKg: 0.35,
      moiAboutShaftKgM2: 1.2e-3,
    };

    const first = renderPanel(lightLowMoi);
    const lightSpeed = displayedBallSpeed();
    first.unmount();
    renderPanel(heavyHighMoi);

    expect(displayedBallSpeed()).toBeGreaterThan(lightSpeed);
    expect(
      screen.getByRole("note", { name: "Impact club physics" }),
    ).toHaveTextContent("Heavy Test Head");
  });

  it("uses the selected club's nominal loft for the available delivery model", () => {
    const driver = getClub("Driver 10.5°");
    const low = renderPanel({ ...driver, name: "Low Loft", loftDeg: 8 });
    const lowAngle = displayedLaunchAngle();
    low.unmount();
    renderPanel({ ...driver, name: "High Loft", loftDeg: 16 });

    expect(displayedLaunchAngle()).toBeGreaterThan(lowAngle);
    expect(
      screen.getByRole("note", { name: "Impact club physics" }),
    ).toHaveTextContent("16.0° nominal loft");
  });

  it("presents reference-frame-aware joint locks and reconciles velocity", () => {
    renderPanel(getClub("Driver 10.5°"));
    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });

    const shoulderLock = screen.getByRole("checkbox", {
      name: "Lock Shoulder Joint",
    });
    const wristLock = screen.getByRole("checkbox", {
      name: "Lock Wrist Joint",
    });
    expect(shoulderLock).toHaveAttribute("title", expect.stringMatching(/absolute.*ground/i));
    expect(wristLock).toHaveAttribute("title", expect.stringMatching(/relative.*upper segment/i));

    const shoulderVelocity = screen.getByRole("textbox", {
      name: "Shoulder Initial Angular Velocity",
    });
    fireEvent.focus(shoulderVelocity);
    fireEvent.change(shoulderVelocity, { target: { value: "35" } });
    fireEvent.blur(shoulderVelocity);
    expect(shoulderVelocity).toHaveValue("35");

    fireEvent.click(shoulderLock);
    expect(shoulderVelocity).toBeDisabled();
    expect(shoulderVelocity).toHaveValue("0");
    expect(wristLock).not.toBeChecked();
    expect(screen.getByRole("status", { name: "Joint lock status" }))
      .toHaveTextContent(/shoulder.*absolute.*locked.*wrist.*free/i);

    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    expect(screen.getByText(/Completed.*Shoulder locked.*absolute/i))
      .toBeInTheDocument();
    expect(screen.queryByText(/Run failed/)).not.toBeInTheDocument();
  });

  it("clears incompatible locks when leaving the double-pendulum model", () => {
    renderPanel(getClub("Driver 10.5°"));
    const source = screen.getByLabelText("Swing Source");
    fireEvent.change(source, { target: { value: "double_pendulum" } });
    fireEvent.click(screen.getByRole("checkbox", { name: "Lock Wrist Joint" }));
    fireEvent.change(source, { target: { value: "triple_pendulum" } });
    expect(screen.queryByRole("checkbox", { name: "Lock Wrist Joint" }))
      .not.toBeInTheDocument();
    fireEvent.change(source, { target: { value: "double_pendulum" } });
    expect(screen.getByRole("checkbox", { name: "Lock Wrist Joint" }))
      .not.toBeChecked();
  });

  it("preserves locks when changing torque execution mode", () => {
    renderPanel(getClub("Driver 10.5°"));
    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "Lock Wrist Joint" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Torque execution mode" }), {
      target: { value: "prescribed" },
    });
    expect(screen.getByRole("checkbox", { name: "Lock Wrist Joint" }))
      .toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    expect(screen.getByText(/Completed.*prescribed torque profile.*Wrist locked.*relative/i))
      .toBeInTheDocument();
  });
});
