import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { getClub, type ClubSpec } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { DEFAULT_TARGET } from "../model/targets";
import { SimulationPanel } from "./SimulationPanel";

beforeAll(() => {
  const context: unknown = new Proxy(function () {} as object, {
    get: (_target, property) =>
      property === "measureText" ? () => ({ width: 0 }) : () => context,
    set: () => true,
    apply: () => context,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    context as CanvasRenderingContext2D,
  );
});

afterEach(cleanup);

function renderPanel(clubSpec?: ClubSpec | null) {
  return render(
    <SimulationPanel
      scenario={{ ...DEFAULT_SCENARIO, impactOffsetToeMm: 20 }}
      loftDeg={10.5}
      clubSpec={clubSpec}
      onScenarioChange={() => undefined}
      target={DEFAULT_TARGET}
      onTargetChange={() => undefined}
    />,
  );
}

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
    expect(timeline).toBeEnabled();
    expect(play).toBeEnabled();
    fireEvent.click(play);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
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
});
