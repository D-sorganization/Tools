import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { getClub } from "../model/club";
import {
  installSimulationPanelTestEnvironment,
  renderSimulationPanel,
} from "./simulationPanelTestSupport";

beforeAll(installSimulationPanelTestEnvironment);
afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe("manual delivery controls", () => {
  it("makes the bounded frame-explicit inputs discoverable only for manual source", () => {
    renderSimulationPanel(getClub("Driver 10.5°"));
    const attack = screen.getByRole("textbox", { name: "Manual Attack Angle" });
    const path = screen.getByRole("textbox", { name: "Manual Club Path" });
    const lean = screen.getByRole("textbox", { name: "Manual Forward Shaft Lean" });
    const datum = screen.getByRole("combobox", { name: "Shaft Axis Datum" });
    expect(attack).toBeEnabled();
    expect(path).toBeEnabled();
    expect(lean).toBeEnabled();
    expect(datum).toBeEnabled();
    expect(datum).toHaveValue("tracked_reference");
    expect(attack).toHaveAttribute("min", "-89");
    expect(path).toHaveAttribute("max", "89");
    expect(lean).toHaveAttribute("max", "60");

    fireEvent.change(attack, { target: { value: "-10" } });
    fireEvent.blur(attack);
    fireEvent.change(path, { target: { value: "6" } });
    fireEvent.blur(path);
    fireEvent.change(lean, { target: { value: "15" } });
    fireEvent.blur(lean);
    fireEvent.change(datum, { target: { value: "generated_hosel" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Simulation" }));
    expect(screen.queryByText(/Run failed/)).not.toBeInTheDocument();
    expect(screen.getByText(/generated_head_profile_hosel/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Swing Source"), {
      target: { value: "double_pendulum" },
    });
    expect(attack).toBeDisabled();
    expect(path).toBeDisabled();
    expect(lean).toBeDisabled();
    expect(datum).toBeDisabled();
  });

  it("restores manual delivery fields from an imported legacy-compatible run", async () => {
    renderSimulationPanel(getClub("Driver 10.5°"));
    const file = new File([JSON.stringify({
      format: "rate_of_closure.simulation_run.web/3",
      parameters: {
        sourceKind: "manual",
        manualAttackAngleDeg: -8,
        manualClubPathDeg: 3,
        manualForwardShaftLeanDeg: 11,
        shaftAxisDatum: "generated_hosel",
      },
    })], "manual-run.json", { type: "application/json" });

    fireEvent.change(screen.getByLabelText("Import Simulation Settings JSON"), {
      target: { files: [file] },
    });

    await waitFor(() => expect(
      screen.getByRole("textbox", { name: "Manual Attack Angle" }),
    ).toHaveValue("-8"));
    expect(screen.getByRole("textbox", { name: "Manual Club Path" }))
      .toHaveValue("3");
    expect(screen.getByRole("textbox", { name: "Manual Forward Shaft Lean" }))
      .toHaveValue("11");
    expect(screen.getByRole("combobox", { name: "Shaft Axis Datum" }))
      .toHaveValue("generated_hosel");
  });
});
