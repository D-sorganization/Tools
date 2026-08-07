import { cleanup, fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { getClub } from "../model/club";
import {
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "../model/spatialTarget";
import { spatialTargetToJson } from "../model/spatialTargetSerialization";
import {
  installSimulationPanelTestEnvironment,
  renderSimulationPanel,
} from "./simulationPanelTestSupport";

beforeAll(installSimulationPanelTestEnvironment);
afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe("integrated simulation spatial target workflow", () => {
  it("uses one canonical target in the editor, flight view, and exports", async () => {
    const onTargetChange = vi.fn();
    renderSimulationPanel(getClub("Driver 10.5°"), onTargetChange);
    fireEvent.click(screen.getByRole("tab", { name: "Flight" }));
    expect(screen.getByRole("status", { name: "Current spatial target" }))
      .toHaveTextContent(/Green Target.*230\.0 m downrange.*surface circle/i);
    expect(screen.getByRole("button", { name: "Export CSV" })).toBeEnabled();

    fireEvent.change(screen.getByLabelText("Target type"), {
      target: { value: "aerial_waypoint" },
    });
    fireEvent.change(await screen.findByLabelText("Target elevation m"), {
      target: { value: "24" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));
    expect(onTargetChange).toHaveBeenCalledWith(expect.objectContaining({
      kind: "aerial_waypoint",
      point: expect.objectContaining({ appCoordinatesM: [230, 24, 0] }),
    }));
  });

  it("imports target and setup atomically and rejects incomplete v4 documents", async () => {
    const onTargetChange = vi.fn();
    renderSimulationPanel(getClub("Driver 10.5°"), onTargetChange);
    const aerial = createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([140, 24, -3], "app"),
      tolerance: sphereTolerance(4),
      elevationSource: "absolute",
    });
    const validFile = new File([JSON.stringify({
      format: "rate_of_closure.simulation_run.web/4",
      spatial_target: JSON.parse(spatialTargetToJson(aerial)),
      parameters: { ball_setup: { support_mode: "ground", tee_height_m: 0 } },
    })], "run.json", { type: "application/json" });
    fireEvent.change(screen.getByLabelText("Import Simulation JSON"), {
      target: { files: [validFile] },
    });
    await screen.findByText(/Imported Ground ball setup and spatial target/i);
    expect(onTargetChange).toHaveBeenCalledWith(aerial);
    expect(screen.getByRole("radio", { name: "Ground" })).toBeChecked();

    onTargetChange.mockClear();
    const invalidFile = new File([JSON.stringify({
      format: "rate_of_closure.simulation_run.web/4",
      parameters: { ball_setup: { support_mode: "tee", tee_height_m: 0.05 } },
    })], "invalid-run.json", { type: "application/json" });
    fireEvent.change(screen.getByLabelText("Import Simulation JSON"), {
      target: { files: [invalidFile] },
    });
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Cannot import simulation settings.*requires spatial_target/i);
    expect(alert).toHaveClass("text-rose-200");
    expect(onTargetChange).not.toHaveBeenCalled();
    expect(screen.getByRole("radio", { name: "Ground" })).toBeChecked();

    const missingSetupFile = new File([JSON.stringify({
      format: "rate_of_closure.simulation_run.web/4",
      spatial_target: JSON.parse(spatialTargetToJson(aerial)),
      parameters: {},
    })], "missing-setup.json", { type: "application/json" });
    fireEvent.change(screen.getByLabelText("Import Simulation JSON"), {
      target: { files: [missingSetupFile] },
    });
    const setupError = await screen.findByText(/requires ball_setup/i);
    expect(setupError).toHaveAttribute("role", "alert");
    expect(onTargetChange).not.toHaveBeenCalled();
    expect(screen.getByRole("radio", { name: "Ground" })).toBeChecked();
  });
});
