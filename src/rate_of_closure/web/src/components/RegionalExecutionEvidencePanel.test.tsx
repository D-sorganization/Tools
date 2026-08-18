import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/ground_regional_execution_golden_v1.json";
import { parseGroundRegionalExecutionResult } from "../model/groundRegionalExecution";
import { RegionalExecutionEvidencePanel } from "./RegionalExecutionEvidencePanel";
import { RegionalExecutionLedgerTables } from "./RegionalExecutionLedgerTables";

describe("RegionalExecutionEvidencePanel", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("shows strict Python evidence and disclaims browser physics", async () => {
    const result = parseGroundRegionalExecutionResult(fixture.representable.result);
    const { container } = render(
      <RegionalExecutionEvidencePanel currentPlan={() => result.regional_plan} />,
    );
    expect(screen.getByText(/does not execute, approximate, or modify physics/i)).toBeVisible();
    expect(screen.getByRole("button", { name: "Download canonical execution evidence JSON" }))
      .toBeDisabled();
    const input = container.querySelector("input[type='file']");
    expect(input).not.toBeNull();
    fireEvent.change(input!, { target: { files: [{
      name: "execution.json",
      size: JSON.stringify(fixture.representable.result).length,
      text: async () => JSON.stringify(fixture.representable.result),
    }] } });

    await waitFor(() => expect(screen.getByLabelText(
      "Regional execution evidence readback",
    )).toBeVisible());
    expect(screen.getByText("partial")).toBeVisible();
    expect(screen.getAllByText("0.254 m")).toHaveLength(2);
    expect(screen.getByText("0.040 m")).toBeVisible();
    expect(screen.getByText("1.155 s")).toBeVisible();
    expect(screen.getByText("impact → skid → roll")).toBeVisible();
    fireEvent.click(screen.getByLabelText("Toggle ground execution events"));
    fireEvent.click(screen.getByLabelText("Toggle regional surface transitions"));
    fireEvent.click(screen.getByLabelText("Toggle ground trajectory samples"));
    expect(screen.getByRole("table", { name: "Ground execution events" })).toBeVisible();
    expect(screen.getByRole("table", { name: "Regional surface transitions" })).toBeVisible();
    expect(screen.getByRole("table", { name: "Ground trajectory samples" })).toBeVisible();
    expect(screen.getByText("first_contact")).toBeVisible();
    expect(screen.getAllByText("(0.000000, 0.021350, 0.000000)")).toHaveLength(2);
    expect(screen.getByText("base / firm-fairway")).toBeVisible();
    expect(screen.getByText("rough-band / regional-rough")).toBeVisible();
    expect(screen.getByText("impact")).toBeVisible();
    expect(screen.getByText(/CENSORED_ENDPOINT/)).toBeVisible();
    expect(screen.getByText(/no browser physics executed/i)).toBeVisible();
    const download = screen.getByRole("button", {
      name: "Download canonical execution evidence JSON",
    });
    expect(download).toBeEnabled();
    const createUrl = vi.fn(() => "blob:regional-execution-panel");
    const revokeUrl = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    fireEvent.click(download);
    expect(click).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:regional-execution-panel");
    expect(screen.getByText(/downloaded canonical evidence/i)).toBeVisible();

    vi.stubGlobal("URL", {
      createObjectURL: () => { throw new Error("simulated object URL failure"); },
      revokeObjectURL: revokeUrl,
    });
    fireEvent.click(download);
    expect(screen.getByRole("alert")).toHaveTextContent("simulated object URL failure");
    expect(screen.getByText(/download failed; prior accepted execution evidence was preserved/i))
      .toBeVisible();
    expect(screen.getAllByText("0.254 m")).toHaveLength(2);

    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: revokeUrl });
    fireEvent.click(download);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    click.mockRestore();

    fireEvent.change(input!, { target: { files: [{
      name: "wrong-plan.json",
      size: 2,
      text: async () => "{}",
    }] } });
    await waitFor(() => expect(screen.getByRole("alert")).toBeVisible());
    expect(screen.getAllByText("0.254 m")).toHaveLength(2);
    expect(screen.getByText(/prior accepted execution evidence was preserved/i)).toBeVisible();
  });

  it("bounds rendered ledgers while retaining the validated readback", () => {
    const result = parseGroundRegionalExecutionResult(fixture.representable.result);
    const first = result.ground_result!.events[0];
    const events = Array.from({ length: 257 }, (_, sequence) => ({
      sequence,
      eventType: first.event_type,
      timeS: first.time_s,
      frame: first.frame,
      positionM: first.position_m,
      velocityBeforeMps: first.velocity_before_m_s,
      velocityAfterMps: first.velocity_after_m_s,
      angularVelocityBeforeRadS: first.angular_velocity_before_rad_s,
      angularVelocityAfterRadS: first.angular_velocity_after_rad_s,
    }));

    const point = result.ground_result!.trajectory[0];
    const trajectory = Array.from({ length: 257 }, (_, index) => ({
      ...point,
      time_s: point.time_s + index * 0.001,
    }));
    render(<RegionalExecutionLedgerTables
      events={events} transitions={[]} trajectory={trajectory} />);

    expect(screen.getAllByText(/showing first 256 of 257 validated rows/i)).toHaveLength(2);
    fireEvent.click(screen.getByLabelText("Toggle ground execution events"));
    const table = screen.getByRole("table", { name: "Ground execution events" });
    expect(within(table).getAllByRole("row")).toHaveLength(257);
    fireEvent.click(screen.getByLabelText("Toggle ground trajectory samples"));
    const trajectoryTable = screen.getByRole("table", { name: "Ground trajectory samples" });
    expect(within(trajectoryTable).getAllByRole("row")).toHaveLength(257);
  });
});
