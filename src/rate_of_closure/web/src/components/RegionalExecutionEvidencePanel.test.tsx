import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import fixture from "../model/__fixtures__/ground_regional_execution_golden_v1.json";
import { parseGroundRegionalExecutionResult } from "../model/groundRegionalExecution";
import { RegionalExecutionEvidencePanel } from "./RegionalExecutionEvidencePanel";
import { RegionalExecutionLedgerTables } from "./RegionalExecutionLedgerTables";

describe("RegionalExecutionEvidencePanel", () => {
  it("shows strict Python evidence and disclaims browser physics", async () => {
    const result = parseGroundRegionalExecutionResult(fixture.representable.result);
    const { container } = render(
      <RegionalExecutionEvidencePanel currentPlan={() => result.regional_plan} />,
    );
    expect(screen.getByText(/does not execute, approximate, or modify physics/i)).toBeVisible();
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
    expect(screen.getByRole("table", { name: "Ground execution events" })).toBeVisible();
    expect(screen.getByRole("table", { name: "Regional surface transitions" })).toBeVisible();
    expect(screen.getByText("first_contact")).toBeVisible();
    expect(screen.getByText("(0.000000, 0.021350, 0.000000)")).toBeVisible();
    expect(screen.getByText("base / firm-fairway")).toBeVisible();
    expect(screen.getByText("rough-band / regional-rough")).toBeVisible();
    expect(screen.getByText(/CENSORED_ENDPOINT/)).toBeVisible();
    expect(screen.getByText(/no browser physics executed/i)).toBeVisible();

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

    render(<RegionalExecutionLedgerTables events={events} transitions={[]} />);

    expect(screen.getByText(/showing first 256 of 257 validated rows/i)).toBeVisible();
    fireEvent.click(screen.getByLabelText("Toggle ground execution events"));
    const table = screen.getByRole("table", { name: "Ground execution events" });
    expect(within(table).getAllByRole("row")).toHaveLength(257);
  });
});
