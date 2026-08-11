import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import fixture from "../model/__fixtures__/ground_regional_execution_golden_v1.json";
import { parseGroundRegionalExecutionResult } from "../model/groundRegionalExecution";
import { RegionalExecutionEvidencePanel } from "./RegionalExecutionEvidencePanel";

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
});
