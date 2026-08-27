/** Accessible target/source selection for serialized Morris results. */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import fixture from "../model/__fixtures__/morris_global_sensitivity_golden_v1.json";
import { parseMorrisReport } from "../model/morrisGlobalSensitivityContract";
import { MorrisResults } from "./MorrisResults";

const reportWithRepeatedName = () => {
  const document = structuredClone(fixture) as Record<string, unknown>;
  const estimates = document.estimates as Array<Record<string, unknown>>;
  document.estimates = [...estimates, ...estimates
    .filter((estimate) => (estimate.target as Record<string, unknown>).name === "clubhead_x_m")
    .map((estimate) => ({
      ...structuredClone(estimate),
      target: {
        ...(structuredClone(estimate.target) as Record<string, unknown>),
        time_s: 0.04,
        point_id: "grip",
      },
    }))];
  return parseMorrisReport(document);
};

describe("Morris results", () => {
  it("selects exact point/phase targets and one input without rerunning analysis", async () => {
    const user = userEvent.setup();
    render(<MorrisResults report={reportWithRepeatedName()} />);

    const target = screen.getByRole("combobox", { name: "Output target" });
    const targetOptions = within(target).getAllByRole("option");
    expect(targetOptions.filter((option) => option.textContent?.includes("Clubhead X M")))
      .toHaveLength(2);
    expect(targetOptions.some((option) => option.textContent?.includes("clubhead · t=0.03 s")))
      .toBe(true);
    expect(targetOptions.some((option) => option.textContent?.includes("grip · t=0.04 s")))
      .toBe(true);

    const source = screen.getByRole("combobox", { name: "Input source" });
    expect(within(source).getAllByRole("option")).toHaveLength(3);
    await user.selectOptions(source, within(source).getAllByRole("option")[2]);
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });
});
