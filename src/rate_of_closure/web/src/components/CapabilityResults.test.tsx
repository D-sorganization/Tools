import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { runCapabilityOptimization } from "../model/capabilityRun";
import {
  buildCapabilityWorkflow,
  defaultCapabilityWorkflowInputs,
} from "../model/capabilityWorkflow";
import { CapabilityResults } from "./CapabilityResults";

const output = (ensembleSize: number) => runCapabilityOptimization(
  buildCapabilityWorkflow({
    ...defaultCapabilityWorkflowInputs(),
    candidateBudget: 1,
    ensembleSize,
    alternativesCount: 1,
  }),
);

describe("CapabilityResults", () => {
  it("exposes the complete ranked-alternative diagnostic contract", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    render(<CapabilityResults output={output(2)} />);

    const table = screen.getByRole("table", { name: "Ranked capability alternatives" });
    [
      "Score", "Miss CVaR", "Downside carry", "Outcomes", "Failure rate",
      "Evidence", "Pareto",
    ].forEach((heading) => {
      expect(within(table).getByRole("columnheader", { name: heading })).toBeInTheDocument();
    });
    expect(within(table).getByText(/ball_speed=.*m\/s/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export results CSV" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export results JSON" })).toBeInTheDocument();
  });

  it("resets raw-row paging when a smaller result replaces the output", async () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const view = render(<CapabilityResults output={output(26)} />);
    fireEvent.click(screen.getByRole("button", { name: "Next rows" }));
    expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();

    view.rerender(<CapabilityResults output={output(2)} />);

    expect(await screen.findByText("Page 1 of 1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next rows" })).toBeDisabled();
  });
});
