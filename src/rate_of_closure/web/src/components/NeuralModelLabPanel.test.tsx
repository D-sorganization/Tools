import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { NeuralModelLabPanel } from "./NeuralModelLabPanel";

describe("NeuralModelLabPanel", () => {
  it("shows manifest-derived quantified blockers and safe private workflow", () => {
    render(<NeuralModelLabPanel />);
    expect(screen.getByText("TrackMan: unavailable")).toBeInTheDocument();
    expect(screen.getByText(/11,699 rows \/ 9,298 strict/)).toBeInTheDocument();
    expect(screen.getByText(/retired_non_group_safe/)).toBeInTheDocument();
    expect(screen.getByText(/2,794 rows \/ 0 strict/)).toBeInTheDocument();
    expect(screen.getByText(/never trains on or persists private rows/i)).toBeInTheDocument();
    const chart = screen.getByRole("img", { name: "Vendor strict eligible input rows chart" });
    const controls = screen.getByRole("region", { name: "Neural Model Lab controls" });
    expect(chart.compareDocumentPosition(controls) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByRole("button", { name: "Submit / Export Request" })).toHaveAttribute("title");
  });
});
