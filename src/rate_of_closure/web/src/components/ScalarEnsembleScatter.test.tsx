import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  buildScalarEnsembleScatter,
  createScalarEnsemble,
  scalarEnsembleRowId,
} from "../model/scalarEnsembleContract";
import { ScalarEnsembleScatter } from "./ScalarEnsembleScatter";

const scatter = buildScalarEnsembleScatter(createScalarEnsemble({
  result_id: "scatter-test",
  provenance: { adapter_id: "test", source_schema_version: "test/v1", source_provenance: "fixture" },
  stages: [{ key: "input", label: "Input" }, { key: "output", label: "Output" }],
  categories: [{ key: "parameter", label: "Parameter" }, { key: "metric", label: "Metric" }],
  variables: [
    { key: "speed", label: "Speed", unit: "mph", stage_key: "input", category_key: "parameter" },
    { key: "carry", label: "Carry", unit: "m", stage_key: "output", category_key: "metric" },
  ],
  cohorts: [
    { key: "complete", label: "Complete" },
    { key: "no_impact", label: "No Impact" },
    { key: "failed", label: "Failed" },
  ],
  rows: [
    { row_id: scalarEnsembleRowId(0), trial_index: 0, cohort: "complete", values: { speed: 90, carry: 180 } },
    { row_id: scalarEnsembleRowId(1), trial_index: 1, cohort: "no_impact", values: { speed: 95, carry: null } },
  ],
}), "speed", "carry");

describe("ScalarEnsembleScatter", () => {
  it("publishes numeric axis ranges and a cohort legend", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    render(<ScalarEnsembleScatter scatter={scatter} label="Engineering" />);

    expect(screen.getByText(/Speed range .* mph; Carry range .* m/)).toBeInTheDocument();
    expect(screen.getByLabelText("Engineering cohort legend")).toHaveTextContent("Complete 1");
    expect(screen.getByLabelText("Engineering cohort legend")).toHaveTextContent("No Impact 0");
    expect(screen.getByLabelText("Engineering cohort legend")).toHaveTextContent("Failed 0");
  });
});
