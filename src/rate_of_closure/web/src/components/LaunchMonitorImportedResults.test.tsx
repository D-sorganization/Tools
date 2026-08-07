import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { LaunchMonitorImportedResults } from "./LaunchMonitorImportedResults";

describe("LaunchMonitorImportedResults", () => {
  it("explains and displays imported private campaign outputs", () => {
    render(<LaunchMonitorImportedResults rows={[
      { shot_id: "a", feature: "spin", component: "PC1", loading: -0.8, pc1: 1.2, pc2: -0.4, held_out_rmse: 3.2, method: "PCA+RF", residual_m: 2.1 },
      { shot_id: "b", feature: "speed", component: "PC1", loading: 0.5, pc1: -0.7, pc2: 0.2, held_out_rmse: 3.2, method: "PCA+RF", residual_m: -1.1 },
    ]} />);
    expect(screen.getByRole("region", { name: "Imported advanced model analysis" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /PCA component one/i })).toBeInTheDocument();
    expect(screen.getByText("PCA Loading Magnitudes")).toBeInTheDocument();
    expect(screen.getByText("Held-Out Performance")).toBeInTheDocument();
    expect(screen.getByText(/does not refit or certify/i)).toBeInTheDocument();
  });
});

