import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { NeuralModelLabPanel } from "./NeuralModelLabPanel";

describe("NeuralModelLabPanel", () => {
  it("states the training boundary and exposes accessible configuration", () => {
    render(<NeuralModelLabPanel />);
    expect(screen.getByRole("heading", { name: "Neural Model Lab" })).toBeInTheDocument();
    expect(screen.getByText(/browser prepares a training request/i)).toBeInTheDocument();
    expect(screen.getByText(/not vendor emulation or certification/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Vendor target")).toHaveAttribute("title");
    expect(screen.getByRole("button", { name: "Export Training Request" })).toHaveAttribute("title");
  });

  it("loads a portable bundle and queries a single row locally", async () => {
    const model = {
      schema: "launch-monitor-neural-bundle/v1",
      modelId: "demo", vendor: "TrackMan", createdAt: "2026-08-06T00:00:00Z",
      features: [{ name: "speed", unit: "mph", mean: 100, scale: 10, min: 50, max: 200 }],
      outputs: [{ name: "carry", unit: "yd", mean: 200, scale: 20 }],
      layers: [{ activation: "linear", weights: [[2]], bias: [0] }],
      metrics: [{ model: "neural_surrogate", split: "test", target: "carry", n: 12, mae: 4, rmse: 5, r2: 0.9 }],
      learningCurve: [{ training_fraction: 1, training_rows: 12, validation_standardized_rmse: 0.2 }],
      provenance: { datasetSha256: "b".repeat(64), rowCount: 12,
        splitMethod: "held-out", interpretation: "small sample" },
    };
    render(<NeuralModelLabPanel />);
    fireEvent.change(screen.getByLabelText("Portable model bundle JSON"), {
      target: { files: [new File([JSON.stringify(model)], "model.json", { type: "application/json" })] },
    });
    expect(await screen.findByText("demo")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("speed (mph)"), { target: { value: "120" } });
    fireEvent.click(screen.getByRole("button", { name: "Query Model" }));
    expect(screen.getByText("280.000 yd")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /learning curve/i })).toBeInTheDocument();
  });

  it("imports custom CSV metadata and reports unavailable vendor models honestly", async () => {
    render(<NeuralModelLabPanel />);
    expect(screen.getByText(/No portable model loaded for TrackMan/i)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Custom training CSV"), {
      target: { files: [new File(["speed,carry\n100,200\n110,220\n"], "shots.csv", { type: "text/csv" })] },
    });
    expect(await screen.findByText(/shots.csv: 2 rows, 2 columns/i)).toBeInTheDocument();
  });

  it("offers batch prediction and backing-data export", () => {
    render(<NeuralModelLabPanel />);
    expect(screen.getByLabelText("Batch query CSV")).toHaveAttribute("title");
    expect(screen.getByRole("button", { name: "Export Prediction Data" })).toHaveAttribute("title");
  });
});
