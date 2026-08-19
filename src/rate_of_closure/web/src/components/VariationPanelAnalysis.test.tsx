import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { VariationPanel } from "./VariationPanel";

class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

let storage: Storage;

beforeEach(() => {
  storage = new MemoryStorage();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VariationPanel analysis execution policy", () => {
  it("executes only the explicitly selected analyses", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    const runs = screen.getByRole("textbox", { name: "Runs" });
    fireEvent.change(runs, { target: { value: "2" } });
    fireEvent.blur(runs);
    const selector = screen.getByRole("combobox", { name: "Analysis execution" });

    await user.selectOptions(selector, "all_together");
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    expect(screen.getByText(/Summary — Dispersion/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Impact and Shot-Outcome Scatter/i })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Scatter horizontal axis" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Scatter vertical axis" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /variation scatter/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Scatter Matrix and Marginal/i })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: /Scatter matrix with marginal histograms/i })).toBeInTheDocument();
    for (const name of [
      "Matrix SVG", "Matrix Selected CSV", "Matrix Plot Definition JSON",
      "Scatter SVG", "Scatter Selected CSV", "Scatter Plot Definition JSON",
    ]) {
      expect(screen.getByRole("button", { name })).toBeEnabled();
    }
    expect(screen.queryByText(/One-at-a-Time Sensitivity/i)).not.toBeInTheDocument();

    await user.selectOptions(selector, "individual");
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    expect(screen.queryByText(/Summary — Dispersion/i)).not.toBeInTheDocument();
    expect(screen.getByText(/One-at-a-Time Sensitivity/i)).toBeInTheDocument();

    await user.selectOptions(selector, "both");
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    expect(screen.getByText(/Summary — Dispersion/i)).toBeInTheDocument();
    expect(screen.getByText(/One-at-a-Time Sensitivity/i)).toBeInTheDocument();
  });

  it("renders every swing trial in the interactive arc inspector", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    await user.selectOptions(screen.getByRole("combobox", { name: "Pipeline" }), "swing");
    fireEvent.change(screen.getByRole("textbox", { name: "Runs" }), {
      target: { value: "2" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Runs" }));
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Analysis execution" }),
      "all_together",
    );

    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));

    expect(screen.getByRole("heading", { name: /All Swing Arcs/i })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Arc modeled point" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Arc outcome cohort" })).toHaveValue("all");
    const source = screen.getByRole("combobox", { name: "Arc perturbation source" });
    const band = screen.getByRole("combobox", { name: "Arc perturbation band" });
    expect(screen.getByText(/2\/2 trials shown/i)).toBeInTheDocument();
    expect(band).toBeDisabled();
    await user.selectOptions(source, "swing_sim.swing.yaw_deg");
    expect(band).toBeEnabled();
    await user.selectOptions(band, "lower");
    fireEvent.change(screen.getByRole("slider", { name: "Arc phase end percent" }), {
      target: { value: "75" },
    });
    expect(screen.getByText(/Displayed Swing Phase: 0–75%/i)).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Quiet-zone metric threshold" })).toHaveValue(5);
    expect(screen.getByRole("img", { name: /interactive all-trial swing arcs/i })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /rms-radius and ranked quiet zones/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Swing Arcs PNG" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Variability SVG" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Arc Plot Definition JSON" })).toBeEnabled();
    expect(screen.getByText(/1\/2 trials shown/i)).toBeInTheDocument();
    expect(screen.getByText(/quiet samples .*common simulation time/i)).toBeInTheDocument();
    await user.click(screen.getByText("Accessible Selected Matrix Data"));
    await user.click(screen.getByRole("button", { name: "Select matrix trial 1" }));
    expect(screen.getByRole("combobox", { name: "Highlighted trial" })).toHaveValue("0");
    expect(screen.getByRole("combobox", { name: "Arc highlighted trial" })).toHaveValue("0");
    await user.selectOptions(screen.getByRole("combobox", { name: "Highlighted trial" }), "0");
    expect(screen.getByRole("combobox", { name: "Arc highlighted trial" })).toHaveValue("0");
    expect(screen.getByRole("button", { name: "Swing Traces CSV" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Swing Ensemble JSON" })).toBeEnabled();
    expect(screen.getByText(/Hits: .*Plotted landings: .*no fabricated landing/i)).toBeInTheDocument();
  });
});
