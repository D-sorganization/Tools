import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CATEGORY_LAUNCH, planFromJson, planToJson, type VariationPlanTs } from "../model/variation";
import { VARIATION_PLAN_LIBRARY_KEY } from "../model/variationPlanLibrary";
import { VariationPanel } from "./VariationPanel";
import { saveBallSetupPreference } from "../model/ballSetupPersistence";

class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;
const ANGLE = `${CATEGORY_LAUNCH}.launch_angle_deg`;

const importedPlan = (): VariationPlanTs => ({
  mode: "launch",
  baseVariables: { [BALL]: 158, [ANGLE]: 14 },
  noise: [
    {
      variableKey: BALL,
      distribution: "normal",
      scale: 2,
      lower: null,
      upper: null,
      specId: "localized-speed",
      timeWindowS: [0.7, 0.8],
      pointIds: ["swing.clubhead"],
    },
    {
      variableKey: ANGLE,
      distribution: "normal",
      scale: 1,
      lower: null,
      upper: null,
      specId: "angle",
      timeWindowS: null,
      pointIds: [],
    },
  ],
  groups: [
    {
      groupId: "launch-group",
      specIds: ["localized-speed", "angle"],
      matrixKind: "correlation",
      matrix: [
        [1, 0.4],
        [0.4, 1],
      ],
    },
  ],
  nRuns: 8,
  seed: 6,
  flightModel: "custom-flight-model",
});

let storage: Storage;

beforeEach(() => {
  storage = new MemoryStorage();
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => "blob:test"),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: vi.fn(),
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VariationPanel v2 plan persistence", () => {
  it("offers Tee Height only for the persisted Tee support context", () => {
    saveBallSetupPreference({
      setup: { supportMode: "ground", teeHeightM: 0 },
      userOverridden: true,
    }, storage);
    const ground = render(<VariationPanel storage={storage} />);
    expect(screen.getByText(/Tee Height is excluded in Ground mode/i)).toBeInTheDocument();
    expect(within(screen.getByLabelText("Variable 1")).queryByRole("option", {
      name: "Tee Height",
    })).not.toBeInTheDocument();
    ground.unmount();

    saveBallSetupPreference({
      setup: { supportMode: "tee", teeHeightM: 0.0381 },
      userOverridden: true,
    }, storage);
    render(<VariationPanel storage={storage} />);
    expect(screen.getByText(/Tee Height is available.*active Tee setup/i)).toBeInTheDocument();
    expect(within(screen.getByLabelText("Variable 1")).getByRole("option", {
      name: "Tee Height",
    })).toBeInTheDocument();
  });

  it("retains the complete imported plan when saving it to the named library", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    const file = new File([planToJson(importedPlan())], "plan.json", {
      type: "application/json",
    });

    await user.upload(screen.getByLabelText("Import variation plan JSON"), file);
    expect(await screen.findByText(/contains 1 grouped correlation/i)).toBeInTheDocument();
    expect(screen.getByText(/cannot yet execute.*scalar browser path/i)).toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Plan name" }), "Imported V2");
    await user.click(screen.getByRole("button", { name: "Save Named Plan" }));

    const stored = JSON.parse(storage.getItem(VARIATION_PLAN_LIBRARY_KEY)!) as {
      plans: Array<{ plan: unknown }>;
    };
    expect(planFromJson(JSON.stringify(stored.plans[0].plan))).toEqual(
      planFromJson(planToJson(importedPlan())),
    );
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    expect(screen.getByRole("status")).toHaveTextContent(/global perturbations/i);
  });

  it("supports loading, duplicating, and deleting named plans", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    await user.type(screen.getByRole("textbox", { name: "Plan name" }), "Baseline");
    await user.click(screen.getByRole("button", { name: "Save Named Plan" }));
    expect(screen.getByRole("combobox", { name: "Saved plan library" })).toHaveTextContent(
      "Baseline",
    );

    await user.click(screen.getByRole("button", { name: "Duplicate Selected Plan" }));
    expect(screen.getByRole("combobox", { name: "Saved plan library" })).toHaveTextContent(
      "Baseline Copy",
    );
    await user.click(screen.getByRole("button", { name: "Load Selected Plan" }));
    expect(screen.getByRole("status")).toHaveTextContent(/loaded/i);
    await user.click(screen.getByRole("button", { name: "Delete Selected Plan" }));
    expect(screen.getByRole("combobox", { name: "Saved plan library" })).not.toHaveTextContent(
      "Baseline Copy",
    );
  });

  it("reports corrupt library recovery without preventing a new save", async () => {
    storage.setItem(VARIATION_PLAN_LIBRARY_KEY, "{broken");
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    expect(screen.getByRole("status")).toHaveTextContent(/corrupt/i);
    await user.type(screen.getByRole("textbox", { name: "Plan name" }), "Recovered");
    await user.click(screen.getByRole("button", { name: "Save Named Plan" }));
    expect(storage.getItem(VARIATION_PLAN_LIBRARY_KEY)).toContain("Recovered");
  });
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
    expect(screen.getByRole("button", { name: "Matrix SVG" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Matrix Selected CSV" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Matrix Plot Definition JSON" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Scatter SVG" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Scatter Plot Definition JSON" })).toBeEnabled();
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
    expect(screen.getByRole("spinbutton", { name: "Quiet-zone RMS threshold millimetres" })).toHaveValue(5);
    expect(screen.getByRole("img", { name: /interactive all-trial swing arcs/i })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /RMS positional variability and quiet zones/i })).toBeInTheDocument();
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
