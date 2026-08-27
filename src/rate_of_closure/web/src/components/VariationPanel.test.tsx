import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CATEGORY_LAUNCH,
  CATEGORY_SWING,
  planFromJson,
  planToJson,
  type VariationPlanTs,
} from "../model/variation";
import {
  parsePersistedVariationPlan,
  persistedVariationPlanJson,
} from "../model/variationPersistedPlan";
import { VARIATION_PLAN_LIBRARY_KEY } from "../model/variationPlanLibrary";
import { VariationPanel } from "./VariationPanel";
import { saveBallSetupPreference } from "../model/ballSetupPersistence";
import {
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "../model/spatialTarget";

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
const SHOULDER_TORQUE = `${CATEGORY_SWING}.shoulder_commanded_torque_offset_nm`;
const WRIST_TORQUE = `${CATEGORY_SWING}.wrist_commanded_torque_offset_nm`;

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
      specId: "speed",
      timeWindowS: null,
      pointIds: [],
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
      specIds: ["speed", "angle"],
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
  it("authors exact half-open localized torque loci only in swing mode", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    expect(within(screen.getByLabelText("Variable 1")).queryByRole("option", {
      name: "Shoulder Commanded Torque Offset",
    })).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Pipeline"), "swing");
    const variable = screen.getByLabelText("Variable 1");
    expect(within(variable).getByRole("option", {
      name: "Shoulder Commanded Torque Offset",
    })).toBeInTheDocument();
    await user.selectOptions(variable, SHOULDER_TORQUE);

    const joint = screen.getByRole("combobox", { name: "Shoulder Commanded Torque Offset topological joint" });
    expect(joint).toHaveValue("joint.shoulder");
    expect(joint).toBeDisabled();
    expect(joint).toHaveAttribute("title", expect.stringMatching(/topological.*swing\.\*/i));
    expect(screen.getByRole("textbox", { name: "Shoulder Commanded Torque Offset window start" }))
      .toHaveAttribute("title", expect.stringMatching(/half-open/i));

    fireEvent.change(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window start",
    }), { target: { value: "0.125" } });
    fireEvent.blur(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window start",
    }));
    fireEvent.change(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window end",
    }), { target: { value: "0.375" } });
    fireEvent.blur(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window end",
    }));

    await user.selectOptions(screen.getByLabelText("Variable 1"), WRIST_TORQUE);
    expect(screen.getByRole("combobox", {
      name: "Wrist Commanded Torque Offset topological joint",
    })).toHaveValue("joint.wrist");
    expect(screen.getByRole("textbox", {
      name: "Wrist Commanded Torque Offset window start",
    })).not.toHaveValue("0.125");
  });

  it("round-trips exact localized IDs, loci, groups, and untouched precision", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    const plan: VariationPlanTs = {
      mode: "swing",
      baseVariables: {},
      noise: [
        { variableKey: SHOULDER_TORQUE, distribution: "normal", scale: 1.123456789,
          lower: null, upper: null, specId: "shoulder-window",
          timeWindowS: [0.123456789, 0.456789123], pointIds: ["joint.shoulder"] },
        { variableKey: WRIST_TORQUE, distribution: "normal", scale: 0.987654321,
          lower: null, upper: null, specId: "wrist-window",
          timeWindowS: [0.2, 0.6], pointIds: ["joint.wrist"] },
      ],
      groups: [{ groupId: "joint-torque-group", specIds: ["shoulder-window", "wrist-window"],
        matrixKind: "correlation", matrix: [[1, 0.25], [0.25, 1]] }],
      nRuns: 4, seed: 8, flightModel: "waterloo_penner",
    };
    await user.upload(screen.getByLabelText("Import variation plan JSON"), new File(
      [planToJson(plan)], "localized.json", { type: "application/json" },
    ));
    expect(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window start",
    })).toHaveValue("0.123456789");
    fireEvent.change(screen.getByRole("textbox", { name: "Seed" }), {
      target: { value: "9" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Seed" }));
    await user.type(screen.getByRole("textbox", { name: "Plan name" }), "Localized");
    await user.click(screen.getByRole("button", { name: "Save Named Plan" }));
    const stored = JSON.parse(storage.getItem(VARIATION_PLAN_LIBRARY_KEY)!) as {
      plans: Array<{ plan_document: unknown }>;
    };
    const saved = parsePersistedVariationPlan(
      JSON.stringify(stored.plans[0].plan_document),
    ).plan;
    expect(saved.seed).toBe(9);
    expect(saved.noise[0]).toMatchObject({
      specId: "shoulder-window", scale: 1.123456789,
      timeWindowS: [0.123456789, 0.456789123], pointIds: ["joint.shoulder"],
    });
    expect(saved.groups).toEqual(plan.groups);
  });

  it("fails a reversed localized window visibly without mutating storage", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    await user.selectOptions(screen.getByLabelText("Pipeline"), "swing");
    await user.selectOptions(screen.getByLabelText("Variable 1"), SHOULDER_TORQUE);
    fireEvent.change(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window start",
    }), { target: { value: "0.4" } });
    fireEvent.blur(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window start",
    }));
    fireEvent.change(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window end",
    }), { target: { value: "0.2" } });
    fireEvent.blur(screen.getByRole("textbox", {
      name: "Shoulder Commanded Torque Offset window end",
    }));
    await user.type(screen.getByRole("textbox", { name: "Plan name" }), "Invalid");
    await user.click(screen.getByRole("button", { name: "Save Named Plan" }));

    expect(screen.getByRole("alert")).toHaveTextContent(/0 ≤ start < end ≤ 1\.5 s/);
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/Cannot save plan.*start < end/i);
    expect(storage.getItem(VARIATION_PLAN_LIBRARY_KEY)).toBeNull();
  });

  it("rejects a missing imported locus without replacing current controls", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    const invalid = JSON.parse(planToJson({
      mode: "swing",
      baseVariables: {},
      noise: [{ variableKey: SHOULDER_TORQUE, distribution: "normal", scale: 1,
        lower: null, upper: null, specId: "shoulder-window",
        timeWindowS: [0.1, 0.2], pointIds: ["joint.shoulder"] }],
      nRuns: 4, seed: 8, flightModel: "waterloo_penner", groups: [],
    })) as { noise: Array<{ time_window_s: null }> };
    invalid.noise[0].time_window_s = null;
    await user.upload(screen.getByLabelText("Import variation plan JSON"), new File(
      [JSON.stringify(invalid)], "invalid-localized.json", { type: "application/json" },
    ));

    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/Cannot load plan.*finite half-open time window/i);
    expect(screen.getByLabelText("Pipeline")).toHaveValue("delivery");
    expect(screen.getByLabelText("Variable 1")).not.toHaveValue(SHOULDER_TORQUE);
  });

  it("visibly distinguishes legacy imports from canonical plan evidence", async () => {
    const user = userEvent.setup();
    render(<VariationPanel storage={storage} />);
    const input = screen.getByLabelText("Import variation plan JSON");

    await user.upload(input, new File([planToJson(importedPlan())], "legacy.json", {
      type: "application/json",
    }));
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/not evidence of historical reproducibility/i);

    await user.upload(input, new File([persistedVariationPlanJson(importedPlan())],
      "canonical.json", { type: "application/json" }));
    expect(screen.getByRole("status", { name: "Variation status" }))
      .not.toHaveTextContent(/historical reproducibility/i);
  });

  it("explains why Morris is disabled when the current context cannot round-trip", () => {
    render(<VariationPanel morrisUnavailableReason="custom club loft is unsupported" />);
    expect(screen.getByRole("button", { name: "Morris Screening" })).toBeDisabled();
    expect(screen.getByRole("status", { name: "Morris availability" })).toHaveTextContent(
      /Morris unavailable: custom club loft is unsupported/i,
    );
  });

  it("reports an aerial target without projecting it onto the ground", () => {
    const target = createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([140, 24, -3], "app"),
      tolerance: sphereTolerance(4),
      elevationSource: "absolute",
    });
    render(<VariationPanel spatialTarget={target} />);
    const summary = screen.getByRole("status", {
      name: "Variation current spatial target",
    });
    expect(summary).toHaveTextContent(/Apex gate.*140\.0 m downrange.*24\.0 m up/i);
    expect(summary).toHaveTextContent(/elevation was not coerced to zero/i);
  });

  it("presents a complete results workspace before the first run", () => {
    render(<VariationPanel storage={storage} />);

    expect(screen.getByRole("region", { name: "Variation results" })).toHaveClass("min-w-0");
    expect(screen.getByRole("heading", { name: "Ready to Analyze Variation" })).toBeVisible();
    expect(screen.getByText("Distribution Matrix")).toBeVisible();
    expect(screen.getByText("Swing Geometry")).toBeVisible();
    expect(screen.getByText("Impact and Flight")).toBeVisible();
    expect(screen.getByText("Sensitivity")).toBeVisible();
    expect(screen.getByText(/fabricated landing coordinates/)).toBeVisible();
  });

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
    await user.type(screen.getByRole("textbox", { name: "Plan name" }), "Imported V2");
    await user.click(screen.getByRole("button", { name: "Save Named Plan" }));

    const stored = JSON.parse(storage.getItem(VARIATION_PLAN_LIBRARY_KEY)!) as {
      plans: Array<{ plan_document: unknown }>;
    };
    expect(parsePersistedVariationPlan(
      JSON.stringify(stored.plans[0].plan_document),
    ).plan).toEqual(
      planFromJson(planToJson(importedPlan())),
    );
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
