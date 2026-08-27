import { describe, expect, it } from "vitest";

import rawContract from "../../../locus_execution_capabilities.v1.json";
import {
  capabilityFor,
  parseLocusExecutionContract,
} from "./locusExecutionCapabilities";
import {
  GROUND_NORMAL_RESTITUTION_KEY,
  GROUND_ROLLING_RESISTANCE_KEY,
  VARIABLE_REGISTRY,
} from "./variationRegistry";

const authorityKeys = rawContract.capabilities.map(({ variable_key: key }) => key);
const sharedBrowserRegistryKeys = VARIABLE_REGISTRY
  .map(({ key }) => key)
  .filter((key) => ![
    GROUND_NORMAL_RESTITUTION_KEY,
    GROUND_ROLLING_RESISTANCE_KEY,
  ].includes(key));

interface MutableCapability {
  time_window_policy: string;
  whole_run: boolean;
  [field: string]: unknown;
}

interface MutableContract {
  capabilities: MutableCapability[];
}

describe("locus execution capability authority", () => {
  it("classifies every shared registry input from the packaged authority", () => {
    const contract = parseLocusExecutionContract(rawContract, authorityKeys);
    expect(contract.schemaVersion).toBe("rate-locus-execution-capabilities/v1");
    expect(contract.mode).toBe("multi_adapter");
    expect(contract.sourceKind).toBe("declared_by_adapter");
    expect([...contract.capabilities.keys()].sort()).toEqual(
      [...authorityKeys].sort(),
    );
    expect([...contract.capabilities.values()].filter(({ supported }) => supported))
      .toHaveLength(19);
    for (const key of sharedBrowserRegistryKeys) expect(contract.capabilities.has(key)).toBe(true);
  });

  it("distinguishes whole-run values from exact topological joint windows", () => {
    expect(capabilityFor("swing_sim.swing.yaw_deg")).toMatchObject({
      adapterId: "global_simulation_value/v1",
      wholeRun: true,
      timeWindowPolicy: "forbidden",
      pointLocusPolicy: "forbidden",
      pointIds: [],
    });
    expect(capabilityFor(
      "swing_sim.swing.shoulder_commanded_torque_offset_nm",
    )).toMatchObject({
      adapterId: "localized_joint_torque_offset/v1",
      wholeRun: false,
      timeWindowPolicy: "required_half_open_seconds",
      pointLocusPolicy: "required_exact_topological",
      pointIds: ["joint.shoulder"],
    });
  });

  it.each<[string, (payload: MutableContract) => void, RegExp]>([
    ["missing registry row", (payload) => { payload.capabilities.pop(); }, /registry coverage/],
    ["duplicate key", (payload) => { payload.capabilities.push({ ...payload.capabilities[0] }); }, /duplicate variable_key/],
    ["unknown timing policy", (payload) => { payload.capabilities[0].time_window_policy = "sometimes"; }, /time_window_policy/],
    ["contradictory whole-run row", (payload) => {
      payload.capabilities[0].whole_run = true;
      payload.capabilities[0].time_window_policy = "required_half_open_seconds";
    }, /whole-run capability cannot require/],
  ])("rejects %s", (_name, mutate, message) => {
    const payload = structuredClone(rawContract) as MutableContract;
    mutate(payload);
    expect(() => parseLocusExecutionContract(payload, authorityKeys)).toThrow(message);
  });

  it("fails closed for undeclared variables", () => {
    expect(() => capabilityFor("swing_sim.swing.future_undeclared_input"))
      .toThrow(/not declared/);
  });
});
