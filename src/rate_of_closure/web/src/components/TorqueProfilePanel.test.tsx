import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  type DoublePendulumRunConfig,
} from "../model/doublePendulum";
import {
  TORQUE_PROFILE_STORAGE_KEY,
  saveTorqueProfileLibrary,
  starterTorqueProfile,
} from "../model/torqueProfileEditor";
import { TorqueProfilePanel } from "./TorqueProfilePanel";
import { type SimulationRunTs } from "../model/simulation";

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
beforeEach(() => { storage = new MemoryStorage(); });

function Harness({ run = null }: { run?: SimulationRunTs | null }) {
  const [config, setConfig] = useState<DoublePendulumRunConfig>(
    PASSIVE_DOUBLE_PENDULUM_RUN,
  );
  return (
    <TorqueProfilePanel
      sourceKind="double_pendulum"
      runConfig={config}
      onRunConfigChange={setConfig}
      storage={storage}
      run={run}
    />
  );
}

afterEach(() => {
  cleanup();
});

describe("TorqueProfilePanel", () => {
  it("makes passive versus prescribed execution explicit", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    expect(screen.getByRole("status", { name: "Torque execution status" }))
      .toHaveTextContent(/passive/i);

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Torque execution mode" }),
      "prescribed",
    );
    expect(screen.getByRole("status", { name: "Torque execution status" }))
      .toHaveTextContent(/prescribed.*Web Starter Drive/i);
  });

  it("saves direct c0-first equations into the persistent profile library", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.clear(screen.getByRole("textbox", { name: "Profile ID" }));
    await user.type(
      screen.getByRole("textbox", { name: "Profile ID" }),
      "profile.custom.v1",
    );
    await user.clear(screen.getByRole("textbox", { name: "Profile name" }));
    await user.type(screen.getByRole("textbox", { name: "Profile name" }), "Custom Drive");
    await user.clear(screen.getByRole("textbox", { name: "Shoulder coefficients" }));
    await user.type(screen.getByRole("textbox", { name: "Shoulder coefficients" }), "12, -3");
    await user.click(screen.getByRole("button", { name: /Save Torque Profile/i }));

    const stored = storage.getItem(TORQUE_PROFILE_STORAGE_KEY) ?? "";
    expect(stored).toContain("profile.custom.v1");
    expect(stored).toContain('"coefficients":[12,-3]');
    expect(screen.getByRole("status", { name: "Torque profile editor status" }))
      .toHaveTextContent(/saved/i);
  });

  it("fits the accessible sample table into ascending c0-first equations", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const samples = screen.getByRole("textbox", { name: "Torque sample rows" });
    await user.clear(samples);
    await user.type(samples, "0,10,0\n1,12,2\n2,14,4");
    await user.selectOptions(screen.getByRole("combobox", { name: "Polynomial degree" }), "1");
    await user.click(screen.getByRole("button", { name: /Fit Torque Curves/i }));

    expect(screen.getByRole("textbox", { name: "Shoulder coefficients" }))
      .toHaveValue("10, 2");
    expect(screen.getByRole("textbox", { name: "Wrist coefficients" }))
      .toHaveValue("0, 2");
    expect(screen.getByRole("table", { name: "Torque sample preview" }))
      .toHaveTextContent("14");
    expect(screen.getByRole("img", { name: /fitted shoulder and wrist torque curves/i }))
      .toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Save Torque Profile/i }));
    expect(storage.getItem(TORQUE_PROFILE_STORAGE_KEY)).toContain('"source":"drawn"');
  });

  it("marks a current-run fit as fitted-run provenance", async () => {
    const user = userEvent.setup();
    const run = {
      torqueRun: {
        mode: "passive",
        profileId: null,
        appliedTorqueHistory: Array.from({ length: 1501 }, (_, index) => index / 1000).map((timeS) => ({
          timeS,
          torquesNm: { "joint.shoulder": 2 + timeS, "joint.wrist": -timeS },
        })),
      },
      impactOutcome: { status: "miss" },
    } as unknown as SimulationRunTs;
    render(<Harness run={run} />);
    await user.selectOptions(screen.getByRole("combobox", { name: "Polynomial degree" }), "1");
    await user.click(screen.getByRole("button", { name: "Fit Current Run to Profile" }));
    const editorRows = (screen.getByRole("textbox", {
      name: "Torque sample rows",
    }) as HTMLTextAreaElement).value;
    expect(editorRows.split("\n")).toHaveLength(101);
    expect(editorRows.split("\n")[0].startsWith("0,")).toBe(true);
    const editorLines = editorRows.split("\n");
    expect(editorLines[editorLines.length - 1].startsWith("1.5,")).toBe(true);
    expect(screen.getByRole("status", { name: "Torque sample display status" }))
      .toHaveTextContent(/101 representative rows from 1501 fitted samples.*25 of 101/i);
    expect(within(screen.getByRole("table", { name: "Torque sample preview" })).getAllByRole("row"))
      .toHaveLength(26);
    expect(screen.getByRole("img", { name: /fitted shoulder and wrist/i }).querySelectorAll("circle"))
      .toHaveLength(82);
    await user.click(screen.getByRole("button", { name: /Save Torque Profile/i }));
    const saved = storage.getItem(TORQUE_PROFILE_STORAGE_KEY) ?? "";
    expect(saved).toContain('"source":"fitted_run"');
    expect(saved).toContain('"impact_outcome":"miss"');
  });

  it("loads persisted profiles and supports canonical import and export", async () => {
    const profile = starterTorqueProfile();
    saveTorqueProfileLibrary([profile], storage);
    const user = userEvent.setup();
    const createUrl = vi.fn(() => "blob:test");
    const revokeUrl = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeUrl });
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    render(<Harness />);
    expect(screen.getByRole("combobox", { name: "Torque profile library" }))
      .toHaveValue(profile.profileId);

    const file = new File([profile.dumps()], "profile.json", { type: "application/json" });
    await user.upload(screen.getByLabelText("Import torque profile JSON"), file);
    expect(await screen.findByRole("status", { name: "Torque profile editor status" }))
      .toHaveTextContent(/imported/i);
    await user.click(screen.getByRole("button", { name: "Export Selected JSON" }));
    expect(createUrl).toHaveBeenCalledOnce();
    expect(anchorClick).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:test");
  });
});
