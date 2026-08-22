import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { defaultMorrisAuthorityBase } from "../model/morrisWorkflowDefaults";
import type { DurableEnsembleRunner } from "../model/durableEnsembleWorkerClient";
import type { DurableEnsembleJob } from "../model/durableEnsembleAuthorityContract";
import type { VariationPlanTs } from "../model/variation";
import { DurableEnsembleWorkflowPanel } from "./DurableEnsembleWorkflowPanel";

const plan: VariationPlanTs = {
  mode: "swing", baseVariables: { "swing_sim.swing.yaw_deg": 0 },
  noise: [{ variableKey: "swing_sim.swing.yaw_deg", distribution: "normal", scale: 1,
    lower: null, upper: null, timeWindowS: null, pointIds: [] }],
  nRuns: 5, seed: 7, flightModel: "waterloo_penner",
  ballSetup: { supportMode: "tee", teeHeightM: 0.0381 }, groups: [],
};
const base = defaultMorrisAuthorityBase(getClub("Driver 10.5°"), {
  ...DEFAULT_SCENARIO, clubheadSpeedMph: 113,
});

afterEach(cleanup);

describe("DurableEnsembleWorkflowPanel", () => {
  it("gates on authority capability and submits a path-free request", async () => {
    const run = vi.fn<DurableEnsembleRunner["run"]>(() => ({
      promise: new Promise<DurableEnsembleJob>(() => undefined), cancel: vi.fn(),
    }));
    const runner: DurableEnsembleRunner = {
      capability: vi.fn(async () => ({
        available: true, apiPrefix: "/api/rate-of-closure/v1" as const,
      })),
      run, close: vi.fn(),
    };
    render(<DurableEnsembleWorkflowPanel plan={plan} base={base} runner={runner} />);
    const button = await screen.findByRole("button", { name: "Run or Resume Ensemble" });
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
    expect(run).toHaveBeenCalledOnce();
    const request = run.mock.calls[0][0];
    expect(request).toMatchObject({ archive_id: "proximal-distal-ensemble", chunk_size: 256 });
    expect(JSON.stringify(request)).not.toMatch(/directory|[A-Z]:\\/i);
  });
});
