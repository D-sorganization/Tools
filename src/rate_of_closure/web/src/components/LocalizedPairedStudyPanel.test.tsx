import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import localizedPlan from "../model/__fixtures__/localized_torque_authoring_v1.json";
import type { LocalizedPairedExecutionServiceTs } from "../model/localizedAttributionExecutionService";
import { executeLocalizedPairedWork } from "../model/localizedAttributionExecution";
import { planFromJson } from "../model/variation";
import { LocalizedPairedStudyPanel } from "./LocalizedPairedStudyPanel";

class DeferredService implements LocalizedPairedExecutionServiceTs {
  calls: Array<{
    request: Parameters<LocalizedPairedExecutionServiceTs["execute"]>[0];
    controls: Parameters<LocalizedPairedExecutionServiceTs["execute"]>[1];
    resolve: (result: Awaited<ReturnType<LocalizedPairedExecutionServiceTs["execute"]>>) => void;
    reject: (error: unknown) => void;
  }> = [];
  execute(request: Parameters<LocalizedPairedExecutionServiceTs["execute"]>[0],
    controls: Parameters<LocalizedPairedExecutionServiceTs["execute"]>[1]) {
    return new Promise<Awaited<ReturnType<LocalizedPairedExecutionServiceTs["execute"]>>>(
      (resolve, reject) => this.calls.push({ request, controls, resolve, reject }),
    );
  }
}

const plan = () => ({ ...planFromJson(JSON.stringify(localizedPlan)), groups: [] });

afterEach(cleanup);

describe("explicit localized paired study UI", () => {
  it("shows exact loci and runs separately with 2N progress", async () => {
    const user = userEvent.setup();
    const service = new DeferredService();
    let accepted = null;
    render(<LocalizedPairedStudyPanel plan={plan()} service={service}
      onAuthorityChange={(authority) => { accepted = authority; }} />);

    await user.click(screen.getByRole("button", {
      name: "Configure & Run Separate Paired Study…",
    }));
    expect(screen.getByText(/joint\.shoulder · \[0\.123456789, 0\.456789123\)/i))
      .toBeInTheDocument();
    expect(screen.getByText(/global factors stay fixed/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Confirm & Run 4 Explicit Trials" }));

    expect(service.calls).toHaveLength(1);
    act(() => service.calls[0].controls.onProgress({ completedRuns: 1, totalRuns: 4 }));
    expect(screen.getByRole("log", { name: "Paired study status" }))
      .toHaveTextContent(/1\/4 explicit trials/i);
    const result = await executeLocalizedPairedWork(service.calls[0].request, () => undefined);
    await act(async () => service.calls[0].resolve(result));
    expect(accepted).toEqual(result.authority);
    expect(screen.getByRole("log", { name: "Paired study status" }))
      .toHaveTextContent(/no causal inference/i);
  });

  it("cancel and failure preserve prior authority and allow rerun", async () => {
    const user = userEvent.setup();
    const service = new DeferredService();
    let accepted = null;
    render(<LocalizedPairedStudyPanel plan={plan()} service={service}
      onAuthorityChange={(authority) => { accepted = authority; }} />);
    await user.click(screen.getByRole("button", { name: /configure & run/i }));
    await user.click(screen.getByRole("button", { name: /confirm & run/i }));
    const firstResult = await executeLocalizedPairedWork(service.calls[0].request, () => undefined);
    await act(async () => service.calls[0].resolve(firstResult));
    expect(accepted).toEqual(firstResult.authority);

    await user.click(screen.getByRole("button", { name: /configure & run/i }));
    await user.click(screen.getByRole("button", { name: /confirm & run/i }));
    await user.click(screen.getByRole("button", { name: "Cancel Separate Paired Study" }));
    expect(service.calls[1].controls.signal.aborted).toBe(true);
    expect(accepted).toEqual(firstResult.authority);

    await user.click(screen.getByRole("button", { name: /configure & run/i }));
    await user.click(screen.getByRole("button", { name: /confirm & run/i }));
    await act(async () => service.calls[2].reject(new Error("planned failure")));
    expect(accepted).toEqual(firstResult.authority);
    expect(screen.getByRole("log", { name: "Paired study status" }))
      .toHaveTextContent(/failed before authority replacement/i);
  });

  it("fails closed for zero delta before service execution", async () => {
    const user = userEvent.setup();
    const service = new DeferredService();
    render(<LocalizedPairedStudyPanel plan={plan()} service={service}
      onAuthorityChange={() => undefined} />);
    await user.click(screen.getByRole("button", { name: /configure & run/i }));
    fireEvent.change(screen.getByLabelText("Planted delta shoulder-window"), {
      target: { value: "0" },
    });
    await user.click(screen.getByRole("button", { name: /confirm & run/i }));
    expect(service.calls).toHaveLength(0);
    expect(screen.getByRole("log", { name: "Paired study status" }))
      .toHaveTextContent(/delta must be nonzero/i);
  });
});
