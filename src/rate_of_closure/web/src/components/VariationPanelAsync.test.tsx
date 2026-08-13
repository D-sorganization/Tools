import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  VariationExecutionControls,
  VariationExecutionRequest,
  VariationExecutionResult,
  VariationExecutionService,
} from "../model/variationExecutionService";
import { runVariation, type VariationDatasetTs } from "../model/variation";
import { VariationPanel } from "./VariationPanel";

interface PendingExecution {
  request: VariationExecutionRequest;
  controls: VariationExecutionControls;
  resolve: (result: VariationExecutionResult) => void;
  reject: (reason: unknown) => void;
}

class DeferredExecutionService implements VariationExecutionService {
  readonly calls: PendingExecution[] = [];

  execute(
    request: VariationExecutionRequest,
    controls: VariationExecutionControls,
  ): Promise<VariationExecutionResult> {
    return new Promise((resolve, reject) => {
      this.calls.push({ request, controls, resolve, reject });
    });
  }
}

const completedResult = (
  call: PendingExecution,
  successful = true,
): VariationExecutionResult => {
  const dataset = runVariation(call.request.plan);
  if (!successful) {
    const failedDataset: VariationDatasetTs = {
      ...dataset,
      outputs: dataset.outputs.map((row) => row.map(() => null)),
      success: dataset.success.map(() => false),
    };
    return { dataset: failedDataset, sensitivity: null, ensemble: null };
  }
  return { dataset, sensitivity: null, ensemble: null };
};

const configureTwoJointRuns = async (user: ReturnType<typeof userEvent.setup>) => {
  fireEvent.change(screen.getByRole("textbox", { name: "Runs" }), {
    target: { value: "2" },
  });
  fireEvent.blur(screen.getByRole("textbox", { name: "Runs" }));
  await user.selectOptions(
    screen.getByRole("combobox", { name: "Analysis execution" }),
    "all_together",
  );
};

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VariationPanel asynchronous Monte Carlo execution", () => {
  it("reports real completed work while the injected authority is busy", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);

    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));

    expect(service.calls).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Run Variation Study" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel Variation Study" })).toBeEnabled();
    expect(screen.getByRole("progressbar", { name: "Variation execution progress" }))
      .toHaveAccessibleName("Variation execution progress");
    expect(screen.getByRole("progressbar", { name: "Variation execution progress" }))
      .toHaveAttribute("max", "2");

    act(() => {
      service.calls[0].controls.onProgress({
        completedRuns: 1,
        totalRuns: 2,
        phase: "joint",
      });
    });
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/running joint variation: 1\/2 evaluated runs/i);

    await act(async () => {
      service.calls[0].resolve(completedResult(service.calls[0]));
    });
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/done: 2\/2 joint runs/i);
    expect(screen.getByRole("button", { name: "Run Variation Study" })).toBeEnabled();
  });

  it("cancels cooperatively and immediately permits a safe rerun", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);

    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    const first = service.calls[0];
    await user.click(screen.getByRole("button", { name: "Cancel Variation Study" }));

    expect(first.controls.signal.aborted).toBe(true);
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/cancelled/i);
    expect(screen.getByRole("button", { name: "Run Variation Study" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    expect(service.calls).toHaveLength(2);
    await act(async () => {
      service.calls[1].resolve(completedResult(service.calls[1]));
    });
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/done: 2\/2 joint runs/i);
  });

  it("suppresses a stale generation that resolves after its rerun", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);

    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    const stale = service.calls[0];
    await user.click(screen.getByRole("button", { name: "Cancel Variation Study" }));
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    const current = service.calls[1];
    await act(async () => {
      current.resolve(completedResult(current));
    });
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/done: 2\/2 joint runs/i);

    await act(async () => {
      stale.resolve(completedResult(stale, false));
    });
    expect(screen.getByRole("status", { name: "Variation status" }))
      .toHaveTextContent(/done: 2\/2 joint runs/i);
    expect(screen.getByRole("status", { name: "Variation status" }))
      .not.toHaveTextContent(/failed/i);
  });

  it("aborts the active generation on unmount", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    const view = render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));

    const active = service.calls[0];
    view.unmount();

    expect(active.controls.signal.aborted).toBe(true);
    act(() => {
      active.controls.onProgress({ completedRuns: 2, totalRuns: 2, phase: "joint" });
      active.resolve(completedResult(active));
    });
  });
});
