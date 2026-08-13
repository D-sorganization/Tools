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

  it("uses one assertive alert owner for first and retained failures", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    await act(async () => service.calls[0].reject(new Error("first failure")));
    expect(screen.getByRole("alert", { name: "Variation status" }))
      .toHaveAttribute("aria-live", "assertive");
    expect(screen.getAllByRole("alert")).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    await act(async () => service.calls[1].resolve(completedResult(service.calls[1])));
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    await act(async () => service.calls[2].reject(new Error("retained failure")));
    expect(screen.getByRole("alert", { name: "Variation status" }))
      .toHaveTextContent("retained failure");
    expect(screen.getAllByRole("alert")).toHaveLength(1);
  });

  it("bounds failure announcements before they can displace the visual", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    await act(async () => service.calls[0].reject(new Error("x".repeat(100_000))));
    expect(screen.getByRole("alert", { name: "Variation status" }).textContent?.length)
      .toBeLessThan(530);
    expect(screen.getByRole("img", { name: "Variation analysis workflow preview" }))
      .toBeVisible();
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

  it("retains one accepted visual through same-authority loading and failure", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);
    const run = screen.getByRole("button", { name: "Run Variation Study" });
    await user.click(run);
    await act(async () => service.calls[0].resolve(completedResult(service.calls[0])));
    const summary = screen.getByRole("heading", { name: /Summary — Dispersion/ });
    expect(summary).toBeVisible();

    await user.click(run);
    const frame = screen.getByLabelText(/prior accepted result retained/i);
    expect(frame).toHaveAttribute("data-phase", "loading");
    expect(summary).toBeVisible();
    await act(async () => service.calls[1].reject(new Error("bounded failure")));
    expect(frame).toHaveAttribute("data-phase", "error");
    expect(summary).toBeVisible();
    expect(screen.getByLabelText("Variation visualization state"))
      .toHaveTextContent(/prior accepted result retained.*bounded failure/i);
    const strip = screen.getByLabelText("Variation visualization state");
    expect(strip.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(strip).not.toHaveClass("absolute");
  });

  it("invalidates retained evidence for every scientific policy edit", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    await act(async () => service.calls[0].resolve(completedResult(service.calls[0])));
    expect(screen.getByRole("heading", { name: /Summary — Dispersion/ })).toBeVisible();

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Analysis execution" }), "both",
    );

    expect(screen.getByLabelText(/analysis preview/i)).toHaveAttribute("data-phase", "empty");
    expect(screen.queryByRole("heading", { name: /Summary — Dispersion/ })).toBeNull();
  });

  it("keeps transient invalid bounds editable while invalidating accepted evidence", async () => {
    const user = userEvent.setup();
    const service = new DeferredExecutionService();
    render(<VariationPanel executionService={service} />);
    await configureTwoJointRuns(user);
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    await act(async () => service.calls[0].resolve(completedResult(service.calls[0])));

    fireEvent.change(screen.getByRole("spinbutton", { name: /lower bound/i }), {
      target: { value: "100" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: /upper bound/i }), {
      target: { value: "10" },
    });

    expect(screen.getByLabelText(/analysis preview/i)).toHaveAttribute("data-phase", "empty");
    expect(screen.queryByRole("heading", { name: /Summary — Dispersion/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Export Plan JSON" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Run Variation Study" }));
    expect(screen.getByRole("alert", { name: "Variation status" }))
      .toHaveTextContent(/cannot run/i);
    expect(service.calls).toHaveLength(1);
  });
});
