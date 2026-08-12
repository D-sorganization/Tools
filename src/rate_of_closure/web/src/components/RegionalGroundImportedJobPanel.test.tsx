import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import jobFixture from "../model/__fixtures__/regional_ground_execution_job_golden_v1.json";
import resultFixture from "../model/__fixtures__/regional_ground_execution_result_golden_v1.json";
import {
  qualifiedRegionalGroundAuthorityCapability,
  unavailableRegionalGroundAuthorityCapability,
} from "../model/regionalGroundAuthority";
import type {
  RegionalGroundAuthorityClient,
  RegionalGroundAuthorityJobStatus,
} from "../model/regionalGroundAuthorityClient";
import { regionalGroundExecutionResultFromJson } from "../model/regionalGroundExecutionResult";
import { parseRegionalGroundExecutionJob } from "../model/regionalGroundExecutionJob";
import { useRegionalGroundExecutionWorkspace } from "../hooks/useRegionalGroundExecutionWorkspace";
import { RegionalGroundImportedJobPanel } from "./RegionalGroundImportedJobPanel";

const job = parseRegionalGroundExecutionJob(jobFixture.job);
const result = regionalGroundExecutionResultFromJson(JSON.stringify(resultFixture.result));
const status = (
  state: RegionalGroundAuthorityJobStatus["status"],
  completed: number,
): RegionalGroundAuthorityJobStatus => ({
  schema_version: "rate-of-closure/regional-ground-authority-job-status/v1",
  job_id: job.job_id,
  job_sha256: job.job_sha256,
  status: state,
  completed,
  total: job.execution_options.max_trials,
  result_available: state === "succeeded",
  failure: state === "failed" ? { code: "execution_failed", stage: "executor" } : null,
});
const client = (overrides: Partial<RegionalGroundAuthorityClient> = {}) => ({
  capability: vi.fn().mockResolvedValue(qualifiedRegionalGroundAuthorityCapability()),
  submit: vi.fn().mockResolvedValue(status("queued", 0)),
  status: vi.fn().mockResolvedValue(status("succeeded", 4)),
  cancel: vi.fn().mockResolvedValue(status("cancelled", 0)),
  result: vi.fn().mockResolvedValue(result),
  ...overrides,
}) satisfies RegionalGroundAuthorityClient;
const browserFile = (source: string, name: string) => ({
  name,
  size: new TextEncoder().encode(source).byteLength,
  arrayBuffer: async () => Uint8Array.from(new TextEncoder().encode(source)).buffer,
});
const uploadJob = async (name = "regional-job.json"): Promise<void> => {
  const source = JSON.stringify(jobFixture.job);
  fireEvent.change(screen.getByLabelText("Import regional-ground execution job JSON"), {
    target: { files: [browserFile(source, name)] },
  });
  expect(await screen.findByText(new RegExp(`Loaded ${name}`))).toBeInTheDocument();
};

function Harness(props: {
  readonly client: RegionalGroundAuthorityClient;
  readonly saveJob?: ReturnType<typeof vi.fn>;
  readonly saveResult?: ReturnType<typeof vi.fn>;
}) {
  const workspace = useRegionalGroundExecutionWorkspace({
    client: props.client,
    authority: { query: props.client.capability, pollIntervalMs: 60_000 },
    executionPollIntervalMs: 250,
  });
  return <RegionalGroundImportedJobPanel workspace={workspace}
    saveJob={props.saveJob} saveResult={props.saveResult} />;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("RegionalGroundImportedJobPanel", () => {
  it("requires exact import and explicit confirmation before executing and saving", async () => {
    const authority = client();
    const saveResult = vi.fn();
    render(<Harness client={authority} saveResult={saveResult} />);

    await uploadJob();
    expect(screen.getByText(job.job_sha256)).toBeInTheDocument();
    expect(screen.getByText(job.input_sha256)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run imported study" })).toBeDisabled();

    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(
      screen.getByRole("button", { name: "Run imported study" }),
    );
    expect(authority.submit).toHaveBeenCalledWith(job, expect.any(AbortSignal));
    expect(screen.getByRole("button", { name: "Cancel study" })).toBeEnabled();

    await waitFor(() => expect(screen.getByLabelText("Imported study execution status"))
      .toHaveTextContent(/succeeded/i));
    expect(authority.result).toHaveBeenCalledWith(job, expect.any(AbortSignal));

    await user.click(
      screen.getByRole("button", { name: /download canonical regional-ground study result/i }),
    );
    expect(saveResult).toHaveBeenCalledWith(result);
  });

  it("preserves the accepted job when a replacement file is invalid", async () => {
    const authority = client();
    render(<Harness client={authority} />);
    await uploadJob("accepted.json");

    fireEvent.change(screen.getByLabelText("Import regional-ground execution job JSON"), {
      target: { files: [browserFile("{}", "invalid.json")] },
    });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/prior accepted job was preserved/i));
    expect(screen.getByText("accepted.json")).toBeInTheDocument();
    expect(screen.getByText(job.job_sha256)).toBeInTheDocument();
  });

  it("fails closed when the local Python authority is not qualified", async () => {
    const unavailable = unavailableRegionalGroundAuthorityCapability(
      "execution_profile_unqualified",
      "Exact execution profile is not qualified.",
    );
    const authority = client({ capability: vi.fn().mockResolvedValue(unavailable) });
    render(<Harness client={authority} />);
    await uploadJob();
    await userEvent.setup().click(screen.getByRole("checkbox"));

    expect(await screen.findByText(/execution_profile_unqualified/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run imported study" })).toBeDisabled();
    expect(authority.submit).not.toHaveBeenCalled();
  });

  it("cancels the exact active job and publishes terminal cancellation", async () => {
    const authority = client({
      cancel: vi.fn().mockResolvedValue(status("cancelled", 0)),
    });
    render(<Harness client={authority} />);
    await uploadJob();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Run imported study" }));
    await user.click(screen.getByRole("button", { name: "Cancel study" }));

    expect(authority.cancel).toHaveBeenCalledWith(job, expect.any(AbortSignal));
    expect(screen.getByText(/cancelled/i)).toBeInTheDocument();
    expect(screen.getByRole("button", {
      name: /download canonical regional-ground study result/i,
    })).toBeDisabled();
  });

  it("does not replace an accepted job when execution starts during async validation", async () => {
    const authority = client();
    render(<Harness client={authority} />);
    await uploadJob("accepted.json");
    let resolveBytes!: (value: ArrayBuffer) => void;
    const pendingBytes = new Promise<ArrayBuffer>((resolve) => { resolveBytes = resolve; });
    const source = JSON.stringify(jobFixture.job);
    fireEvent.change(screen.getByLabelText("Import regional-ground execution job JSON"), {
      target: { files: [{ name: "replacement.json", size: source.length,
        arrayBuffer: () => pendingBytes }] },
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Run imported study" }));
    await act(async () => {
      resolveBytes(Uint8Array.from(new TextEncoder().encode(source)).buffer);
      await pendingBytes;
    });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/prior accepted job was preserved/i));
    expect(screen.getByText("accepted.json")).toBeInTheDocument();
    expect(screen.queryByText("replacement.json")).not.toBeInTheDocument();
  });
});
