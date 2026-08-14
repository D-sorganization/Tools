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
  RegionalGroundAuthorityJobStatus,
  RegionalGroundPreparationAuthorityClient,
} from "../model/regionalGroundAuthorityClient";
import { regionalGroundExecutionResultFromJson } from "../model/regionalGroundExecutionResult";
import {
  parseRegionalGroundExecutionJob,
  parseRegionalGroundExecutionLaunch,
} from "../model/regionalGroundExecutionJob";
import { regionalGroundVariationRequestFromJson } from "../model/regionalGroundVariationRequestWire";
import {
  useRegionalGroundExecutionWorkspace,
  type RegionalGroundCurrentEditorSource,
} from "../hooks/useRegionalGroundExecutionWorkspace";
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
const client = (overrides: Partial<RegionalGroundPreparationAuthorityClient> = {}) => ({
  capability: vi.fn().mockResolvedValue(qualifiedRegionalGroundAuthorityCapability()),
  prepare: vi.fn().mockResolvedValue(job),
  submit: vi.fn().mockResolvedValue(status("queued", 0)),
  status: vi.fn().mockResolvedValue(status("succeeded", 4)),
  cancel: vi.fn().mockResolvedValue(status("cancelled", 0)),
  result: vi.fn().mockResolvedValue(result),
  ...overrides,
}) satisfies RegionalGroundPreparationAuthorityClient;
const variationRequest = regionalGroundVariationRequestFromJson(
  JSON.stringify(job.variation_request),
);
const currentEditors = (
  request = variationRequest,
): RegionalGroundCurrentEditorSource => ({
  launch: job.launch,
  variationRequestPort: {
    snapshot: () => request,
    apply: vi.fn(),
  },
});
const browserFile = (source: string, name: string) => ({
  name,
  size: new TextEncoder().encode(source).byteLength,
  arrayBuffer: async () => Uint8Array.from(new TextEncoder().encode(source)).buffer,
});
const uploadJob = async (name = "regional-job.json"): Promise<void> => {
  const source = JSON.stringify(jobFixture.job);
  fireEvent.change(screen.getByTestId("regional-ground-execution-job-file-input"), {
    target: { files: [browserFile(source, name)] },
  });
  expect(await screen.findByText(new RegExp(`Loaded ${name}`))).toBeInTheDocument();
};

function Harness(props: {
  readonly client: RegionalGroundPreparationAuthorityClient;
  readonly saveJob?: ReturnType<typeof vi.fn>;
  readonly saveResult?: ReturnType<typeof vi.fn>;
  readonly preparationSource?: RegionalGroundCurrentEditorSource;
}) {
  const workspace = useRegionalGroundExecutionWorkspace({
    client: props.client,
    authority: { query: props.client.capability, pollIntervalMs: 60_000 },
    executionPollIntervalMs: 250,
    preparationSource: props.preparationSource,
    preparationJobIdFactory: () => job.job_id,
  });
  return <RegionalGroundImportedJobPanel workspace={workspace}
    saveJob={props.saveJob} saveResult={props.saveResult} />;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("RegionalGroundImportedJobPanel", () => {
  it("disables preparation while the current variation snapshot is invalid", async () => {
    const authority = client();
    const invalidEditors: RegionalGroundCurrentEditorSource = {
      ...currentEditors(),
      variationRequestPort: {
        snapshot: () => { throw new Error("illustrative surface is not validated"); },
        apply: vi.fn(),
      },
    };
    render(<Harness client={authority} preparationSource={invalidEditors} />);

    const prepare = await screen.findByRole("button", { name: "Prepare Current Job" });
    expect(prepare).toBeDisabled();
    expect(screen.getByText(/Current-editor preparation is unavailable until/))
      .toBeInTheDocument();
    expect(authority.prepare).not.toHaveBeenCalled();
  });

  it("prepares current editors transactionally without confirming or running", async () => {
    const authority = client();
    render(<Harness client={authority} preparationSource={currentEditors()} />);

    const prepare = await screen.findByRole("button", { name: "Prepare Current Job" });
    expect(prepare).toBeEnabled();
    await userEvent.setup().click(prepare);

    await waitFor(() => expect(authority.prepare).toHaveBeenCalledTimes(1));
    expect(authority.submit).not.toHaveBeenCalled();
    expect(screen.getByText(job.job_sha256)).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Run imported study" })).toBeDisabled();
    expect(screen.getByText(/fixed Waterloo\/Penner flight profile/i)).toBeInTheDocument();
    expect(screen.getByText(/calibration provenance is unvalidated/i)).toBeInTheDocument();
  });

  it("preserves an imported job when current-editor preparation fails", async () => {
    const authority = client({ prepare: vi.fn().mockRejectedValue(new Error("invalid")) });
    render(<Harness client={authority} preparationSource={currentEditors()} />);
    await uploadJob("accepted.json");

    await userEvent.setup().click(screen.getByRole("button", { name: "Prepare Current Job" }));

    await waitFor(() => expect(screen.getByRole("alert"))
      .toHaveTextContent(/prior accepted job was preserved/i));
    expect(screen.getByText("accepted.json")).toBeInTheDocument();
    expect(screen.getByText(job.job_sha256)).toBeInTheDocument();
  });

  it("discards a prepared response when a relevant editor revision changes", async () => {
    let resolvePreparation!: (value: typeof job) => void;
    const pending = new Promise<typeof job>((resolve) => {
      resolvePreparation = resolve;
    });
    const authority = client({ prepare: vi.fn().mockReturnValue(pending) });
    const initial = currentEditors();
    const changed: RegionalGroundCurrentEditorSource = {
      ...currentEditors(),
      launch: parseRegionalGroundExecutionLaunch({
        ...job.launch,
        azimuth_angle_rad: 0.01,
      }),
    };
    const view = render(<Harness client={authority} preparationSource={initial} />);
    await userEvent.setup().click(await screen.findByRole("button", {
      name: "Prepare Current Job",
    }));

    view.rerender(<Harness client={authority} preparationSource={changed} />);
    await act(async () => {
      resolvePreparation(job);
      await pending;
    });

    expect(await screen.findByText(/stale response was discarded/i)).toBeInTheDocument();
    expect(screen.queryByText(job.job_sha256)).not.toBeInTheDocument();
    expect(authority.submit).not.toHaveBeenCalled();
  });

  it("accepts an in-flight response after an analysis-only editor change", async () => {
    let resolvePreparation!: (value: typeof job) => void;
    const pending = new Promise<typeof job>((resolve) => {
      resolvePreparation = resolve;
    });
    const authority = client({ prepare: vi.fn().mockReturnValue(pending) });
    const view = render(<Harness client={authority} preparationSource={currentEditors()} />);
    await userEvent.setup().click(await screen.findByRole("button", {
      name: "Prepare Current Job",
    }));

    // A new source/port identity represents a workspace render, while the canonical
    // launch and variation request sent to Python remain exactly unchanged.
    view.rerender(<Harness client={authority} preparationSource={currentEditors()} />);
    await act(async () => {
      resolvePreparation(job);
      await pending;
    });

    expect(await screen.findByText(job.job_sha256)).toBeInTheDocument();
    expect(screen.queryByText(/stale response was discarded/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/prepared job is stale/i)).not.toBeInTheDocument();
  });

  it("retains but invalidates a prepared job after relevant editor input drift", async () => {
    const authority = client();
    const initial = currentEditors();
    const changedVariationRequest = regionalGroundVariationRequestFromJson(JSON.stringify({
      ...job.variation_request,
      result_id: "changed-editor-result",
    }));
    const changed = currentEditors(changedVariationRequest);
    const view = render(<Harness client={authority} preparationSource={initial} />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Prepare Current Job" }));
    await screen.findByText(job.job_sha256);
    await user.click(screen.getByRole("checkbox"));
    expect(screen.getByRole("checkbox")).toBeChecked();

    view.rerender(<Harness client={authority} preparationSource={changed} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/prepared job is stale/i);
    expect(screen.getByText(job.job_sha256)).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Run imported study" })).toBeDisabled();
    expect(authority.submit).not.toHaveBeenCalled();
  });

  it("keeps imported jobs independent from current editor drift", async () => {
    const authority = client();
    const view = render(<Harness client={authority} preparationSource={currentEditors()} />);
    await uploadJob("external-authority.json");
    const changed: RegionalGroundCurrentEditorSource = {
      ...currentEditors(),
      launch: parseRegionalGroundExecutionLaunch({
        ...job.launch,
        azimuth_angle_rad: 0.02,
      }),
    };

    view.rerender(<Harness client={authority} preparationSource={changed} />);

    expect(screen.queryByText(/prepared job is stale/i)).not.toBeInTheDocument();
    expect(screen.getByText(job.job_sha256)).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).toBeEnabled();
    await userEvent.setup().click(screen.getByRole("checkbox"));
    expect(screen.getByRole("button", { name: "Run imported study" })).toBeEnabled();
  });

  it("exposes one accessible import action while retaining a programmatic file input", async () => {
    render(<Harness client={client()} />);

    expect(screen.getAllByRole("button", { name: "Import execution job…" })).toHaveLength(1);
    expect(screen.queryByLabelText("Import regional-ground execution job JSON"))
      .not.toBeInTheDocument();
    expect(screen.getByTestId("regional-ground-execution-job-file-input"))
      .toHaveAttribute("hidden");
    expect(await screen.findByLabelText("Local authority capability"))
      .toHaveTextContent(/qualified_execution_profile/);
  });

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

  it("recovers retained status read-only without confirmation or submission", async () => {
    const authority = client();
    render(<Harness client={authority} />);
    await uploadJob("recovered.json");

    await userEvent.setup().click(
      screen.getByRole("button", { name: "Recover retained status" }),
    );

    await waitFor(() => expect(authority.status)
      .toHaveBeenCalledWith(job, expect.any(AbortSignal)));
    expect(authority.submit).not.toHaveBeenCalled();
    expect(authority.result).toHaveBeenCalledWith(job, expect.any(AbortSignal));
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByText(/read-only lookup/i)).toBeInTheDocument();
  });

  it("preserves the accepted job when a replacement file is invalid", async () => {
    const authority = client();
    render(<Harness client={authority} />);
    await uploadJob("accepted.json");

    fireEvent.change(screen.getByTestId("regional-ground-execution-job-file-input"), {
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
    fireEvent.change(screen.getByTestId("regional-ground-execution-job-file-input"), {
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
