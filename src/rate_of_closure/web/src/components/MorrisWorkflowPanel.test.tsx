import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "../model/__fixtures__/morris_ui_parity_v1.json";
import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import type { MorrisAuthorityClient } from "../model/morrisAuthorityClient";
import { parseMorrisAuthorityCapability, type MorrisAuthorityCapability } from "../model/morrisAuthorityCapability";
import { parseMorrisJobEnvelope, type MorrisJobEnvelope } from "../model/morrisAuthorityContract";
import { morrisAuthorityBaseIdentity, type MorrisAuthorityBase } from "../model/morrisAuthorityRequest";
import { defaultMorrisAuthorityBase } from "../model/morrisWorkflowDefaults";
import { MorrisWorkflowPanel } from "./MorrisWorkflowPanel";

const capability = parseMorrisAuthorityCapability({
  schema_id: "rate-of-closure/morris-authority-capability", schema_version: 1,
  available: true, api_prefix: "/api/rate-of-closure/v1",
  request_schema_id: "rate-of-closure/morris-request", job_schema_id: "rate-of-closure/morris-job",
});
const completed = parseMorrisJobEnvelope(fixture.completed_job);
const PINNED_SCENARIO = Object.freeze({ ...DEFAULT_SCENARIO, clubheadSpeedMph: 113 });
const queued: MorrisJobEnvelope = Object.freeze({
  ...completed, status: "queued", completedSamples: 0, report: null,
});

const keyedPanel = (
  client: MorrisAuthorityClient, base: MorrisAuthorityBase,
) => <MorrisWorkflowPanel key={morrisAuthorityBaseIdentity(base)}
  client={client} base={base} pollIntervalMs={1} />;

const clientWith = (overrides: Partial<MorrisAuthorityClient> = {}): MorrisAuthorityClient => {
  let requestId = completed.requestId;
  let jobId = completed.jobId;
  const contextualize = (job: MorrisJobEnvelope): MorrisJobEnvelope => Object.freeze({
    ...job, requestId, jobId,
  });
  const create = overrides.create ?? vi.fn(async () => queued);
  const status = overrides.status ?? vi.fn(async () => completed);
  const cancel = overrides.cancel ?? vi.fn(async () => Object.freeze({
    ...queued, status: "cancelled" as const, cancelRequested: true,
  }));
  return {
    capability: overrides.capability ?? vi.fn(async () => capability),
    create: vi.fn(async (request: unknown, signal?: AbortSignal) => {
      requestId = (request as { request_id: string }).request_id;
      const job = await create(request, signal);
      jobId = job.jobId;
      return contextualize(job);
    }),
    status: vi.fn(async (currentJobId: string, signal?: AbortSignal) => (
      contextualize(await status(currentJobId, signal))
    )),
    cancel: vi.fn(async (currentJobId: string, signal?: AbortSignal) => (
      contextualize(await cancel(currentJobId, signal))
    )),
  };
};

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("MorrisWorkflowPanel", () => {
  it("gates execution on capability and preserves canonical editable factor order", async () => {
    const client = clientWith();
    render(<MorrisWorkflowPanel client={client}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    expect(screen.getByRole("status")).toHaveTextContent(/checking.*authority/i);
    expect(await screen.findByRole("button", { name: "Run Morris Screening" })).toBeEnabled();
    expect(screen.getAllByRole("row").slice(1).map((row) => row.getAttribute("data-variable-key"))).toEqual([
      "swing_sim.swing.yaw_deg", "swing_sim.swing.side_tilt_deg",
      "swing_sim.swing.forward_tilt_deg", "swing_sim.swing.damping_shoulder",
      "swing_sim.swing.damping_wrist", "swing_sim.impact.delivery.impact_offset_toe_mm",
      "swing_sim.impact.delivery.impact_offset_high_mm", "swing_sim.club.head_mass_kg",
      "swing_sim.club.head_moi_kg_m2", "swing_sim.ball_setup.tee_height_m",
    ]);
    expect(screen.getByRole("textbox", { name: "Swing-Plane Yaw lower bound" })).toBeEnabled();
  });

  it("submits the strict request, polls sequentially, and presents target-local rankings", async () => {
    const user = userEvent.setup();
    const client = clientWith();
    render(<MorrisWorkflowPanel client={client}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    await screen.findByText(/authority available/i);
    const lower = screen.getByRole("textbox", { name: "Swing-Plane Yaw lower bound" });
    await user.clear(lower);
    await user.type(lower, "-5");
    await user.tab();
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await screen.findByRole("heading", { name: /clubhead x m ranking/i });
    expect(client.create).toHaveBeenCalledTimes(1);
    const request = vi.mocked(client.create).mock.calls[0][0] as {
      factors: Array<{ variable_key: string; lower: number }>; trajectories: number;
    };
    expect(request.factors.map((factor) => factor.variable_key)).toEqual([
      "swing_sim.swing.yaw_deg", "swing_sim.swing.side_tilt_deg",
      "swing_sim.swing.forward_tilt_deg", "swing_sim.swing.damping_shoulder",
      "swing_sim.swing.damping_wrist", "swing_sim.impact.delivery.impact_offset_toe_mm",
      "swing_sim.impact.delivery.impact_offset_high_mm", "swing_sim.club.head_mass_kg",
      "swing_sim.club.head_moi_kg_m2", "swing_sim.ball_setup.tee_height_m",
    ]);
    expect(request.factors[0].lower).toBe(-5);
    expect(request.trajectories).toBe(12);
    expect(client.status).toHaveBeenCalledTimes(1);
    const results = screen.getByRole("region", { name: "Morris screening results" });
    expect(within(results).getAllByRole("row")[1]).toHaveTextContent(/Clubhead Speed.*3\.0000/);
    expect(screen.getByText(/Morris sigma.*does not isolate/i)).toBeVisible();
  });

  it("cancels an active job and aborts outstanding work on unmount", async () => {
    const user = userEvent.setup();
    let releaseStatus: ((value: MorrisJobEnvelope) => void) | undefined;
    const status = vi.fn((_jobId: string, signal?: AbortSignal) => new Promise<MorrisJobEnvelope>(
      (resolve, reject) => {
        releaseStatus = resolve;
        signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      },
    ));
    const client = clientWith({ status });
    const view = render(<MorrisWorkflowPanel client={client}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await waitFor(() => expect(status).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "Cancel Morris Screening" }));
    expect(client.cancel).toHaveBeenCalledWith("parity-job", expect.any(AbortSignal));
    expect(await screen.findByRole("status")).toHaveTextContent(/cancelled/i);
    await act(async () => { releaseStatus?.(completed); });
    expect(status).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("heading", { name: /ranking/i })).not.toBeInTheDocument();
    view.unmount();
  });

  it("is honestly unavailable without an injected authority client", () => {
    render(<MorrisWorkflowPanel client={null}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} />);
    expect(screen.getByRole("status")).toHaveTextContent(/not connected.*no browser physics fallback/i);
    expect(screen.getByRole("button", { name: "Run Morris Screening" })).toBeDisabled();
  });

  it("aborts capability discovery when the workflow unmounts", () => {
    const aborted = vi.fn();
    const client = clientWith({
      capability: vi.fn((signal?: AbortSignal) => new Promise<MorrisAuthorityCapability>(() => {
        signal?.addEventListener("abort", aborted, { once: true });
      })),
    });
    const view = render(<MorrisWorkflowPanel client={client}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} />);
    view.unmount();
    expect(aborted).toHaveBeenCalledTimes(1);
  });

  it("aborts a pending status request when unmounted", async () => {
    const user = userEvent.setup();
    const aborted = vi.fn();
    const status = vi.fn((_jobId: string, signal?: AbortSignal) => new Promise<MorrisJobEnvelope>(
      (_resolve, reject) => signal?.addEventListener("abort", () => {
        aborted();
        reject(new DOMException("Aborted", "AbortError"));
      }, { once: true }),
    ));
    const view = render(<MorrisWorkflowPanel client={clientWith({ status })}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await waitFor(() => expect(status).toHaveBeenCalledTimes(1));
    view.unmount();
    expect(aborted).toHaveBeenCalledTimes(1);
  });

  it("continues polling after a nonterminal cancel response until cancelled", async () => {
    const user = userEvent.setup();
    let firstStatus = true;
    const terminal = Object.freeze({ ...queued, status: "cancelled" as const, cancelRequested: true });
    const status = vi.fn((_jobId: string, signal?: AbortSignal): Promise<MorrisJobEnvelope> => {
      if (!firstStatus) return Promise.resolve(terminal);
      firstStatus = false;
      return new Promise((_resolve, reject) => signal?.addEventListener("abort", () => (
        reject(new DOMException("Aborted", "AbortError"))
      ), { once: true }));
    });
    const cancelling = Object.freeze({ ...queued, status: "running" as const, cancelRequested: true });
    const client = clientWith({ status, cancel: vi.fn(async () => cancelling) });
    render(<MorrisWorkflowPanel client={client}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await waitFor(() => expect(status).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "Cancel Morris Screening" }));
    expect(await screen.findByRole("status")).toHaveTextContent(/study cancelled/i);
    expect(status).toHaveBeenCalledTimes(2);
  });

  it("disables Run during POST and prevents rapid duplicate submission", async () => {
    const user = userEvent.setup();
    let resolveCreate: ((job: MorrisJobEnvelope) => void) | undefined;
    const create = vi.fn(() => new Promise<MorrisJobEnvelope>((resolve) => { resolveCreate = resolve; }));
    render(<MorrisWorkflowPanel client={clientWith({ create })}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    await screen.findByText(/authority available/i);
    const run = screen.getByRole("button", { name: "Run Morris Screening" });
    await user.dblClick(run);
    expect(run).toBeDisabled();
    expect(create).toHaveBeenCalledTimes(1);
    await act(async () => { resolveCreate?.(completed); });
    expect(await screen.findByRole("heading", { name: /ranking/i })).toBeVisible();
  });

  it("rejects a create response that does not echo the submitted request identity", async () => {
    const user = userEvent.setup();
    const client = { ...clientWith(), create: vi.fn(async () => queued) };
    render(keyedPanel(client, defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)));
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    expect(await screen.findByText(/create.*request identity/i)).toBeVisible();
    expect(client.status).not.toHaveBeenCalled();
  });

  it("rejects status identity drift and never pivots polling to the returned job", async () => {
    const user = userEvent.setup();
    const baseClient = clientWith();
    const status = vi.fn(async () => {
      const request = vi.mocked(baseClient.create).mock.calls[0]?.[0] as { request_id: string };
      return Object.freeze({ ...queued, requestId: request.request_id, jobId: "pivot-job" });
    });
    const client = { ...baseClient, status };
    render(keyedPanel(client, defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)));
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    expect(await screen.findByText(/status.*job identity/i)).toBeVisible();
    expect(status).toHaveBeenCalledTimes(1);
    expect(status).toHaveBeenCalledWith("parity-job", expect.any(AbortSignal));
  });

  it("rejects a cancel response that changes the pinned request identity", async () => {
    const user = userEvent.setup();
    const status = vi.fn((_jobId: string, signal?: AbortSignal) => new Promise<MorrisJobEnvelope>(
      (_resolve, reject) => signal?.addEventListener("abort", () => (
        reject(new DOMException("Aborted", "AbortError"))
      ), { once: true }),
    ));
    const baseClient = clientWith({ status });
    const cancel = vi.fn(async () => Object.freeze({
      ...queued, status: "running" as const, requestId: "pivot-request",
    }));
    const client = { ...baseClient, cancel };
    render(keyedPanel(client, defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)));
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await waitFor(() => expect(status).toHaveBeenCalledOnce());
    await user.click(screen.getByRole("button", { name: "Cancel Morris Screening" }));
    expect(await screen.findByText(/cancel.*request identity/i)).toBeVisible();
    expect(cancel).toHaveBeenCalledWith("parity-job", expect.any(AbortSignal));
  });

  it("clears a completed report when a factor is edited", async () => {
    const user = userEvent.setup();
    const client = clientWith();
    render(keyedPanel(client, defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)));
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await screen.findByRole("heading", { name: /ranking/i });
    const lower = screen.getByRole("textbox", { name: "Swing-Plane Yaw lower bound" });
    await user.clear(lower);
    await user.type(lower, "-5");
    await user.tab();
    expect(screen.queryByRole("heading", { name: /ranking/i })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/authority available/i);
  });

  it("clears a completed report when a design control is edited", async () => {
    const user = userEvent.setup();
    const client = clientWith();
    render(keyedPanel(client, defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)));
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await screen.findByRole("heading", { name: /ranking/i });
    const trajectories = screen.getByRole("textbox", { name: "Trajectories" });
    await user.clear(trajectories);
    await user.type(trajectories, "13");
    await user.tab();
    expect(screen.queryByRole("heading", { name: /ranking/i })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/authority available/i);
  });

  it("preserves edits for an identical base and resets completed state for a changed base", async () => {
    const user = userEvent.setup();
    const client = clientWith();
    const initial = defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO);
    const changed = Object.freeze({ ...initial, impactOffsetToeMm: 4 });
    const view = render(keyedPanel(client, initial));
    await screen.findByText(/authority available/i);
    const lower = screen.getByRole("textbox", { name: "Impact Toward Toe lower bound" });
    await user.clear(lower);
    await user.type(lower, "-7");
    await user.tab();
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await screen.findByRole("heading", { name: /ranking/i });

    view.rerender(keyedPanel(client, Object.freeze({ ...initial })));
    expect(screen.getByRole("textbox", { name: "Impact Toward Toe lower bound" })).toHaveValue("-7");
    expect(screen.getByRole("heading", { name: /ranking/i })).toBeVisible();

    view.rerender(keyedPanel(client, changed));
    expect(await screen.findByText(/authority available/i)).toBeVisible();
    expect(screen.queryByRole("heading", { name: /ranking/i })).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Impact Toward Toe lower bound" })).toHaveValue("-4");
  });

  it("aborts an in-flight job when the authority base changes", async () => {
    const user = userEvent.setup();
    const aborted = vi.fn();
    const status = vi.fn((_jobId: string, signal?: AbortSignal) => new Promise<MorrisJobEnvelope>(
      (_resolve, reject) => signal?.addEventListener("abort", () => {
        aborted();
        reject(new DOMException("Aborted", "AbortError"));
      }, { once: true }),
    ));
    const client = clientWith({ status });
    const initial = defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO);
    const view = render(keyedPanel(client, initial));
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    await waitFor(() => expect(status).toHaveBeenCalledTimes(1));

    view.rerender(keyedPanel(client, Object.freeze({ ...initial, impactOffsetHighMm: 2 })));
    expect(aborted).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/authority available/i)).toBeVisible();
    expect(screen.queryByRole("heading", { name: /ranking/i })).not.toBeInTheDocument();
  });

  it("announces sanitized authority failure details without fabricating results", async () => {
    const user = userEvent.setup();
    const failed: MorrisJobEnvelope = Object.freeze({
      ...queued, status: "failed", error: Object.freeze({
        code: "evaluation_failed", message: "The authority could not evaluate this design.",
      }),
    });
    const client = clientWith({ status: vi.fn(async () => failed) });
    render(<MorrisWorkflowPanel client={client}
      base={defaultMorrisAuthorityBase(getClub("Driver 10.5°"), PINNED_SCENARIO)} pollIntervalMs={1} />);
    await screen.findByText(/authority available/i);
    await user.click(screen.getByRole("button", { name: "Run Morris Screening" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      /study failed.*could not evaluate.*evaluation_failed/i,
    );
    expect(screen.queryByRole("region", { name: "Morris screening results" })).not.toBeInTheDocument();
  });
});
