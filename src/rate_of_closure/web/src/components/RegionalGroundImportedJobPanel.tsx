/** Visible execution workflow for one App-owned exact authority job. */

import { type ChangeEvent, useRef, useState } from "react";

import type { RegionalGroundExecutionWorkspace } from "../hooks/useRegionalGroundExecutionWorkspace";
import { RegionalGroundAuthorityRequestError } from "../model/regionalGroundAuthorityClient";
import { downloadRegionalGroundExecutionJob } from "../model/regionalGroundExecutionJobFiles";
import { downloadRegionalGroundExecutionResult } from "../model/regionalGroundExecutionResultFiles";

interface RegionalGroundImportedJobPanelProps {
  readonly workspace: RegionalGroundExecutionWorkspace;
  readonly saveJob?: typeof downloadRegionalGroundExecutionJob;
  readonly saveResult?: typeof downloadRegionalGroundExecutionResult;
}

const INITIAL_MESSAGE = "Import an exact execution job to begin.";

const safeRequestMessage = (error: unknown): string => error instanceof RegionalGroundAuthorityRequestError
  ? `Local authority request failed (${error.code}). Reconcile before replacing or rerunning this job.`
  : "Local authority request failed. Reconcile before replacing or rerunning this job.";

export function RegionalGroundImportedJobPanel(props: RegionalGroundImportedJobPanelProps) {
  const input = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState(INITIAL_MESSAGE);
  const [error, setError] = useState<string | null>(null);
  const { workspace } = props;
  const { acceptedJob: job, authority, confirmed, execution, sourceName } = workspace;
  const active = execution.controls.statusEnabled;

  const importJob = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined || active) return;
    try {
      await workspace.importFile(file);
      setError(null);
      setMessage(`Loaded ${file.name}. Review and confirm the exact imported authority before running.`);
    } catch {
      setError(`Could not import ${file.name}. The prior accepted job was preserved.`);
    }
  };

  const run = async () => {
    setError(null);
    try {
      await workspace.run();
    } catch (reason) {
      setError(safeRequestMessage(reason));
    }
  };
  const cancel = async () => {
    setError(null);
    try { await execution.cancel(); } catch (reason) { setError(safeRequestMessage(reason)); }
  };
  const reconcile = async () => {
    setError(null);
    try { await execution.reconcile(); } catch (reason) { setError(safeRequestMessage(reason)); }
  };
  const saveJob = () => {
    if (job === null) return;
    try {
      (props.saveJob ?? downloadRegionalGroundExecutionJob)(job);
      setError(null);
      setMessage("Downloaded the canonical imported job JSON.");
    } catch { setError("Could not download the canonical job. The accepted job remains loaded."); }
  };
  const saveResult = () => {
    if (execution.result === null) return;
    try {
      (props.saveResult ?? downloadRegionalGroundExecutionResult)(execution.result);
      setError(null);
      setMessage("Downloaded the canonical job-bound result JSON.");
    } catch { setError("Could not download the canonical result. The validated result remains loaded."); }
  };
  const clear = () => {
    try {
      workspace.clear();
      setError(null);
      setMessage(INITIAL_MESSAGE);
    } catch (reason) { setError(safeRequestMessage(reason)); }
  };

  const canRun = job !== null && confirmed && execution.controls.submitEnabled;
  const visibleMessage = job !== null && message === INITIAL_MESSAGE
    ? `Loaded ${sourceName}. Review and confirm the exact imported authority before running.`
    : message;
  return (
    <section aria-labelledby="regional-ground-imported-job-heading"
      className="rounded-xl border border-emerald-500/30 bg-emerald-950/10 p-4">
      <h3 id="regional-ground-imported-job-heading" className="font-semibold text-slate-100">
        Imported study execution
      </h3>
      <p className="mt-1 text-sm text-slate-300">
        Run only a complete Python-qualified regional-ground-execution-job/v1 file.
        Exact validated job bytes go to the isolated local Python authority; this page
        neither constructs jobs from editor state nor executes physics in the browser.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <input ref={input} type="file" accept=".json,application/json" className="sr-only"
          aria-label="Import regional-ground execution job JSON" disabled={active}
          onChange={(event) => { void importJob(event); }} />
        <button type="button" disabled={active} onClick={() => input.current?.click()}
          className="rounded-md border border-sky-500/60 px-3 py-2 text-sm text-sky-200 disabled:opacity-40">
          Import execution job…
        </button>
        <button type="button" disabled={job === null || active} onClick={clear}
          className="rounded-md border border-slate-600 px-3 py-2 text-sm disabled:opacity-40">
          Clear imported job
        </button>
        <button type="button" disabled={job === null || active} onClick={saveJob}
          aria-label="Download canonical regional-ground execution job"
          className="rounded-md border border-sky-500/70 px-3 py-2 text-sm text-sky-200 disabled:opacity-40">
          Save job JSON…
        </button>
      </div>
      <p className="mt-3 text-xs text-slate-400" aria-label="Local authority capability" aria-live="polite">
        Authority: {authority.checking ? "checking" : authority.capability.reason_code} — {authority.capability.detail}
      </p>
      {job !== null && <>
        <dl aria-label="Imported execution job summary"
          className="mt-3 grid gap-2 rounded border border-slate-700/80 p-3 text-xs sm:grid-cols-2 xl:grid-cols-4">
          <div><dt className="text-slate-500">File</dt><dd>{sourceName}</dd></div>
          <div><dt className="text-slate-500">Job ID</dt><dd>{job.job_id}</dd></div>
          <div><dt className="text-slate-500">Schema</dt><dd>{job.schema_version}</dd></div>
          <div><dt className="text-slate-500">Model</dt><dd>{job.flight.model_id} {job.flight.model_version}</dd></div>
          <div><dt className="text-slate-500">Trials</dt><dd>{job.execution_options.max_trials}</dd></div>
          <div><dt className="text-slate-500">Producer</dt><dd>{job.provenance.producer} / {job.provenance.source_revision}</dd></div>
          <div className="sm:col-span-2"><dt className="text-slate-500">Job SHA-256</dt><dd className="break-all">{job.job_sha256}</dd></div>
          <div className="sm:col-span-2"><dt className="text-slate-500">Input SHA-256</dt><dd className="break-all">{job.input_sha256}</dd></div>
          <div className="sm:col-span-2"><dt className="text-slate-500">Qualified-plan SHA-256</dt><dd className="break-all">{job.qualified_plan_sha256}</dd></div>
        </dl>
        <label className="mt-3 flex items-start gap-2 text-sm text-slate-200">
          <input type="checkbox" checked={confirmed} disabled={active || execution.job !== null}
            onChange={(event) => workspace.setConfirmed(event.currentTarget.checked)} />
          I reviewed the imported job identity, model, trial count, provenance, and digests
          and want the local Python authority to execute exactly this job.
        </label>
      </>}
      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" disabled={!canRun} onClick={() => { void run(); }}
          className="rounded-md border border-emerald-500/70 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-200 disabled:opacity-40">
          Run imported study
        </button>
        <button type="button" disabled={!execution.controls.cancelEnabled} onClick={() => { void cancel(); }}
          className="rounded-md border border-amber-500/70 px-3 py-2 text-sm text-amber-200 disabled:opacity-40">
          Cancel study
        </button>
        <button type="button" disabled={!execution.controls.statusEnabled || execution.phase !== "request_failed"}
          onClick={() => { void reconcile(); }}
          className="rounded-md border border-violet-500/70 px-3 py-2 text-sm text-violet-200 disabled:opacity-40">
          Reconcile status
        </button>
        <button type="button" disabled={!execution.controls.resultEnabled} onClick={saveResult}
          aria-label="Download canonical regional-ground study result"
          className="rounded-md border border-sky-500/70 px-3 py-2 text-sm text-sky-200 disabled:opacity-40">
          Save result JSON…
        </button>
      </div>
      {execution.progress !== null && <progress className="mt-3 w-full"
        aria-label="Regional-ground study progress" value={execution.progress.completed}
        max={execution.progress.total} />}
      <p role="status" aria-live="polite" aria-label="Imported study execution status" className="mt-2 text-sm text-slate-300">
        {execution.phase.replace(/_/g, " ")} — {execution.progress === null
          ? visibleMessage : `${execution.progress.completed} / ${execution.progress.total} accepted trials`}
      </p>
      {execution.result !== null && <p className="mt-2 text-xs text-emerald-200">
        Validated result {execution.result.dataset.result_id}; {execution.result.dataset.rows.length} rows;
        dataset {execution.result.dataset_sha256}.
      </p>}
      {execution.failure !== null && <p role="alert" className="mt-2 text-sm text-rose-200">
        Execution failed ({execution.failure.code}, {execution.failure.stage}). No partial result was published.
      </p>}
      {error !== null && <p role="alert" className="mt-2 text-sm text-rose-200">{error}</p>}
    </section>
  );
}
