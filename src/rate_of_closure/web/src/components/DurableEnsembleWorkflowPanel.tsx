import { useEffect, useMemo, useRef, useState } from "react";

import {
  serializeDurableEnsembleRequest,
  type DurableEnsembleCapability,
  type DurableEnsembleJob,
} from "../model/durableEnsembleAuthorityContract";
import {
  createDurableEnsembleRunner,
  type DurableEnsembleRunController,
  type DurableEnsembleRunner,
} from "../model/durableEnsembleWorkerClient";
import type { MorrisAuthorityBase } from "../model/morrisAuthorityRequest";
import type { VariationPlanTs } from "../model/variation";
import { BUTTON_CLASS, INPUT_CLASS, PANEL_CLASS } from "./variationUi";

interface Props {
  readonly plan: VariationPlanTs;
  readonly base: MorrisAuthorityBase;
  readonly runner?: DurableEnsembleRunner;
}

const stableId = (prefix: string): string => {
  const suffix = globalThis.crypto?.randomUUID?.() ?? String(Date.now());
  return `${prefix}-${suffix}`;
};

export function DurableEnsembleWorkflowPanel({ plan, base, runner: injected }: Props) {
  const owned = useMemo(() => injected ?? createDurableEnsembleRunner(), [injected]);
  const [capability, setCapability] = useState<DurableEnsembleCapability | null>(null);
  const [archiveId, setArchiveId] = useState("proximal-distal-ensemble");
  const [chunkSize, setChunkSize] = useState(256);
  const [job, setJob] = useState<DurableEnsembleJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<DurableEnsembleRunController | null>(null);
  useEffect(() => {
    let live = true;
    void owned.capability().then((value) => { if (live) setCapability(value); })
      .catch((reason: unknown) => { if (live) setError(reason instanceof Error ? reason.message : "Authority unavailable."); });
    return () => { live = false; controller.current?.cancel(); if (injected === undefined) owned.close(); };
  }, [injected, owned]);
  const busy = job?.status === "queued" || job?.status === "running";
  const run = () => {
    try {
      setError(null); setJob(null);
      const request = serializeDurableEnsembleRequest({
        requestId: stableId("durable"), archiveId, plan, base, chunkSize,
      });
      controller.current = owned.run(request, setJob);
      void controller.current.promise.catch((reason: unknown) => setError(
        reason instanceof Error ? reason.message : "Durable ensemble run failed.",
      ));
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Durable ensemble request is invalid.");
    }
  };
  const evidence = job?.evidence;
  const status = error ?? (capability === null ? "Checking local authority capability…"
    : capability.available ? "Local durable ensemble authority is available."
      : "Durable ensemble analysis is unavailable in this static client.");
  return <section aria-label="Durable ensemble analysis" className={`${PANEL_CLASS} space-y-4`}>
    <div><h2 className="text-xl font-semibold">Durable Ensemble Analysis</h2>
      <p className="mt-1 max-w-4xl text-sm text-slate-400">Run or resume a bounded, checkpointed ensemble in the local Python authority. The browser Worker transports lifecycle records only; it performs no numerical analysis.</p></div>
    <p role="status" className={`rounded border p-3 text-sm ${error || capability?.available !== true ? "border-amber-500/40 text-amber-100" : "border-emerald-500/30 text-emerald-100"}`}>{status}</p>
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs text-slate-300">Archive Identifier<input className={`${INPUT_CLASS} mt-1`} value={archiveId} disabled={busy} onChange={(event) => setArchiveId(event.target.value)} /></label>
      <label className="text-xs text-slate-300">Chunk Size<input className={`${INPUT_CLASS} mt-1`} type="number" min={1} max={4096} value={chunkSize} disabled={busy} onChange={(event) => setChunkSize(Number(event.target.value))} /></label>
    </div>
    {job && <div aria-label="Durable ensemble progress"><progress className="h-2 w-full" max={job.totalTrials} value={job.completedTrials} /><p className="text-xs text-slate-400">{job.completedTrials} of {job.totalTrials} verified trials; status: {job.status}.</p></div>}
    <div className="flex gap-2"><button type="button" className={BUTTON_CLASS} disabled={capability?.available !== true || busy} onClick={run}>Run or Resume Ensemble</button><button type="button" className={BUTTON_CLASS} disabled={!busy} onClick={() => controller.current?.cancel()}>Cancel Ensemble</button></div>
    {evidence && <div className="overflow-x-auto"><h3 className="font-semibold">Verified Prefix Moments</h3><table className="mt-2 min-w-full text-xs"><thead><tr><th>Output</th><th>Count</th><th>Mean</th><th>Sample SD</th></tr></thead><tbody>{evidence.outputMoments.map((moment) => <tr key={moment.name}><td>{moment.name} ({moment.unit})</td><td>{moment.availableCount}</td><td>{moment.mean ?? "Unavailable"}</td><td>{moment.sampleStd ?? "Unavailable"}</td></tr>)}</tbody></table></div>}
    <p className="text-xs text-slate-400">Model-scenario output is not human evidence or a coaching recommendation. In-progress evidence covers only the verified contiguous prefix; row-level trials, quantiles, and correlations are not retained.</p>
  </section>;
}

