/** Contextual File-menu controls for the App-owned execution workspace. */

import { useRef, useState, type ChangeEvent } from "react";

import type { RegionalGroundExecutionWorkspace } from "../hooks/useRegionalGroundExecutionWorkspace";
import { APP_COMMAND_ID, type AppCommandId } from "../model/appCommands";
import { downloadRegionalGroundExecutionJob } from "../model/regionalGroundExecutionJobFiles";
import {
  downloadRegionalGroundExecutionResult,
  downloadRegionalGroundExecutionRowsCsv,
} from "../model/regionalGroundExecutionResultFiles";

interface Props {
  readonly workspace: RegionalGroundExecutionWorkspace;
  readonly onCommand: (command: AppCommandId) => void;
  readonly commandClassName: string;
}

export function RegionalGroundExecutionFileCommands(props: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const active = props.workspace.execution.controls.statusEnabled;
  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined) return;
    try {
      await props.workspace.importFile(file);
      setError(null);
      setStatus(`Imported ${file.name}. No browser physics executed.`);
    } catch {
      setError("Execution-job import failed; the prior accepted job and result were preserved.");
    }
  };
  const runDownload = (command: AppCommandId, action: () => void, complete: string) => {
    props.onCommand(command);
    try {
      action();
      setError(null);
      setStatus(complete);
    } catch {
      setError("Download failed; the accepted validated evidence remains in memory.");
    }
  };
  const job = props.workspace.acceptedJob;
  const result = props.workspace.execution.result;
  return (
    <div className="mb-2 border-b border-slate-800 pb-2">
      <p className="mb-2 text-xs leading-relaxed text-slate-400">
        Exact imported authority job and validated result. Browser downloads remain destination-owned.
      </p>
      <input ref={input} type="file" accept=".json,application/json" className="sr-only"
        disabled={active} aria-label="Open Regional-Ground Execution Job file"
        onChange={(event) => { void importFile(event); }} />
      <button type="button" disabled={active} className={props.commandClassName}
        data-command-id={APP_COMMAND_ID.fileOpenRegionalGroundExecutionJob}
        onClick={() => {
          props.onCommand(APP_COMMAND_ID.fileOpenRegionalGroundExecutionJob);
          input.current?.click();
        }}>
        Open Regional-Ground Execution Job
      </button>
      <button type="button" disabled={job === null || active} className={props.commandClassName}
        data-command-id={APP_COMMAND_ID.fileSaveRegionalGroundExecutionJobAs}
        onClick={() => job !== null && runDownload(
          APP_COMMAND_ID.fileSaveRegionalGroundExecutionJobAs,
          () => downloadRegionalGroundExecutionJob(job),
          "Canonical job download prepared.",
        )}>
        Save Regional-Ground Execution Job As
      </button>
      <button type="button" disabled={result === null} className={props.commandClassName}
        data-command-id={APP_COMMAND_ID.fileSaveRegionalGroundExecutionResultAs}
        onClick={() => result !== null && runDownload(
          APP_COMMAND_ID.fileSaveRegionalGroundExecutionResultAs,
          () => downloadRegionalGroundExecutionResult(result),
          "Canonical result download prepared.",
        )}>
        Save Regional-Ground Execution Result As
      </button>
      <button type="button" disabled={result === null} className={props.commandClassName}
        data-command-id={APP_COMMAND_ID.fileExportRegionalGroundExecutionRowsCsv}
        onClick={() => result !== null && runDownload(
          APP_COMMAND_ID.fileExportRegionalGroundExecutionRowsCsv,
          () => downloadRegionalGroundExecutionRowsCsv(result),
          "Lossless scalar-row CSV download prepared.",
        )}>
        Export Regional-Ground Execution Rows CSV
      </button>
      {status !== null && <p role="status" className="mt-2 text-xs text-slate-400">{status}</p>}
      {error !== null && <p role="alert" className="mt-2 text-xs text-rose-200">{error}</p>}
    </div>
  );
}
