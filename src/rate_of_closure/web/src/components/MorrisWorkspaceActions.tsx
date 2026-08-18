import { useRef } from "react";

import type { MorrisWorkspaceDocument } from "../model/morrisWorkspaceDocument";
import {
  MAX_MORRIS_WORKSPACE_BYTES,
  morrisWorkspaceReportToCsv,
  morrisWorkspaceToJson,
} from "../model/morrisWorkspaceDocument";
import { BUTTON_CLASS, downloadText, readFileText } from "./variationUi";

interface MorrisWorkspaceActionsProps {
  readonly workspace: MorrisWorkspaceDocument | null;
  readonly busy: boolean;
  readonly onImportText: (source: string) => void;
  readonly onImportError: (message: string) => void;
}

export function MorrisWorkspaceActions({
  workspace,
  busy,
  onImportText,
  onImportError,
}: MorrisWorkspaceActionsProps): JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);
  const reportAvailable = workspace?.completedEvidence !== null && workspace !== null;
  return <div className="flex flex-wrap gap-2">
    <button type="button" className={BUTTON_CLASS} disabled={busy || workspace === null}
      title="Download the authority base, all Morris drafts, design controls, and completed aggregate evidence."
      onClick={() => workspace !== null && downloadText(
        "rate_of_closure_morris_workspace.json",
        morrisWorkspaceToJson(workspace),
        "application/json",
      )}>Export Workspace JSON</button>
    <button type="button"
      className={`${BUTTON_CLASS} focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400`}
      title="Atomically replace this Morris setup with a strict v1 workspace file, aborting active work."
      onClick={() => inputRef.current?.click()}>
      Import Workspace JSON
    </button>
    <input ref={inputRef} aria-label="Import Morris workspace JSON" type="file" accept="application/json"
        className="hidden" onChange={(event) => {
          const file = event.target.files?.[0];
          if (file && file.size > MAX_MORRIS_WORKSPACE_BYTES) {
            onImportError(`Morris workspace exceeds the ${MAX_MORRIS_WORKSPACE_BYTES}-byte file limit`);
          } else if (file) {
            void readFileText(file).then(onImportText).catch((error: unknown) => (
              onImportError(error instanceof Error ? error.message : "File read failed")
            ));
          }
          event.target.value = "";
        }} />
    <button type="button" className={BUTTON_CLASS} disabled={!reportAvailable}
      title="Download completed aggregate Morris estimates with denominator and provenance fields."
      onClick={() => reportAvailable && workspace !== null && downloadText(
        "rate_of_closure_morris_report.csv",
        morrisWorkspaceReportToCsv(workspace),
        "text/csv",
      )}>Export Aggregate CSV</button>
  </div>;
}
