import { useRef, useState, type ChangeEvent } from "react";

import { APP_COMMAND_ID, type AppCommandId } from "../model/appCommands";
import {
  downloadRegionalGroundVariationRequest,
  readRegionalGroundVariationRequestFile,
} from "../model/regionalGroundVariationRequestFiles";
import type { RegionalGroundVariationRequestPort } from "../model/regionalGroundVariationWorkspace";

interface RegionalGroundVariationFileCommandsProps {
  readonly requestPort: RegionalGroundVariationRequestPort;
  readonly onCommand: (command: AppCommandId) => void;
  readonly commandClassName: string;
}

const errorMessage = (caught: unknown, fallback: string): string =>
  caught instanceof Error ? caught.message : fallback;

const useRegionalGroundVariationFileCommands = (
  props: RegionalGroundVariationFileCommandsProps,
) => {
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined) return;
    try {
      const request = await readRegionalGroundVariationRequestFile(file);
      props.requestPort.apply(request);
      setError(null);
      setStatus(`Imported ${file.name}. No physics executed.`);
    } catch (caught) {
      setError(errorMessage(caught, "Regional-ground variation request import failed"));
      setStatus("Import failed; the current editors and prior accepted request were preserved.");
    }
  };
  const open = () => {
    props.onCommand(APP_COMMAND_ID.fileOpenRegionalGroundVariationRequest);
    input.current?.click();
  };
  const download = () => {
    props.onCommand(APP_COMMAND_ID.fileSaveRegionalGroundVariationRequestAs);
    try {
      downloadRegionalGroundVariationRequest(props.requestPort.snapshot());
      setError(null);
      setStatus(
        "Download prepared. Your browser controls the destination and overwrite behavior.",
      );
    } catch (caught) {
      setError(errorMessage(caught, "Regional-ground variation request download failed"));
      setStatus("Download failed; no browser filesystem state was changed.");
    }
  };
  return { input, error, status, importFile, open, download };
};

/** Accessible contextual File commands for the App-owned combined request. */
export function RegionalGroundVariationFileCommands(
  props: RegionalGroundVariationFileCommandsProps,
) {
  const files = useRegionalGroundVariationFileCommands(props);
  return (
    <div className="mb-2 border-b border-slate-800 pb-2">
      <p className="mb-2 text-xs leading-relaxed text-slate-400">
        Combined seeded variation and regional-ground request. Browser downloads cannot promise
        a native path, atomic replacement, or recent-file access.
      </p>
      <input ref={files.input} type="file" accept=".json,application/json" className="sr-only"
        aria-label="Open Regional-Ground Variation Request file"
        onChange={(event) => { void files.importFile(event); }} />
      <button type="button"
        data-command-id={APP_COMMAND_ID.fileOpenRegionalGroundVariationRequest}
        onClick={files.open} className={props.commandClassName}>
        Open Regional-Ground Variation Request
      </button>
      <button type="button"
        data-command-id={APP_COMMAND_ID.fileSaveRegionalGroundVariationRequestAs}
        onClick={files.download} className={props.commandClassName}>
        Save Regional-Ground Variation Request As
      </button>
      {files.error !== null && <p role="alert" className="mt-2 text-xs text-rose-200">{files.error}</p>}
      {files.status !== null && <p role="status" aria-label="Regional-ground request status" className="mt-2 text-xs text-slate-400">{files.status}</p>}
    </div>
  );
}
