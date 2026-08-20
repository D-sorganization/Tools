import type { RefObject } from "react";

import { FIELD_GUIDANCE } from "../model/units";
import type { ViewMode } from "./ClubCanvas";

interface Props {
  readonly playing: boolean;
  readonly speed: number;
  readonly mode: ViewMode;
  readonly modes: readonly ViewMode[];
  readonly showCg: boolean;
  readonly hasMesh: boolean;
  readonly meshError: string | null;
  readonly fileInputRef: RefObject<HTMLInputElement>;
  readonly onPlayingChange: (playing: boolean) => void;
  readonly onSpeedChange: (speed: number) => void;
  readonly onModeChange: (mode: ViewMode) => void;
  readonly onShowCgChange: (show: boolean) => void;
  readonly onStlChosen: (file: File | undefined) => void;
  readonly onProceduralHead: () => void;
}

/** Accessible playback and head-source controls, isolated from rendering. */
export function ClubCanvasPlaybackControls(props: Props) {
  return (
    <div
      aria-label="Playback controls"
      className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/60 px-4 py-2.5 text-sm shadow-lg shadow-black/20 backdrop-blur"
    >
      <button
        type="button"
        onClick={() => props.onPlayingChange(!props.playing)}
        title="Play or pause the impact animation"
        className="w-16 rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >
        {props.playing ? "Pause" : "Play"}
      </button>
      <label className="flex items-center gap-2">
        <span className="text-slate-400">Playback Speed</span>
        <input type="range" min={0.1} max={3} step={0.1} value={props.speed}
          onChange={(event) => props.onSpeedChange(Number(event.target.value))}
          aria-label="Playback speed multiplier" />
        <span className="w-8 text-slate-300">{props.speed.toFixed(1)}x</span>
      </label>
      <label className="flex items-center gap-2">
        <span className="text-slate-400">Display</span>
        <select value={props.mode}
          title="Display mode: head fixed in place or moving through space"
          onChange={(event) => props.onModeChange(event.target.value as ViewMode)}
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none"
        >
          {props.modes.map((mode) => <option key={mode} value={mode}>{mode}</option>)}
        </select>
      </label>
      <input ref={props.fileInputRef} type="file" accept=".stl" className="hidden"
        aria-hidden="true" tabIndex={-1}
        onChange={(event) => {
          props.onStlChosen(event.target.files?.[0]);
          event.target.value = "";
        }} />
      <button type="button" onClick={() => props.fileInputRef.current?.click()}
        title="Render a user-supplied STL clubhead mesh in place of the procedural wireframe (read locally, never uploaded)."
        className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >Load Clubhead STL…</button>
      <label title={FIELD_GUIDANCE.showCgMarker}
        className="flex items-center gap-2 text-slate-300">
        <input type="checkbox" checked={props.showCg}
          onChange={(event) => props.onShowCgChange(event.target.checked)}
          aria-label="Show CG" />
        Show CG
      </label>
      <button type="button" disabled={!props.hasMesh} onClick={props.onProceduralHead}
        title="Return to the default wireframe head."
        className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors enabled:hover:border-sky-400 disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >Procedural Head</button>
      {props.meshError && <span role="alert" className="text-xs text-rose-400">
        STL load failed: {props.meshError}
      </span>}
    </div>
  );
}
