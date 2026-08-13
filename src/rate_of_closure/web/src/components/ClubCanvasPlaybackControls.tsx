import { useRef } from "react";

import { FIELD_GUIDANCE } from "../model/units";

export const VIEW_MODES = [
  "Head Fixed in Place",
  "Head Moving Through Space",
] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

interface PlaybackControlProps {
  playing: boolean;
  speed: number;
  mode: ViewMode;
  showCg: boolean;
  meshLoaded: boolean;
  meshError: string | null;
  onPlayingChange: (playing: boolean) => void;
  onSpeedChange: (speed: number) => void;
  onModeChange: (mode: ViewMode) => void;
  onShowCgChange: (showCg: boolean) => void;
  onStlChosen: (file: File | undefined) => void;
  onProceduralHead: () => void;
}

/** Accessible playback and clubhead-source controls for one viewport. */
export function ClubCanvasPlaybackControls({
  playing,
  speed,
  mode,
  showCg,
  meshLoaded,
  meshError,
  onPlayingChange,
  onSpeedChange,
  onModeChange,
  onShowCgChange,
  onStlChosen,
  onProceduralHead,
}: PlaybackControlProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <div
      aria-label="Playback controls"
      className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/60 px-4 py-2.5 text-sm shadow-lg shadow-black/20 backdrop-blur"
    >
      <button
        type="button"
        onClick={() => onPlayingChange(!playing)}
        title="Play or pause the impact animation"
        className="w-16 rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >
        {playing ? "Pause" : "Play"}
      </button>
      <label className="flex items-center gap-2">
        <span className="text-slate-400">Playback Speed</span>
        <input
          type="range"
          min={0.1}
          max={3}
          step={0.1}
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
          aria-label="Playback speed multiplier"
        />
        <span className="w-8 text-slate-300">{speed.toFixed(1)}x</span>
      </label>
      <label className="flex items-center gap-2">
        <span className="text-slate-400">Display</span>
        <select
          value={mode}
          aria-label="Clubhead display mode"
          title="Display mode: head fixed in place or moving through space"
          onChange={(event) => onModeChange(event.target.value as ViewMode)}
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none"
        >
          {VIEW_MODES.map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </select>
      </label>
      <input
        ref={fileInputRef}
        type="file"
        accept=".stl"
        className="hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(event) => {
          onStlChosen(event.target.files?.[0]);
          event.target.value = "";
        }}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        title="Render a user-supplied STL clubhead mesh in place of the procedural wireframe (read locally, never uploaded)."
        className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >
        Load Clubhead STL…
      </button>
      <label title={FIELD_GUIDANCE.showCgMarker} className="flex items-center gap-2 text-slate-300">
        <input
          type="checkbox"
          checked={showCg}
          onChange={(event) => onShowCgChange(event.target.checked)}
          aria-label="Show CG"
        />
        Show CG
      </label>
      <button
        type="button"
        disabled={!meshLoaded}
        onClick={onProceduralHead}
        title="Return to the default wireframe head."
        className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors enabled:hover:border-sky-400 disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
      >
        Procedural Head
      </button>
      {meshError && (
        <span role="alert" className="text-xs text-rose-400">
          STL load failed: {meshError}
        </span>
      )}
    </div>
  );
}
