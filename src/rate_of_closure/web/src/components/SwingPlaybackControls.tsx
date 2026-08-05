import { type Dispatch, type SetStateAction } from "react";

import { type SimulationRunTs } from "../model/simulation";

const RATE_PRESETS = [
  ["0.1×", 0.1], ["0.25×", 0.25], ["0.5×", 0.5],
  ["1× real-time", 1], ["2×", 2],
] as const;

interface Props {
  run: SimulationRunTs | null;
  playing: boolean;
  setPlaying: Dispatch<SetStateAction<boolean>>;
  time: number;
  setTime: Dispatch<SetStateAction<number>>;
  loop: boolean;
  setLoop: (value: boolean) => void;
  rate: number;
  setRate: (value: number) => void;
  toggles: Array<[string, boolean, (value: boolean) => void, string, string]>;
}

const buttonClass =
  "rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-300 " +
  "hover:border-slate-500 disabled:opacity-40";

export function SwingPlaybackControls(props: Props) {
  const { run, playing, setPlaying, time, setTime, loop, setLoop, rate, setRate, toggles } = props;
  return (
    <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
      <button
        type="button"
        onClick={() => setPlaying((current) => !current && run !== null)}
        disabled={!run}
        title="Play or pause the swing playback"
        className={buttonClass}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        onClick={() => setTime((current) => Math.max(0, current - 0.001))}
        disabled={!run}
        title="Step the playback one millisecond backward"
        className={buttonClass}
      >−1 frame</button>
      <button
        type="button"
        onClick={() => setTime((current) => Math.min(run?.totalDurationS ?? 0, current + 0.001))}
        disabled={!run}
        title="Step the playback one millisecond forward"
        className={buttonClass}
      >+1 frame</button>
      <input
        type="range"
        min={0}
        max={run?.totalDurationS ?? 1}
        step={0.001}
        value={time}
        onChange={(event) => setTime(Number(event.target.value))}
        disabled={!run}
        className="min-w-32 flex-1"
        aria-label="Playback timeline"
      />
      <span className="tabular-nums text-slate-400">{time.toFixed(3)} s</span>
      <label className="flex items-center gap-1 text-slate-300">
        <input
          type="checkbox"
          checked={loop}
          title="Restart the playback automatically when it reaches the end"
          onChange={(event) => setLoop(event.target.checked)}
        />
        Loop
      </label>
      <select
        value={rate}
        onChange={(event) => setRate(Number(event.target.value))}
        className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100"
        aria-label="Playback rate"
      >
        {RATE_PRESETS.map(([label, value]) => <option key={label} value={value}>{label}</option>)}
      </select>
      {toggles.map(([label, checked, setChecked, guidance, color]) => (
        <label key={label} className={`flex items-center gap-1 ${color}`} title={guidance}>
          <input
            type="checkbox"
            checked={checked}
            onChange={(event) => setChecked(event.target.checked)}
          />
          {label}
        </label>
      ))}
    </div>
  );
}
