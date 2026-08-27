import { type Dispatch, type SetStateAction } from "react";

import { type SimulationRunTs } from "../model/simulation";

export interface SwingPlaybackControlsProps {
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
  "hover:border-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:opacity-40";

export function SwingPlaybackControls(props: SwingPlaybackControlsProps) {
  const { run, playing, setPlaying, time, setTime, loop, setLoop, rate, setRate, toggles } = props;
  const inspectionLabel = run?.impactOutcome.status === "miss"
    ? "Closest Approach"
    : "Impact";
  const inspectionTime = run === null
    ? 0
    : (run.impactTimeS ?? run.impactOutcome.candidateTimeS);
  const togglePlayback = (): void => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (!run) return;
    if (time >= run.totalDurationS - Number.EPSILON) setTime(0);
    setPlaying(true);
  };
  return (
    <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
      <button
        type="button"
        onClick={togglePlayback}
        disabled={!run}
        title="Play or pause the swing playback"
        className={buttonClass}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        onClick={() => {
          setPlaying(false);
          setTime(0);
        }}
        disabled={!run}
        title="Stop playback and return to the first frame"
        className={buttonClass}
      >Restart</button>
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
      <button
        type="button"
        onClick={() => {
          setPlaying(false);
          setTime(inspectionTime);
        }}
        disabled={!run}
        title={run?.impactOutcome.status === "miss"
          ? "Jump to the sampled closest approach; no impact occurred"
          : "Jump to the exact impact event"}
        data-event-time={inspectionTime}
        className={buttonClass}
      >Jump to {inspectionLabel}</button>
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
      <label className="flex min-w-44 items-center gap-2 text-slate-300">
        Speed
        <input
        type="range"
        min={0.05}
        max={4}
        step={0.05}
        value={rate}
        onChange={(event) => setRate(Number(event.target.value))}
        className="min-w-24 flex-1 accent-sky-400"
        aria-label="Playback speed"
        title="Adjust playback continuously from 0.05× to 4× real-time"
      />
        <output className="w-12 tabular-nums text-sky-300">{rate.toFixed(2)}×</output>
      </label>
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
