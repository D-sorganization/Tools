/**
 * Subject-neutral React transport controls for 3D shot playback (#4800 P8).
 *
 * The single React implementation of the playback transport chrome —
 * play/pause, restart, event jumps, scrub, speed, and the sole animation
 * frame — mirroring the Qt
 * `ui/pyqt6/playback_transport_controls.PlaybackTransportControls`. Every
 * semantic decision (scrub quantization, the canonical speed set, the
 * wall-clock advance) delegates to `../model/playbackTransport`, whose
 * Python twin drives the Qt surfaces, so both runtimes share one timeline
 * model.
 *
 * The bar carries no ball-flight vocabulary: `FlightPlayback3D` binds it
 * with "Ball Flight" wording and Launch/Apex/Landing events, and the
 * putting vertical binds it with "Putt" wording and Strike/Finish events.
 * Time is a controlled prop, so the owning surface stays the authority on
 * which frame is drawn; an externally set time pauses playback exactly as
 * the Qt widget's `jump_to_time` does.
 *
 * Camera seam (#4571): camera state belongs to `model/cameraCommands.ts`
 * and the canvas viewports; this component owns only the timeline.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  DEFAULT_SPEED,
  PLAYBACK_SPEEDS,
  SCRUB_STEPS,
  advancePlayback,
  scrubValue,
  timeAtScrub,
} from "../model/playbackTransport";

/** One named jump target on the recorded timeline. */
export interface PlaybackEvent {
  readonly label: string;
  readonly timeS: number;
}

interface Props {
  /** Title-case subject naming the controls, e.g. "Ball Flight". */
  readonly subjectLabel: string;
  /** Sentence-case subject phrase, e.g. "ball flight". */
  readonly subjectPhrase: string;
  readonly timeS: number;
  readonly durationS: number;
  /** Jump targets; the first renders before Play, the rest after Restart. */
  readonly events: readonly PlaybackEvent[];
  readonly scrubTitle: string;
  readonly onTimeChange: (timeS: number) => void;
}

const BUTTON_CLASS =
  "rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40";

function sentenceCase(phrase: string): string {
  return phrase.charAt(0).toUpperCase() + phrase.slice(1);
}

export function PlaybackTransportBar({
  subjectLabel,
  subjectPhrase,
  timeS,
  durationS,
  events,
  scrubTitle,
  onTimeChange,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(DEFAULT_SPEED);
  const timeRef = useRef(timeS);
  const emittedRef = useRef(timeS);
  const changeRef = useRef(onTimeChange);
  changeRef.current = onTimeChange;

  // A time the owner set itself pauses playback; a time this bar emitted
  // during animation does not.
  useEffect(() => {
    timeRef.current = timeS;
    if (timeS !== emittedRef.current) {
      emittedRef.current = timeS;
      setPlaying(false);
    }
  }, [timeS]);

  useEffect(() => setPlaying(false), [durationS]);

  const emit = useCallback((next: number) => {
    emittedRef.current = next;
    timeRef.current = next;
    changeRef.current(next);
  }, []);

  useEffect(() => {
    if (!playing || durationS <= 0) return;
    let animationId = 0;
    let previous = performance.now();
    const animate = (now: number) => {
      const elapsed = Math.max(0, now - previous) / 1000;
      previous = now;
      const step = advancePlayback(timeRef.current, elapsed, speed, durationS);
      emit(step.timeS);
      if (step.finished) setPlaying(false);
      else animationId = window.requestAnimationFrame(animate);
    };
    animationId = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(animationId);
  }, [playing, durationS, speed, emit]);

  const disabled = durationS <= 0;
  const jump = (targetTime: number) => {
    setPlaying(false);
    emit(targetTime);
  };
  const toggle = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (timeS >= durationS) emit(0);
    setPlaying(!disabled);
  };
  const eventButton = (event: PlaybackEvent) => (
    <button
      key={event.label}
      type="button"
      disabled={disabled}
      onClick={() => jump(event.timeS)}
      aria-label={`Jump to ${event.label}`}
      className={BUTTON_CLASS}
    >
      {event.label}
    </button>
  );

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {events.length > 0 ? eventButton(events[0]) : null}
      <button
        type="button"
        disabled={disabled}
        onClick={toggle}
        aria-label={`${playing ? "Pause" : "Play"} ${subjectLabel}`}
        className="rounded border border-sky-500/60 bg-sky-500/10 px-3 py-1 font-semibold text-sky-200 disabled:opacity-40"
      >
        {playing ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          emit(0);
          setPlaying(!disabled);
        }}
        aria-label={`Restart ${subjectLabel}`}
        className={BUTTON_CLASS}
      >
        Restart
      </button>
      {events.slice(1).map(eventButton)}
      <input
        type="range"
        min={0}
        max={SCRUB_STEPS}
        step={1}
        value={scrubValue(timeS, durationS)}
        disabled={disabled}
        onChange={(event) =>
          jump(timeAtScrub(Number(event.target.value), durationS))
        }
        aria-label={`${subjectLabel} Time`}
        title={scrubTitle}
        className="min-w-36 flex-1"
      />
      <label className="flex items-center gap-1 text-slate-300">
        Speed
        <select
          value={speed}
          onChange={(event) => setSpeed(Number(event.target.value))}
          aria-label="Playback Speed"
          className="w-[4.5rem] rounded border border-slate-700 bg-slate-900 px-1 py-1"
        >
          {PLAYBACK_SPEEDS.map((option) => (
            <option key={option} value={option}>
              {option}×
            </option>
          ))}
        </select>
      </label>
      <output
        aria-label={`${sentenceCase(subjectPhrase)} playback position`}
        className="min-w-20 text-right tabular-nums text-slate-300"
      >
        {timeS.toFixed(2)} / {durationS.toFixed(2)} s
      </output>
    </div>
  );
}
