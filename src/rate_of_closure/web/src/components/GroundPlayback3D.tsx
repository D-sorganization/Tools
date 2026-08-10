/** Accessible, lifecycle-safe playback of one strict ground result. */

import { useEffect, useMemo, useRef, useState } from "react";

import type { GroundPhase } from "../model/flightGroundTypes";
import type { GroundPlaybackTimeline } from "../model/groundPlayback";
import { GroundPlaybackComparison } from "../model/groundPlaybackComparison";
import type {
  GroundPlaybackState,
  GroundPlaybackViewState,
} from "../model/groundPlaybackWorkspace";
import { normalizeGroundPlaybackYawDegrees } from "../model/groundPlaybackWorkspace";
import type { Vec3 } from "../model/simulation";
import { observeCanvas } from "./canvasDisplay";
import {
  drawGroundPlayback,
  FLIGHT_PLAYBACK_LOGICAL_SIZE,
  type PlaybackCamera,
} from "./flightPlaybackDrawing";

export interface GroundPlaybackPortableState {
  readonly playback: GroundPlaybackState;
  readonly view: GroundPlaybackViewState;
}

interface Props {
  readonly timeline: GroundPlaybackTimeline;
  readonly comparisonTimeline?: GroundPlaybackTimeline;
  readonly showComparison?: boolean;
  readonly initialState?: GroundPlaybackPortableState;
  readonly onStateChange?: (state: GroundPlaybackPortableState) => void;
}

const INITIAL_CAMERA: PlaybackCamera = {
  yawRad: -0.65,
  pitchRad: 0.38,
  zoom: 1,
};
const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;
const PHASES: readonly GroundPhase[] = ["bounce", "skid", "roll"];

const cameraFromView = (
  view: GroundPlaybackViewState | undefined,
): PlaybackCamera =>
  view
    ? {
        yawRad: (view.yawDeg * Math.PI) / 180,
        pitchRad: (view.pitchDeg * Math.PI) / 180,
        zoom: view.zoom,
      }
    : INITIAL_CAMERA;
const stableDegrees = (value: number): number => Number(value.toFixed(11));
const viewFromCamera = (camera: PlaybackCamera): GroundPlaybackViewState => ({
  yawDeg: stableDegrees(
    normalizeGroundPlaybackYawDegrees((camera.yawRad * 180) / Math.PI),
  ),
  pitchDeg: stableDegrees((camera.pitchRad * 180) / Math.PI),
  zoom: camera.zoom,
});

export function GroundPlayback3D({
  timeline,
  comparisonTimeline,
  showComparison = false,
  initialState,
  onStateChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(
    null,
  );
  const initialTime = initialState?.playback.timeS ?? timeline.startTimeS;
  const timeRef = useRef(initialTime);
  const [timeS, setTimeS] = useState(initialTime);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(initialState?.playback.loop ?? false);
  const [speed, setSpeed] = useState<number>(initialState?.playback.speed ?? 1);
  const [camera, setCamera] = useState(() =>
    cameraFromView(initialState?.view),
  );
  const comparison = useMemo(
    () =>
      comparisonTimeline
        ? new GroundPlaybackComparison(timeline, comparisonTimeline)
        : null,
    [timeline, comparisonTimeline],
  );
  const startTimeS = comparison?.startTimeS ?? timeline.startTimeS;
  const endTimeS = comparison?.endTimeS ?? timeline.endTimeS;
  const durationS = endTimeS - startTimeS;
  const pairedFrame = useMemo(
    () => comparison?.frameAt(timeS) ?? null,
    [comparison, timeS],
  );
  const frame = pairedFrame?.primary ?? timeline.frameAt(timeS);

  useEffect(() => {
    setPlaying(false);
    const restoredTime = initialState?.playback.timeS ?? timeline.startTimeS;
    timeRef.current = restoredTime;
    setTimeS(restoredTime);
    setSpeed(initialState?.playback.speed ?? 1);
    setLoop(initialState?.playback.loop ?? false);
    setCamera(cameraFromView(initialState?.view));
  }, [timeline, initialState]);
  useEffect(() => {
    timeRef.current = timeS;
  }, [timeS]);
  useEffect(() => {
    const clamped = Math.min(Math.max(timeRef.current, startTimeS), endTimeS);
    timeRef.current = clamped;
    setTimeS(clamped);
  }, [startTimeS, endTimeS]);
  useEffect(() => {
    const portableTimeS =
      comparison?.frameAt(timeS).timeS ?? timeline.frameAt(timeS).timeS;
    onStateChange?.({
      playback: { timeS: portableTimeS, speed, loop },
      view: viewFromCamera(camera),
    });
  }, [timeS, speed, loop, camera, timeline, comparison, onStateChange]);
  useEffect(() => {
    if (!playing || durationS <= 0) return;
    let animationId = 0;
    let previous = performance.now();
    const animate = (now: number) => {
      const elapsed = Math.max(0, now - previous) / 1000;
      previous = now;
      const candidate = timeRef.current + elapsed * speed;
      const next =
        loop && candidate >= endTimeS
          ? startTimeS + ((candidate - startTimeS) % durationS)
          : Math.min(endTimeS, candidate);
      timeRef.current = next;
      setTimeS(next);
      if (!loop && next >= endTimeS) setPlaying(false);
      else animationId = window.requestAnimationFrame(animate);
    };
    animationId = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(animationId);
  }, [playing, speed, loop, durationS, startTimeS, endTimeS]);
  useEffect(
    () =>
      observeCanvas(canvasRef, () => {
        if (canvasRef.current)
          drawGroundPlayback(
            canvasRef.current,
            timeline.result.trajectory,
            frame.positionM as Vec3,
            camera,
            timeline.endLabel,
            timeline.result.events,
            showComparison && pairedFrame && comparison
              ? {
                  points: comparison.comparison.result.trajectory,
                  events: comparison.comparison.result.events,
                  ballPosition: pairedFrame.comparison.positionM as Vec3,
                  endLabel: comparison.comparison.endLabel,
                }
              : undefined,
          );
      }),
    [timeline, comparison, pairedFrame, frame, camera, showComparison],
  );

  const jump = (target: number) => {
    setPlaying(false);
    timeRef.current = target;
    setTimeS(target);
  };
  const toggle = () => {
    if (playing) setPlaying(false);
    else {
      if (timeS >= endTimeS) jump(startTimeS);
      setPlaying(durationS > 0);
    }
  };
  const step = (direction: -1 | 1) =>
    jump(
      comparison?.stepTime(timeS, direction) ??
        timeline.stepTime(timeS, direction),
    );

  return (
    <section className="space-y-3" aria-label="Ground trajectory playback">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => jump(startTimeS)}
          aria-label="Jump to First Contact"
          className="rounded border border-slate-700 px-2 py-1"
        >
          First Contact
        </button>
        <button
          type="button"
          onClick={() => step(-1)}
          aria-label="Previous exact ground frame"
          className="rounded border border-slate-700 px-2 py-1"
        >
          −1 frame
        </button>
        <button
          type="button"
          onClick={toggle}
          aria-label={playing ? "Pause ground result" : "Play ground result"}
          className="rounded border border-sky-500/60 bg-sky-500/10 px-3 py-1 font-semibold text-sky-200"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          onClick={() => step(1)}
          aria-label="Next exact ground frame"
          className="rounded border border-slate-700 px-2 py-1"
        >
          +1 frame
        </button>
        {PHASES.map((phase) => {
          const target = timeline.phaseTime(phase);
          return target === null ? null : (
            <button
              key={phase}
              type="button"
              onClick={() => jump(target)}
              aria-label={`Jump to ${phase[0].toUpperCase()}${phase.slice(1)}`}
              className="rounded border border-slate-700 px-2 py-1"
            >
              {phase}
            </button>
          );
        })}
        <button
          type="button"
          onClick={() => jump(endTimeS)}
          aria-label={`Jump to ${timeline.endLabel}`}
          className="rounded border border-slate-700 px-2 py-1"
        >
          {timeline.endLabel}
        </button>
        <input
          type="range"
          min={startTimeS}
          max={endTimeS}
          step={0.001}
          value={timeS}
          onChange={(event) => jump(Number(event.target.value))}
          aria-label="Ground playback absolute time"
          className="min-w-36 flex-1"
        />
        <label className="flex items-center gap-1">
          Speed
          <select
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
            aria-label="Ground playback speed"
            className="rounded border border-slate-700 bg-slate-900 px-1 py-1"
          >
            {SPEEDS.map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={loop}
            onChange={(event) => setLoop(event.target.checked)}
            aria-label="Loop ground playback"
          />
          Loop
        </label>
        <button
          type="button"
          onClick={() => setCamera(INITIAL_CAMERA)}
          aria-label="Reset ground 3D view"
          className="rounded border border-slate-700 px-2 py-1"
        >
          Reset 3D view
        </button>
      </div>
      <p
        role="status"
        aria-label="Ground playback position"
        className="text-sm text-slate-200"
      >
        <strong>
          {frame.phase[0].toUpperCase()}
          {frame.phase.slice(1)}
        </strong>
        {" · "}absolute {(pairedFrame?.timeS ?? frame.timeS).toFixed(4)} s
        {" · "}elapsed{" "}
        {((pairedFrame?.timeS ?? frame.timeS) - startTimeS).toFixed(4)} s
        {pairedFrame && (
          <>
            {" "}
            · primary {pairedFrame.primaryState} · comparison{" "}
            {pairedFrame.comparisonState}
          </>
        )}
      </p>
      <canvas
        ref={canvasRef}
        width={FLIGHT_PLAYBACK_LOGICAL_SIZE.width}
        height={FLIGHT_PLAYBACK_LOGICAL_SIZE.height}
        tabIndex={0}
        aria-label="Interactive 3D ground playback"
        aria-description="Phase-colored strict result in target_frame x downrange, y up, z right. Drag to orbit and wheel to zoom."
        style={{ width: "100%", height: "auto", aspectRatio: "860 / 420" }}
        className="w-full touch-none rounded-lg border border-slate-800 bg-slate-950/60 outline-none focus:ring-2 focus:ring-sky-500"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture?.(event.pointerId);
          dragRef.current = {
            pointerId: event.pointerId,
            x: event.clientX,
            y: event.clientY,
          };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.pointerId !== event.pointerId) return;
          const deltaX = event.clientX - drag.x;
          const deltaY = event.clientY - drag.y;
          dragRef.current = {
            pointerId: event.pointerId,
            x: event.clientX,
            y: event.clientY,
          };
          setCamera((current) => ({
            ...current,
            yawRad: current.yawRad + deltaX * 0.008,
            pitchRad: Math.max(
              -1.35,
              Math.min(1.35, current.pitchRad + deltaY * 0.008),
            ),
          }));
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
        onWheel={(event) => {
          event.preventDefault();
          const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
          setCamera((current) => ({
            ...current,
            zoom: Math.max(0.4, Math.min(4, current.zoom * factor)),
          }));
        }}
      />
      <div
        className="flex flex-wrap gap-3 text-xs text-slate-300"
        aria-label="Ground phase legend"
      >
        {(["Impact", "Bounce", "Skid", "Roll", "Rest"] as const).map(
          (label, index) => (
            <span
              key={label}
              className="rounded border px-2 py-0.5"
              style={{
                borderColor: [
                  "#ef476f",
                  "#a78bfa",
                  "#f59e0b",
                  "#34d399",
                  "#60a5fa",
                ][index],
              }}
            >
              {label}
            </span>
          ),
        )}
        <span>Carry / first contact</span>
        <span>
          {timeline.isComplete ? "Total / rest" : "Observed total / end"}
        </span>
        {showComparison && comparison && (
          <span className="text-cyan-300">
            Dashed path / diamonds: comparison
          </span>
        )}
      </div>
      <p className="text-xs text-slate-500">
        Locked orthographic physical scale: x downrange, y up, z right. Drag to
        orbit; wheel to zoom.
      </p>
    </section>
  );
}
