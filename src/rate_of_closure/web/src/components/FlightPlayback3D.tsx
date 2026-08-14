/** Rotatable, zoomable, lifecycle-safe real-time ball-flight playback. */

import { useEffect, useMemo, useRef, useState } from "react";

import {
  drawFlightPlayback,
  FLIGHT_PLAYBACK_LOGICAL_SIZE,
} from "./flightPlaybackDrawing";
import { PlaybackTimeline, validatePlaybackPoints } from "../model/flightPlayback";
import type { FlightPoint } from "../model/flight";
import type { SpatialTargetTs } from "../model/spatialTarget";
import { spatialTargetSummary } from "./spatialTargetPresentation";
import { observeCanvas } from "./canvasDisplay";
import { CameraControlBar } from "./CameraControlBar";
import { pointerCoordinates } from "./pointerCoordinates";
import {
  applyManualOverride,
  applyCameraPreset,
  movingSubjectCameraState,
  recenterCamera,
  safeTrackingZoom,
  setFaceOnSide,
  setTrackingEnabled,
  updateTrackingTarget,
  withCameraZoom,
  withManualOrbit,
} from "../model/cameraCommands";
import {
  applyCameraPreference,
  preferenceFromCameraState,
  type CameraPreference,
} from "../model/cameraPreferences";

interface Props {
  points: readonly FlightPoint[];
  comparisonPoints?: readonly FlightPoint[];
  spatialTarget?: SpatialTargetTs;
  synchronizedTimeS?: number;
  hideTransport?: boolean;
  cameraPreference?: CameraPreference;
  onCameraPreferenceChange?: (preference: CameraPreference) => void;
}

const SPEEDS = [0.25, 0.5, 1, 2, 4];
const BALL_CLEARANCE_RADIUS_M = 0.05;

export function FlightPlayback3D({
  points,
  comparisonPoints = [],
  spatialTarget,
  synchronizedTimeS,
  hideTransport = false,
  cameraPreference,
  onCameraPreferenceChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const timeRef = useRef(0);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [camera, setCamera] = useState(() => cameraPreference === undefined
    ? movingSubjectCameraState()
    : applyCameraPreference(movingSubjectCameraState(), cameraPreference));
  const timeline = useMemo(
    () => (points.length > 0 ? new PlaybackTimeline(points) : null),
    [points],
  );
  const duration = timeline?.duration ?? 0;
  const apexTime = timeline?.apexTime ?? 0;
  const displayedTime = synchronizedTimeS === undefined
    ? time
    : Math.min(Math.max(synchronizedTimeS, 0), duration);
  useMemo(() => {
    if (comparisonPoints.length > 0) validatePlaybackPoints(comparisonPoints);
  }, [comparisonPoints]);

  const updatePreferenceCamera = (
    transform: (current: typeof camera) => typeof camera,
  ) => {
    setCamera((current) => {
      const next = transform(current);
      if (cameraPreference !== undefined) {
        onCameraPreferenceChange?.(
          preferenceFromCameraState(next, cameraPreference),
        );
      }
      return next;
    });
  };

  useEffect(() => {
    if (cameraPreference === undefined) return;
    setCamera((current) => applyCameraPreference(current, cameraPreference));
  }, [cameraPreference]);

  useEffect(() => {
    setPlaying(false);
    timeRef.current = 0;
    setTime(0);
  }, [points]);

  useEffect(() => {
    timeRef.current = time;
  }, [time]);

  useEffect(() => {
    if (synchronizedTimeS !== undefined || !playing || duration <= 0) return;
    let animationId = 0;
    let previous = performance.now();
    const animate = (now: number) => {
      const elapsed = Math.max(0, now - previous) / 1000;
      previous = now;
      const unboundedNext = timeRef.current + elapsed * speed;
      const next = loop && unboundedNext >= duration
        ? unboundedNext % duration
        : Math.min(duration, unboundedNext);
      timeRef.current = next;
      setTime(next);
      if (!loop && next >= duration) setPlaying(false);
      else animationId = window.requestAnimationFrame(animate);
    };
    animationId = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(animationId);
  }, [playing, duration, speed, loop, synchronizedTimeS]);

  const frame = useMemo(
    () => timeline?.frameAt(displayedTime) ?? null,
    [timeline, displayedTime],
  );

  const baseHalfExtentM = useMemo(() => {
    const positions = [...points, ...comparisonPoints].map((point) => point.position);
    if (positions.length === 0) return 1;
    const spans = [0, 1, 2].map((axis) => {
      const values = positions.map((position) => position[axis]);
      return (Math.max(...values) - Math.min(...values)) / 2;
    });
    return Math.max(1, ...spans);
  }, [points, comparisonPoints]);

  useEffect(() => {
    if (!frame) return;
    setCamera((current) => updateTrackingTarget(
      current, frame.position, Math.max(0.25, baseHalfExtentM * 0.25),
    ));
  }, [frame, baseHalfExtentM]);

  useEffect(() => {
    const draw = () => {
      if (!canvasRef.current || (!frame && !spatialTarget)) return;
      drawFlightPlayback(
        canvasRef.current, points, comparisonPoints, frame?.position ?? null, camera, spatialTarget,
      );
    };
    return observeCanvas(canvasRef, draw);
  }, [points, comparisonPoints, frame, camera, spatialTarget]);

  const togglePlayback = () => {
    if (playing) setPlaying(false);
    else {
      if (time >= duration) {
        timeRef.current = 0;
        setTime(0);
      }
      setPlaying(duration > 0);
    }
  };

  const jump = (targetTime: number) => {
    setPlaying(false);
    timeRef.current = targetTime;
    setTime(targetTime);
    const targetFrame = timeline?.frameAt(targetTime);
    if (targetFrame) {
      setCamera((current) => current.trackingEnabled
        ? recenterCamera(current, targetFrame.position) : current);
    }
  };

  const stepFrame = (direction: -1 | 1) => {
    if (timeline === null) return;
    jump(timeline.stepTime(timeRef.current, direction));
  };

  return (
    <section className="space-y-3" aria-label="3D ball-flight playback">
      {!hideTransport && <div className="flex flex-wrap items-center gap-2 text-xs">
        <button
          type="button"
          disabled={duration <= 0}
          onClick={() => jump(0)}
          aria-label="Jump to Launch"
          className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
        >
          Launch
        </button>
        <button
          type="button"
          disabled={duration <= 0}
          onClick={() => stepFrame(-1)}
          aria-label="Step Back One Frame"
          title="Pause and step to the previous solver-owned trajectory sample"
          className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
        >
          −1 frame
        </button>
        <button
          type="button"
          disabled={duration <= 0}
          onClick={() => stepFrame(1)}
          aria-label="Step Forward One Frame"
          title="Pause and step to the next solver-owned trajectory sample"
          className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
        >
          +1 frame
        </button>
        <button
          type="button"
          disabled={duration <= 0}
          onClick={togglePlayback}
          aria-label={playing ? "Pause Ball Flight" : "Play Ball Flight"}
          className="rounded border border-sky-500/60 bg-sky-500/10 px-3 py-1 font-semibold text-sky-200 disabled:opacity-40"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          disabled={duration <= 0}
          onClick={() => {
            jump(0);
            setPlaying(true);
          }}
          aria-label="Restart Ball Flight"
          className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
        >
          Restart
        </button>
        <button
          type="button"
          disabled={duration <= 0}
          onClick={() => jump(apexTime)}
          aria-label="Jump to Apex"
          className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
        >
          Apex
        </button>
        <button
          type="button"
          disabled={duration <= 0}
          onClick={() => jump(duration)}
          aria-label="Jump to Landing"
          className="rounded border border-slate-700 px-2 py-1 text-slate-200 disabled:opacity-40"
        >
          Landing
        </button>
        <input
          type="range"
          min={0}
          max={duration}
          step={0.001}
          value={time}
          disabled={duration <= 0}
          onChange={(event) => jump(Number(event.target.value))}
          aria-label="Ball Flight Time"
          title="Physical trajectory time [s], interpolated between solver samples"
          className="min-w-36 flex-1"
        />
        <label className="flex items-center gap-1 text-slate-300">
          Speed
          <select
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
            aria-label="Playback Speed"
            className="rounded border border-slate-700 bg-slate-900 px-1 py-1"
          >
            {SPEEDS.map((option) => (
              <option key={option} value={option}>{option}×</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1 text-slate-300">
          <input
            type="checkbox"
            checked={loop}
            onChange={(event) => setLoop(event.target.checked)}
            aria-label="Loop Ball Flight Playback"
          />
          Loop
        </label>
        <output className="min-w-24 text-right tabular-nums text-slate-300">
          {time.toFixed(2)} / {duration.toFixed(2)} s
        </output>
      </div>}
      <CameraControlBar state={camera} subjectLabel="Ball"
        onPreset={(preset) => updatePreferenceCamera((current) => applyCameraPreset(current, preset))}
        onFaceOnSide={(side) => updatePreferenceCamera((current) => setFaceOnSide(current, side))}
        onTracking={(enabled) => updatePreferenceCamera((current) => setTrackingEnabled(
          current, enabled, frame?.position ?? [0, 0, 0],
        ))}
        onAutoFit={(enabled) => updatePreferenceCamera((current) => ({
          ...current,
          autoFitEnabled: enabled,
          zoom: enabled
            ? safeTrackingZoom(current.zoom, BALL_CLEARANCE_RADIUS_M, baseHalfExtentM)
            : current.zoom,
        }))}
        onRecenter={() => setCamera((current) => recenterCamera(
          current, frame?.position ?? [0, 0, 0],
        ))} />
      <canvas
        ref={canvasRef}
        width={FLIGHT_PLAYBACK_LOGICAL_SIZE.width}
        height={FLIGHT_PLAYBACK_LOGICAL_SIZE.height}
        style={{
          width: "100%",
          height: "auto",
          aspectRatio: `${FLIGHT_PLAYBACK_LOGICAL_SIZE.width} / ${FLIGHT_PLAYBACK_LOGICAL_SIZE.height}`,
        }}
        tabIndex={0}
        aria-label="Interactive 3D ball-flight playback"
        aria-description={spatialTarget ? `Includes ${spatialTargetSummary(spatialTarget)}` : undefined}
        title="Drag to rotate; use the mouse wheel to zoom. App frame: x target, y up, z right; SI metres and seconds."
        className="w-full touch-none rounded-lg border border-slate-800 bg-slate-950/60 outline-none focus:ring-2 focus:ring-sky-500"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture?.(event.pointerId);
          const pointer = pointerCoordinates(event.nativeEvent);
          dragRef.current = { pointerId: event.pointerId, ...pointer };
          setCamera(applyManualOverride);
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.pointerId !== event.pointerId) return;
          const pointer = pointerCoordinates(event.nativeEvent);
          const deltaX = pointer.x - drag.x;
          const deltaY = pointer.y - drag.y;
          dragRef.current = { pointerId: event.pointerId, ...pointer };
          setCamera((current) => withManualOrbit(
            current,
            current.yawRad + deltaX * 0.008,
            Math.max(-1.35, Math.min(1.35, current.pitchRad + deltaY * 0.008)),
          ));
        }}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
        onWheel={(event) => {
          event.preventDefault();
          const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
          updatePreferenceCamera((current) => withCameraZoom(
            current,
            current.autoFitEnabled
              ? safeTrackingZoom(
                current.zoom * factor, BALL_CLEARANCE_RADIUS_M, baseHalfExtentM,
              )
              : current.zoom * factor,
          ));
        }}
      />
      {spatialTarget && (
        <p role="status" aria-label="Active 3D spatial target"
          className="rounded border border-amber-400/30 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
          Active 3D target · {spatialTargetSummary(spatialTarget)}
        </p>
      )}
      <p className="text-xs text-slate-500">
        Drag to rotate and wheel to zoom. Orthographic axes use one locked physical scale per metre:
        x targets downrange, y points up, and z points right. Time is solver trajectory time [s].
      </p>
    </section>
  );
}
