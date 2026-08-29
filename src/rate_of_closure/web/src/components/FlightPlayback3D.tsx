/** Rotatable, zoomable, lifecycle-safe real-time ball-flight playback. */

import { useEffect, useMemo, useRef, useState } from "react";

import {
  drawFlightPlayback,
  FLIGHT_PLAYBACK_LOGICAL_SIZE,
  type PlaybackCamera,
} from "./flightPlaybackDrawing";
import { PlaybackTimeline, validatePlaybackPoints } from "../model/flightPlayback";
import { PlaybackTransportBar } from "./PlaybackTransportBar";
import type { FlightPoint } from "../model/flight";
import type { SpatialTargetTs } from "../model/spatialTarget";
import { spatialTargetSummary } from "./spatialTargetPresentation";
import { observeCanvas } from "./canvasDisplay";

interface Props {
  points: readonly FlightPoint[];
  comparisonPoints?: readonly FlightPoint[];
  spatialTarget?: SpatialTargetTs;
  selectedTimeS?: number | null;
  selectedCommandId?: number;
}

// #4571 extension point: named cameras (Face-On / Down-the-Line) own their
// state in model/cameraCommands.ts. When that seam lands for playback, seed
// this orbit camera from `canvasAngles(cameraPreset(...))` instead of the
// constant below; playback must never re-implement camera state (#4800 P8).
const INITIAL_CAMERA: PlaybackCamera = {
  yawRad: -0.65,
  pitchRad: 0.38,
  zoom: 1,
};

export function FlightPlayback3D({
  points, comparisonPoints = [], spatialTarget, selectedTimeS = null, selectedCommandId = 0,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const [time, setTime] = useState(0);
  const [camera, setCamera] = useState(INITIAL_CAMERA);
  const timeline = useMemo(
    () => (points.length > 0 ? new PlaybackTimeline(points) : null),
    [points],
  );
  const duration = timeline?.duration ?? 0;
  const apexTime = timeline?.apexTime ?? 0;
  useMemo(() => {
    if (comparisonPoints.length > 0) validatePlaybackPoints(comparisonPoints);
  }, [comparisonPoints]);

  useEffect(() => setTime(0), [points]);

  useEffect(() => {
    if (selectedTimeS === null || !Number.isFinite(selectedTimeS)) return;
    setTime(Math.max(0, Math.min(duration, selectedTimeS)));
  }, [selectedTimeS, selectedCommandId, duration]);

  const frame = useMemo(
    () => timeline?.frameAt(time) ?? null,
    [timeline, time],
  );

  useEffect(() => {
    const draw = () => {
      if (!canvasRef.current || (!frame && !spatialTarget)) return;
      drawFlightPlayback(
        canvasRef.current, points, comparisonPoints, frame?.position ?? null, camera, spatialTarget,
      );
    };
    return observeCanvas(canvasRef, draw);
  }, [points, comparisonPoints, frame, camera, spatialTarget]);

  return (
    <section className="space-y-3" aria-label="3D ball-flight playback">
      <PlaybackTransportBar
        subjectLabel="Ball Flight"
        subjectPhrase="ball flight"
        timeS={time}
        durationS={duration}
        events={[
          { label: "Launch", timeS: 0 },
          { label: "Apex", timeS: apexTime },
          { label: "Landing", timeS: duration },
        ]}
        scrubTitle="Physical trajectory time [s], interpolated between solver samples"
        onTimeChange={setTime}
      />
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
        className="w-full touch-none rounded-lg border border-slate-800 bg-slate-950/60 focus-visible:outline-none focus:ring-2 focus:ring-sky-500"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture?.(event.pointerId);
          dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.pointerId !== event.pointerId) return;
          const deltaX = event.clientX - drag.x;
          const deltaY = event.clientY - drag.y;
          dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
          setCamera((current) => ({
            ...current,
            yawRad: current.yawRad + deltaX * 0.008,
            pitchRad: Math.max(-1.35, Math.min(1.35, current.pitchRad + deltaY * 0.008)),
          }));
        }}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
        onWheel={(event) => {
          event.preventDefault();
          const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
          setCamera((current) => ({
            ...current,
            zoom: Math.max(0.4, Math.min(4, current.zoom * factor)),
          }));
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
