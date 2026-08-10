/** Orbitable, exportable engineering still for one exact impact event. */

import { useEffect, useMemo, useRef, useState } from "react";

import type { ClubSpec } from "../model/club";
import type { ImpactScenario } from "../model/impact";
import { impactKinematics } from "../model/impactKinematics";
import type { SimulationRunTs } from "../model/simulation";
import { project } from "./clubCanvasGeometry";
import { impactSceneGeometry } from "./impactSceneGeometry";
import { impactSceneExportPayload } from "./impactSceneExport";
import { impactSceneSvg } from "./impactSceneSvg";
import { CameraControlBar } from "./CameraControlBar";
import { pointerCoordinates } from "./pointerCoordinates";
import {
  applyManualOverride,
  applyCameraPreset,
  defaultCameraState,
  recenterCamera,
  safeTrackingZoom,
  setFaceOnSide,
  withCameraZoom,
  withManualOrbit,
} from "../model/cameraCommands";

const VELOCITY_KEYS = ["total", "axisTranslation", "shaftRotation", "otherRotation", "withoutShaft"] as const;
const DPLANE_LAYERS = [
  ["faceNormal", "Face-Center Normal", "Delivered face-center normal in the app frame."],
  ["faceCenterTravel", "Face-Center Travel", "Rigid-body face-center velocity direction including omega cross r."],
  ["dplaneNormal", "D-Plane Normal", "Normal to the plane spanned by face-center travel and face normal."],
  ["projectedPath", "Projected Path", "Face-center travel projected onto the ground plane."],
  ["spinLoftSector", "Spin-Loft Sector", "Shaded exact 3D angle between face-center travel and face normal."],
] as const;
const PREFERENCE_KEY = "rate-of-closure.impact-scene-layers.v1";
const DEFAULT_VISIBLE = [...VELOCITY_KEYS, ...DPLANE_LAYERS.map(([key]) => key)];

function initialVisible(): ReadonlySet<string> {
  try {
    if (typeof localStorage?.getItem !== "function") return new Set(DEFAULT_VISIBLE);
    const stored = JSON.parse(localStorage.getItem(PREFERENCE_KEY) ?? "null");
    if (Array.isArray(stored) && stored.every((value) => typeof value === "string")) {
      return new Set(stored);
    }
  } catch { /* Ignore corrupt local preference and restore declared defaults. */ }
  return new Set(DEFAULT_VISIBLE);
}

function download(name: string, contents: Blob): void {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(contents);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

export function ImpactSceneCanvas({ run, scenario, club }: {
  run: SimulationRunTs;
  scenario: ImpactScenario;
  club: ClubSpec;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [camera, setCamera] = useState(() => ({
    ...defaultCameraState(),
    zoom: 2.2,
  }));
  const [visible, setVisible] = useState<ReadonlySet<string>>(initialVisible);
  const scene = useMemo(() => impactKinematics(run, scenario, club), [run, scenario, club]);
  const geometry = useMemo(() => impactSceneGeometry(scene, visible), [scene, visible]);

  useEffect(() => {
    try {
      if (typeof localStorage?.setItem === "function") {
        localStorage.setItem(PREFERENCE_KEY, JSON.stringify([...visible].sort()));
      }
    } catch { /* Storage denial must not disable impact inspection. */ }
  }, [visible]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.width * 0.58 * dpr));
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#020617";
    context.fillRect(0, 0, canvas.width, canvas.height);
    for (const fill of geometry.fills) {
      const points = fill.points.map((point) => project(
        point, canvas.width, canvas.height, camera.zoom, camera.yawRad, camera.pitchRad,
        camera.targetM,
      ));
      context.beginPath();
      points.forEach(([x, y], index) => index === 0 ? context.moveTo(x, y) : context.lineTo(x, y));
      context.closePath();
      context.globalAlpha = fill.alpha;
      context.fillStyle = fill.color;
      context.fill();
      context.globalAlpha = 1;
    }
    for (const line of geometry.lines) {
      const points = line.points.map((point) => project(
        point, canvas.width, canvas.height, camera.zoom, camera.yawRad, camera.pitchRad,
        camera.targetM,
      ));
      context.beginPath();
      context.strokeStyle = line.color;
      context.lineWidth = line.width * dpr;
      context.setLineDash((line.dash ?? []).map((value) => value * dpr));
      points.forEach(([x, y], index) => index === 0 ? context.moveTo(x, y) : context.lineTo(x, y));
      context.stroke();
      if (line.arrow && points.length >= 2) {
        const [startX, startY] = points[points.length - 2];
        const [tipX, tipY] = points[points.length - 1];
        const angle = Math.atan2(tipY - startY, tipX - startX);
        context.fillStyle = line.color;
        context.beginPath();
        context.moveTo(tipX, tipY);
        context.lineTo(tipX - 10 * dpr * Math.cos(angle - 0.45), tipY - 10 * dpr * Math.sin(angle - 0.45));
        context.lineTo(tipX - 10 * dpr * Math.cos(angle + 0.45), tipY - 10 * dpr * Math.sin(angle + 0.45));
        context.closePath();
        context.fill();
      }
    }
    const [ballX, ballY] = project(geometry.ballCenter, canvas.width, canvas.height, camera.zoom, camera.yawRad, camera.pitchRad, camera.targetM);
    context.fillStyle = "rgba(248, 250, 252, 0.16)";
    context.strokeStyle = "#e2e8f0";
    context.lineWidth = 1.5 * dpr;
    context.beginPath();
    context.arc(ballX, ballY, Math.min(canvas.width, canvas.height) * camera.zoom * 0.02135, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    const [contactX, contactY] = project(geometry.contactPoint, canvas.width, canvas.height, camera.zoom, camera.yawRad, camera.pitchRad, camera.targetM);
    context.setLineDash([]);
    context.strokeStyle = "#fde047";
    context.lineWidth = 2 * dpr;
    context.beginPath();
    context.arc(contactX, contactY, 6 * dpr, 0, Math.PI * 2);
    context.stroke();
  }, [camera, geometry]);

  return (
    <section aria-label="Interactive Impact Scene" className="mb-3 space-y-2 rounded-lg border border-sky-400/30 bg-slate-950/70 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-sky-200">Interactive Impact Still</h3>
          <p className="text-xs text-slate-400">Drag to orbit · scroll to zoom · arrow keys rotate · exact event at {scene.eventTimeS.toFixed(4)} s</p>
        </div>
      </div>
      <CameraControlBar state={camera} subjectLabel="Impact" trackingAvailable={false}
        onPreset={(preset) => setCamera((current) => applyCameraPreset(current, preset))}
        onFaceOnSide={(side) => setCamera((current) => setFaceOnSide(current, side))}
        onTracking={() => undefined}
        onAutoFit={(enabled) => setCamera((current) => ({
          ...current,
          autoFitEnabled: enabled,
          zoom: enabled ? safeTrackingZoom(current.zoom, 0.35, 1) : current.zoom,
        }))}
        onRecenter={() => setCamera((current) => recenterCamera(current, geometry.contactPoint))} />
      <div className="flex flex-wrap gap-3 text-xs">
        {DPLANE_LAYERS.map(([key, label, meaning]) => <label key={key} title={meaning} className="flex cursor-pointer items-center gap-1 text-slate-300">
          <input type="checkbox" checked={visible.has(key)} onChange={() => setVisible((current) => {
            const next = new Set(current);
            if (next.has(key)) next.delete(key); else next.add(key);
            return next;
          })} /> {label}
        </label>)}
        {scene.vectors.map((vector) => <label key={vector.key} title={vector.meaning} className="flex cursor-pointer items-center gap-1 text-slate-300">
          <input type="checkbox" checked={visible.has(vector.key)} onChange={() => setVisible((current) => {
            const next = new Set(current);
            if (next.has(vector.key)) next.delete(vector.key); else next.add(vector.key);
            return next;
          })} /> {vector.label}
        </label>)}
      </div>
      <p className="text-xs text-cyan-200" aria-live="polite">
        <b>Face-Center Spin Loft:</b> {scene.faceCenterDPlane.spinLoft3dDeg?.toFixed(2) ?? "Unavailable"}° (exact 3D) ·{" "}
        {scene.faceCenterDPlane.planarSpinLoftDeg?.toFixed(2) ?? "Unavailable"}° planar approximation · residual{" "}
        {scene.faceCenterDPlane.spinLoftResidualDeg?.toFixed(2) ?? "Unavailable"}°. Positive D-plane tilt is face-right and fade-side only under the current right-handed display convention.
      </p>
      <canvas
        ref={canvasRef}
        tabIndex={0}
        role="img"
        aria-label="Rotatable 3D wedge impact scene with frame-explicit engineering vectors"
        className="w-full cursor-grab touch-none rounded border border-slate-800 active:cursor-grabbing focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        onPointerDown={(event) => {
          dragRef.current = pointerCoordinates(event.nativeEvent);
          event.currentTarget.setPointerCapture?.(event.pointerId);
          setCamera(applyManualOverride);
        }}
        onPointerMove={(event) => {
          if (!dragRef.current) return;
          const pointer = pointerCoordinates(event.nativeEvent);
          const dx = pointer.x - dragRef.current.x;
          const dy = pointer.y - dragRef.current.y;
          setCamera((current) => withManualOrbit(
            current,
            current.yawRad - dx * 0.008,
            Math.max(-1.4, Math.min(1.4, current.pitchRad + dy * 0.008)),
          ));
          dragRef.current = pointer;
        }}
        onPointerUp={(event) => { dragRef.current = null; event.currentTarget.releasePointerCapture?.(event.pointerId); }}
        onWheel={(event) => { event.preventDefault(); setCamera((current) => withCameraZoom(
          current,
          current.zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1),
        )); }}
        onKeyDown={(event) => {
          const delta = event.shiftKey ? 0.2 : 0.08;
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") setCamera((current) => withManualOrbit(
            current, current.yawRad + (event.key === "ArrowLeft" ? delta : -delta), current.pitchRad,
          ));
          if (event.key === "ArrowUp" || event.key === "ArrowDown") setCamera((current) => withManualOrbit(
            current, current.yawRad,
            Math.max(-1.4, Math.min(1.4, current.pitchRad + (event.key === "ArrowUp" ? delta : -delta))),
          ));
        }}
      />
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={() => canvasRef.current?.toBlob((blob) => { if (blob) download("wedge-impact.png", blob); }, "image/png")} className="rounded border border-slate-600 px-2 py-1 text-xs hover:border-sky-400">Export High-Resolution PNG</button>
        <button type="button" onClick={() => download("wedge-impact.svg", new Blob([impactSceneSvg(geometry, { yaw: camera.yawRad, pitch: camera.pitchRad, zoom: camera.zoom })], { type: "image/svg+xml" }))} className="rounded border border-slate-600 px-2 py-1 text-xs hover:border-sky-400">Export Vector SVG</button>
        <button type="button" onClick={() => download("wedge-impact.json", new Blob([JSON.stringify(impactSceneExportPayload(scene, visible, { yaw: camera.yawRad, pitch: camera.pitchRad, zoom: camera.zoom }), null, 2)], { type: "application/json" }))} className="rounded border border-slate-600 px-2 py-1 text-xs hover:border-sky-400">Export Scene Data</button>
      </div>
    </section>
  );
}
