/**
 * Animated 3D clubhead on a plain canvas — no WebGL dependency.
 *
 * Simple orthographic projection of the same wireframe the PyQt6 view
 * draws: face plate, body outline, shaft stub, impact point, and the
 * reference vs impact-point velocity arrows, spinning under the
 * scenario's angular velocity. Playback is user-controllable —
 * play/pause, 0.1x-3x speed, and Head Fixed vs Head Moving display
 * modes — matching the desktop app.
 *
 * Optional photorealistic mode: an STL file input (client-side
 * FileReader, nothing uploaded) swaps the procedural wireframe for the
 * user's clubhead mesh, normalized onto the same envelope and rendered
 * as flat-shaded painter's-algorithm triangles — depth-sorted by
 * distance along the camera's forward axis, shaded by |normal · light|
 * with the same fixed world light as the desktop app.
 */

import { useEffect, useRef, useState } from "react";

import { solve, type ImpactScenario } from "../model/impact";
import {
  AUTO_FIT_CLEARANCE_FRACTION,
  applyManualCameraOverride,
  applyCameraView,
  autoFitCamera,
  defaultCameraState,
  enforceTrackingClearance,
  recenterCamera,
  setAutoFitFallback,
  setCameraTracking,
  setFaceOnSide,
  trackingStateId,
  updateTrackingTarget,
  withCameraZoom,
  type CameraState,
  type CameraViewId,
  type FaceOnSide,
} from "../model/cameraPresets";
import { loadHeadMesh, type HeadMesh } from "../model/mesh";
import { getChartColor } from "../model/theme";
import {
  SHAFT_LEN,
  add,
  apply,
  headParts,
  projectAroundTarget,
  rodrigues,
  type Vec3,
} from "./clubCanvasGeometry";
import { drawEngineeringCgSymbol } from "./engineeringSymbols";
import { ClubCameraControls } from "./ClubCameraControls";
import {
  ClubCanvasPlaybackControls,
  VIEW_MODES,
  type ViewMode,
} from "./ClubCanvasPlaybackControls";

export { VIEW_MODES } from "./ClubCanvasPlaybackControls";
export type { ViewMode } from "./ClubCanvasPlaybackControls";

const SPAN_MS = 8.0;
const STEPS = 48;

// H6 accent alignment (#4125): chart-palette accents come from the
// shared model/theme.ts palette; only the neutral body tone is local.
const COLORS = {
  face: getChartColor(0),
  body: "#8b949e",
  shaft: getChartColor(7),
  vRef: getChartColor(1),
  vPoint: getChartColor(3),
  impact: getChartColor(6),
  cog: getChartColor(2),
};

// STL-mesh shading constants — identical to the PyQt6 club view.
const LIGHT_LEN = Math.hypot(0.3, 0.8, 0.5);
const LIGHT_DIR: Vec3 = [0.3 / LIGHT_LEN, 0.8 / LIGHT_LEN, 0.5 / LIGHT_LEN];
const MESH_BASE_RGB = [0.56, 0.62, 0.7] as const;
const MESH_AMBIENT = 0.22;
const MESH_SPECULAR = 0.32;



export function ClubCanvas({
  scenario,
  externalMesh = null,
  hoselPoint = null,
  cogPoint = null,
  initialPhase = 0,
}: {
  scenario: ImpactScenario;
  /** A generated head (e.g. parametric club head) to render; the STL
   *  loader and the Procedural Head reset keep working alongside it. */
  externalMesh?: HeadMesh | null;
  /** Generated head's hosel — the shaft line attaches there (H1). */
  hoselPoint?: Vec3 | null;
  /** Generated head's divergence-theorem volumetric COG (H1). */
  cogPoint?: Vec3 | null;
  /** Deterministic phase seam used by camera-fit regression tests. */
  initialPhase?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef(initialPhase);
  const priorDrawPhaseRef = useRef(initialPhase);
  // Orbit camera state lives in refs so dragging never re-runs effects.
  const initialCamera = useRef(defaultCameraState());
  const [camera, setCamera] = useState<CameraState>(initialCamera.current);
  const [canonicalOrientation, setCanonicalOrientation] = useState(true);
  const cameraRef = useRef(initialCamera.current);
  const yawRef = useRef(initialCamera.current.yawRad);
  const pitchRef = useRef(initialCamera.current.pitchRad);
  const zoomRef = useRef(initialCamera.current.zoom);
  const subjectRadiusRef = useRef(0.4);
  const subjectRef = useRef<Vec3>([0, 0, 0]);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [mode, setMode] = useState<ViewMode>(VIEW_MODES[1]);
  const [mesh, setMesh] = useState<HeadMesh | null>(null);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [showCg, setShowCg] = useState(true);

  const storeCamera = (next: CameraState) => {
    cameraRef.current = next;
    yawRef.current = next.yawRad;
    pitchRef.current = next.pitchRad;
    zoomRef.current = next.zoom;
    setCamera(next);
  };

  const updateCamera = (update: (current: CameraState) => CameraState) => {
    storeCamera(update(cameraRef.current));
  };

  useEffect(() => {
    if (externalMesh) {
      setMesh(externalMesh);
      setMeshError(null);
    }
  }, [externalMesh]);

  const onStlChosen = (file: File | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setMesh(loadHeadMesh(reader.result as ArrayBuffer));
        setMeshError(null);
      } catch (err) {
        setMeshError(err instanceof Error ? err.message : String(err));
      }
    };
    reader.onerror = () => setMeshError("could not read the selected file");
    reader.readAsArrayBuffer(file);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const result = solve(scenario);
    const omega = result.omegaDps.map((c) => (c * Math.PI) / 180) as Vec3;
    const parts = headParts(scenario);
    const moving = mode === VIEW_MODES[1];
    const baseZoom = moving ? 0.9 : 1.6;
    const speedMps = scenario.clubheadSpeedMph * 0.44704;

    const draw = () => {
      // Render at device resolution so the canvas stays sharp on
      // high-DPI displays and at any layout width.
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const bw = Math.max(1, Math.round(rect.width * dpr));
      const bh = Math.max(1, Math.round(rect.width * 0.68 * dpr));
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      const { width: w, height: h } = canvas;
      ctx.clearRect(0, 0, w, h);
      const backdrop = ctx.createRadialGradient(
        w / 2,
        h * 0.55,
        h * 0.1,
        w / 2,
        h * 0.55,
        h * 0.9,
      );
      backdrop.addColorStop(0, "rgba(30, 41, 59, 0.55)");
      backdrop.addColorStop(1, "rgba(2, 6, 23, 0)");
      ctx.fillStyle = backdrop;
      ctx.fillRect(0, 0, w, h);
      const phase = phaseRef.current - 0.5;
      const timeS = (phase * SPAN_MS) / 1000;
      const rot = rodrigues(omega, timeS);
      const offset: Vec3 = moving ? [speedMps * timeS, 0, 0] : [0, 0, 0];
      subjectRef.current = offset;
      const priorCamera = cameraRef.current;
      const wrapped = phaseRef.current < priorDrawPhaseRef.current;
      priorDrawPhaseRef.current = phaseRef.current;
      const tracked = wrapped && priorCamera.trackingEnabled
        && !priorCamera.trackingSuspended
        ? recenterCamera(priorCamera, offset)
        : updateTrackingTarget(priorCamera, offset);
      if (tracked !== priorCamera) {
        cameraRef.current = tracked;
        yawRef.current = tracked.yawRad;
        pitchRef.current = tracked.pitchRad;
        zoomRef.current = tracked.zoom;
        setCamera(tracked);
      }
      const place = (p: Vec3): Vec3 => add(apply(rot, p), offset);

      // Resolve every discontinuous geometry/target change before deriving the
      // projection scale. The fallback must affect this frame, not the next.
      let shift: Vec3 = [0, 0, 0];
      if (mesh) {
        let xMax = -Infinity;
        for (const tri of mesh.triangles) {
          for (const vertex of tri) if (vertex[0] > xMax) xMax = vertex[0];
        }
        shift = [scenario.comToFaceMm / 1000 - xMax, 0, 0];
      }
      const generated = mesh !== null && mesh === externalMesh;
      let hosel = parts.hosel;
      let shaftEnd = parts.shaftEnd;
      if (generated && hoselPoint) {
        hosel = add(hoselPoint, shift);
        const lie = (scenario.lieAngleDeg * Math.PI) / 180;
        shaftEnd = [
          hosel[0],
          hosel[1] + Math.sin(lie) * SHAFT_LEN,
          hosel[2] - Math.cos(lie) * SHAFT_LEN,
        ];
      }
      const fitPoints = mesh
        ? mesh.triangles.flatMap((triangle) =>
          triangle.map((point) => add(point, shift)))
        : [...parts.face, ...parts.back];
      fitPoints.push(hosel, shaftEnd, parts.impact);
      subjectRadiusRef.current = Math.max(
        1e-9,
        ...fitPoints.map((point) => {
          const placed = place(point);
          const target = cameraRef.current.targetM;
          return Math.hypot(
            placed[0] - target[0],
            placed[1] - target[1],
            placed[2] - target[2],
          );
        }),
      );
      const cleared = enforceTrackingClearance(
        cameraRef.current,
        subjectRadiusRef.current,
        moving ? 0.42 : 0.24,
      );
      if (cleared !== cameraRef.current) {
        cameraRef.current = cleared;
        zoomRef.current = cleared.zoom;
        setCamera(cleared);
      }
      const zoom = baseZoom * zoomRef.current;
      const yaw = yawRef.current;
      const pitch = pitchRef.current;
      const frameFits = subjectRadiusRef.current * zoomRef.current
        <= (moving ? 0.42 : 0.24) * (1 - AUTO_FIT_CLEARANCE_FRACTION) + 1e-12;
      canvas.dataset.cameraRenderedZoom = zoomRef.current.toFixed(6);
      canvas.dataset.cameraRenderedSubjectFits = String(frameFits);
      const projectPoint = (point: Vec3) => projectAroundTarget(
        point, cameraRef.current.targetM, w, h, zoom, yaw, pitch,
      );

      const line = (pts: Vec3[], color: string, lw: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw * dpr;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        pts.forEach((p, i) => {
          const [px, py] = projectPoint(p);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      };

      if (moving) {
        ctx.setLineDash([4, 6]);
        line(
          [
            [-0.5, -0.05, 0],
            [0.5, -0.05, 0],
          ],
          COLORS.body,
          0.8,
        );
        ctx.setLineDash([]);
      }

      // Put the mesh's forward extent (its face plane) at com_to_face.
      if (mesh) {
        // Painter's algorithm: camera forward axis from the orbit
        // angles (same basis as project()); triangles sorted by
        // centroid depth along it, farthest drawn first.
        const fwd: Vec3 = [
          Math.cos(pitch) * Math.cos(yaw),
          Math.sin(pitch),
          Math.cos(pitch) * Math.sin(yaw),
        ];
        const shaded = mesh.triangles.map((tri, t) => {
          const placed = tri.map((v) => place(add(v, shift))) as [
            Vec3,
            Vec3,
            Vec3,
          ];
          const cx = (placed[0][0] + placed[1][0] + placed[2][0]) / 3;
          const cy = (placed[0][1] + placed[1][1] + placed[2][1]) / 3;
          const cz = (placed[0][2] + placed[1][2] + placed[2][2]) / 3;
          const depth = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
          const n = apply(rot, mesh.normals[t]);
          const lambert = Math.abs(
            n[0] * LIGHT_DIR[0] + n[1] * LIGHT_DIR[1] + n[2] * LIGHT_DIR[2],
          );
          const diffuse = (1 - MESH_AMBIENT - MESH_SPECULAR) * lambert;
          const specular = MESH_SPECULAR * lambert ** 20;
          const intensity = MESH_AMBIENT + diffuse + specular;
          return { placed, depth, intensity };
        });
        shaded.sort((a, b) => a.depth - b.depth);
        for (const { placed, intensity } of shaded) {
          const rgb = MESH_BASE_RGB.map((c) =>
            Math.round(Math.min(1, c * intensity) * 255),
          );
          ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
          ctx.beginPath();
          placed.forEach((p, i) => {
            const [px, py] = projectPoint(p);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.closePath();
          ctx.fill();
        }
      } else {
        line(parts.face.map(place), COLORS.face, 2.5);
        line(parts.back.map(place), COLORS.body, 1.2);
        parts.face.forEach((p, i) =>
          line([place(p), place(parts.back[i])], COLORS.body, 0.8),
        );
      }
      // Hosel-true shaft (H1) was included in the pre-raster fit envelope.
      line([place(hosel), place(shaftEnd)], COLORS.shaft, 2.5);

      if (showCg) {
        // Volumetric COG marker (divergence theorem); wireframe and
        // non-watertight STLs fall back to the reference point, which
        // is the spec CG location.
        const cgModel: Vec3 =
          generated && cogPoint ? add(cogPoint, shift) : [0, 0, 0];
        const [cx, cy] = projectPoint(place(cgModel));
        const r = 5 * dpr;
        drawEngineeringCgSymbol(ctx, cx, cy, r, COLORS.cog);
        ctx.fillStyle = COLORS.cog;
        ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillText("CG", cx + 9 * dpr, cy - 8 * dpr);
      }

      const arrow = (origin: Vec3, vec: Vec3, color: string) => {
        const scale = 0.0035;
        const tip: Vec3 = [
          origin[0] + vec[0] * scale,
          origin[1] + vec[1] * scale,
          origin[2] + vec[2] * scale,
        ];
        const [ox, oy] = projectPoint(origin);
        const [tx, ty] = projectPoint(tip);
        const angle = Math.atan2(ty - oy, tx - ox);
        const headLen = 11 * dpr;
        // Stop the shaft short so the filled head forms a clean point.
        const bx = tx - Math.cos(angle) * headLen * 0.7;
        const by = ty - Math.sin(angle) * headLen * 0.7;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5 * dpr;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(
          tx - headLen * Math.cos(angle - 0.45),
          ty - headLen * Math.sin(angle - 0.45),
        );
        ctx.lineTo(
          tx - headLen * Math.cos(angle + 0.45),
          ty - headLen * Math.sin(angle + 0.45),
        );
        ctx.closePath();
        ctx.fill();
      };
      const vRefMps = scenario.clubheadSpeedMph * 0.44704;
      arrow(offset, [vRefMps, 0, 0], COLORS.vRef);
      arrow(place(parts.impact), result.pointVelocityMps, COLORS.vPoint);

      const [ix, iy] = projectPoint(place(parts.impact));
      ctx.fillStyle = COLORS.impact;
      ctx.shadowColor = "rgba(255, 214, 10, 0.6)";
      ctx.shadowBlur = 8 * dpr;
      ctx.beginPath();
      ctx.arc(ix, iy, 5 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#94a3b8";
      ctx.font = `${12 * dpr}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillText(`t = ${(timeS * 1000).toFixed(1)} ms`, 12 * dpr, h - 12 * dpr);

      if (playing) {
        phaseRef.current = (phaseRef.current + speed / STEPS) % 1.0;
      }
    };

    const timer = window.setInterval(draw, 40);
    draw();
    return () => window.clearInterval(timer);
  }, [scenario, playing, speed, mode, mesh, externalMesh, hoselPoint, cogPoint, showCg]);

  return (
    <div className="space-y-2">
      <ClubCanvasPlaybackControls
        playing={playing}
        speed={speed}
        mode={mode}
        showCg={showCg}
        meshLoaded={mesh !== null}
        meshError={meshError}
        onPlayingChange={setPlaying}
        onSpeedChange={setSpeed}
        onModeChange={setMode}
        onShowCgChange={setShowCg}
        onStlChosen={onStlChosen}
        onProceduralHead={() => {
          setMesh(null);
          setMeshError(null);
        }}
      />
      <ClubCameraControls
        state={camera}
        activeViewId={canonicalOrientation ? camera.presetId : null}
        onView={(view: CameraViewId) => {
          setCanonicalOrientation(true);
          updateCamera((current) => applyCameraView(current, view));
        }}
        onFaceOnSide={(side: FaceOnSide) => {
          if (cameraRef.current.presetId === "camera.view.face_on") {
            setCanonicalOrientation(true);
          }
          updateCamera((current) => setFaceOnSide(current, side));
        }}
        onReset={() => {
          setCanonicalOrientation(true);
          updateCamera((current) => applyCameraView(current, "camera.view.isometric"));
        }}
        onAutoFit={() => updateCamera((current) => autoFitCamera(
          current,
          subjectRadiusRef.current,
          mode === VIEW_MODES[1] ? 0.42 : 0.24,
        ))}
        onTrackingChange={(enabled) => updateCamera((current) =>
          setCameraTracking(current, enabled, subjectRef.current))}
        onAutoFitFallbackChange={(enabled) => updateCamera((current) =>
          enforceTrackingClearance(
            setAutoFitFallback(current, enabled),
            subjectRadiusRef.current,
            mode === VIEW_MODES[1] ? 0.42 : 0.24,
          ))}
        onRecenter={() => updateCamera((current) =>
          recenterCamera(current, subjectRef.current))}
      />
      <canvas
        ref={canvasRef}
        width={840}
        height={571}
        className="w-full cursor-grab touch-none rounded-xl border border-slate-800/80 bg-slate-950/80 shadow-lg shadow-black/30 active:cursor-grabbing"
        role="img"
        aria-label="Animated 3D clubhead rotating under the scenario's angular velocity. Drag to orbit; scroll to zoom; Track Clubhead follows motion."
        tabIndex={0}
        data-camera-view={canonicalOrientation ? camera.presetId : "custom"}
        data-camera-yaw={camera.yawRad.toFixed(12)}
        data-camera-pitch={camera.pitchRad.toFixed(12)}
        data-camera-zoom={camera.zoom.toFixed(6)}
        data-camera-target={camera.targetM.map((value) => value.toFixed(12)).join(",")}
        data-camera-subject={subjectRef.current.map((value) => value.toFixed(12)).join(",")}
        data-camera-tracking-state={trackingStateId(camera)}
        data-camera-auto-fit-fallback={String(camera.autoFitFallbackEnabled)}
        data-camera-subject-fits={String(
          subjectRadiusRef.current * camera.zoom
            <= (mode === VIEW_MODES[1] ? 0.42 : 0.24)
              * (1 - AUTO_FIT_CLEARANCE_FRACTION) + 1e-12
        )}
        onPointerDown={(e) => {
          dragRef.current = { x: e.clientX, y: e.clientY };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!dragRef.current) return;
          yawRef.current -= (e.clientX - dragRef.current.x) * 0.008;
          pitchRef.current = Math.max(
            -1.4,
            Math.min(1.4, pitchRef.current + (e.clientY - dragRef.current.y) * 0.008),
          );
          dragRef.current = { x: e.clientX, y: e.clientY };
          setCanonicalOrientation(false);
          updateCamera((current) => applyManualCameraOverride({
              ...current,
              yawRad: yawRef.current,
              pitchRad: pitchRef.current,
            }));
        }}
        onPointerUp={(e) => {
          dragRef.current = null;
          e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerLeave={() => {
          dragRef.current = null;
        }}
        onWheel={(e) => {
          updateCamera((current) => withCameraZoom(
            current,
            current.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1),
          ));
        }}
      />
      <p className="text-xs text-slate-500">
        Drag to orbit (tracking pauses); scroll to zoom. Track Clubhead and
        Re-center keep the moving head in view without changing zoom.
      </p>
    </div>
  );
}
