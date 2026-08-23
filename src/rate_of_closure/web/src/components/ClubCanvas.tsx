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
 * Optional flat-shaded mode: an STL file input (client-side
 * FileReader, nothing uploaded) swaps the procedural wireframe for the
 * user's clubhead mesh, normalized onto the same envelope and rendered
 * as flat-shaded painter's-algorithm triangles — depth-sorted by
 * distance along the camera's forward axis, shaded by |normal · light|
 * with the same fixed world light as the desktop app.
 */

import {
  useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction,
} from "react";

import { solve, type ImpactScenario } from "../model/impact";
import {
  applyClubCameraAction,
  applyClubCameraDrag,
  cameraStatus,
  type ClubCamera,
} from "../model/clubCamera";
import {
  importedMeshSource,
  cleanSourceName,
  proceduralMeshSource,
  type ClubMeshSource,
} from "../model/clubMeshSource";
import { MAX_STL_BYTES } from "../model/mesh";
import { FIELD_GUIDANCE } from "../model/units";
import { useClubCanvasRenderer } from "./useClubCanvasRenderer";

export const VIEW_MODES = [
  "Head Fixed in Place",
  "Head Moving Through Space",
] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

export function ClubCanvas({
  scenario,
  source,
  onSourceChange,
  camera,
  onCameraChange,
}: {
  scenario: ImpactScenario;
  source: ClubMeshSource;
  onSourceChange: Dispatch<SetStateAction<ClubMeshSource>>;
  camera: ClubCamera;
  onCameraChange: Dispatch<SetStateAction<ClubCamera>>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef(0);
  const dragRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const readerRef = useRef<FileReader | null>(null);
  const importEpochRef = useRef(0);
  const [playing, setPlaying] = useState(
    () => !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
  );
  const [speed, setSpeed] = useState(1.0);
  const [mode, setMode] = useState<ViewMode>(VIEW_MODES[1]);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [loadingName, setLoadingName] = useState<string | null>(null);
  const [showCg, setShowCg] = useState(true);
  const result = useMemo(() => solve(scenario), [scenario]);

  useEffect(() => {
    const media = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!media) return undefined;
    const changed = () => { if (media.matches) setPlaying(false); };
    media.addEventListener?.("change", changed);
    return () => media.removeEventListener?.("change", changed);
  }, []);

  useEffect(() => {
    importEpochRef.current += 1;
    readerRef.current?.abort();
    readerRef.current = null;
    setLoadingName(null);
    setMeshError(null);
  }, [source]);

  useEffect(() => () => {
    importEpochRef.current += 1;
    readerRef.current?.abort();
  }, []);

  const handleRenderError = useCallback((message: string | null) => {
    setRenderError(message);
    if (message) setPlaying(false);
  }, []);

  const onStlChosen = (file: File | undefined) => {
    if (!file) return;
    readerRef.current?.abort();
    const epoch = importEpochRef.current + 1;
    importEpochRef.current = epoch;
    readerRef.current = null;
    setLoadingName(null);
    if (!file.name.toLowerCase().endsWith(".stl")) {
      setMeshError("STL load failed: mesh file must use the .stl suffix");
      return;
    }
    if (file.size > MAX_STL_BYTES) {
      setMeshError("STL load failed: STL must not exceed 2 MiB");
      return;
    }
    const sourceAtStart = source;
    const reader = new FileReader();
    readerRef.current = reader;
    setLoadingName(cleanSourceName(file.name));
    setMeshError(null);
    reader.onload = async () => {
      if (epoch !== importEpochRef.current || !(reader.result instanceof ArrayBuffer)) {
        if (epoch === importEpochRef.current) {
          setMeshError("STL load failed: file reader returned non-binary data");
        }
        return;
      }
      try {
        const candidate = await importedMeshSource(
          file.name, reader.result, sourceAtStart.generation + 1,
        );
        if (epoch !== importEpochRef.current) return;
        onSourceChange((current) => current === sourceAtStart ? candidate : current);
        setMeshError(null);
      } catch (err) {
        if (epoch === importEpochRef.current) {
          const detail = err instanceof Error ? err.message : String(err);
          setMeshError(`STL load failed: ${detail.slice(0, 512)}`);
        }
      } finally {
        if (epoch === importEpochRef.current) setLoadingName(null);
      }
    };
    reader.onerror = () => {
      if (epoch === importEpochRef.current) {
        setLoadingName(null);
        setMeshError("STL load failed: could not read the selected file");
      }
    };
    reader.onabort = () => {
      if (epoch === importEpochRef.current) setLoadingName(null);
    };
    reader.readAsArrayBuffer(file);
  };

  useClubCanvasRenderer({
    canvasRef, phaseRef, scenario, result, playing, speed, mode, source, camera, showCg,
    onRenderError: handleRenderError,
  });

  return (
    <div className="space-y-2">
      <div
        aria-label="Playback controls"
        className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/60 px-4 py-2.5 text-sm shadow-lg shadow-black/20 backdrop-blur"
      >
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
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
            onChange={(e) => setSpeed(Number(e.target.value))}
            aria-label="Playback speed multiplier"
          />
          <span className="w-8 text-slate-300">{speed.toFixed(1)}x</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-slate-400">Display</span>
          <select
            value={mode}
            title="Display mode: head fixed in place or moving through space"
            onChange={(e) => setMode(e.target.value as ViewMode)}
            className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            {VIEW_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
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
          onChange={(e) => {
            onStlChosen(e.target.files?.[0]);
            e.target.value = "";
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
        <label
          title={FIELD_GUIDANCE.showCgMarker}
          className="flex items-center gap-2 text-slate-300"
        >
          <input
            type="checkbox"
            checked={showCg}
            onChange={(e) => setShowCg(e.target.checked)}
            aria-label="Show reference marker"
          />
          Show reference marker
        </label>
        <button
          type="button"
          disabled={source.kind === "procedural"}
          onClick={() => {
            importEpochRef.current += 1;
            readerRef.current?.abort();
            onSourceChange(proceduralMeshSource(source.generation + 1));
            setMeshError(null);
          }}
          title="Return to the default wireframe head."
          className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors enabled:hover:border-sky-400 disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        >
          Procedural Head
        </button>
        {meshError && (
          <span role="alert" className="text-xs text-rose-400">
            {meshError}
          </span>
        )}
        {renderError && (
          <span role="alert" className="text-xs text-rose-400">
            {renderError}
          </span>
        )}
        {loadingName && (
          <span role="status" className="text-xs text-sky-300">
            Reading {loadingName.slice(0, 64)}; prior head and camera remain displayed.
          </span>
        )}
        <button type="button"
          title="Restore the canonical clubhead camera view"
          onClick={() => onCameraChange((prior) => applyClubCameraAction(prior, "home"))}
          className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400">
          Reset View
        </button>
      </div>
      <canvas
        ref={canvasRef}
        width={840}
        height={571}
        className="w-full cursor-grab touch-none rounded-xl border border-slate-800/80 bg-slate-950/80 shadow-lg shadow-black/30 active:cursor-grabbing focus-visible:outline focus-visible:outline-4 focus-visible:outline-sky-400"
        role="img"
        tabIndex={0}
        aria-label="Interactive 3D clubhead camera. Arrow keys orbit, plus and minus zoom, Home resets."
        onPointerDown={(e) => {
          if (!e.isPrimary || e.button !== 0 || dragRef.current) return;
          dragRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY };
          try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { dragRef.current = null; }
        }}
        onPointerMove={(e) => {
          if (!dragRef.current || dragRef.current.id !== e.pointerId) return;
          const deltaX = e.clientX - dragRef.current.x;
          const deltaY = e.clientY - dragRef.current.y;
          onCameraChange((prior) => applyClubCameraDrag(prior, deltaX, deltaY));
          dragRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY };
        }}
        onPointerUp={(e) => {
          if (!dragRef.current || dragRef.current.id !== e.pointerId) return;
          dragRef.current = null;
          if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
          }
        }}
        onPointerCancel={(e) => {
          if (dragRef.current?.id === e.pointerId) dragRef.current = null;
        }}
        onLostPointerCapture={(e) => {
          if (dragRef.current?.id === e.pointerId) dragRef.current = null;
        }}
        onWheel={(e) => {
          if (e.deltaY === 0) return;
          e.preventDefault();
          onCameraChange((prior) => applyClubCameraAction(
            prior, e.deltaY < 0 ? "zoom_in" : "zoom_out",
          ));
        }}
        onKeyDown={(event) => {
          const actions = new Map<string, Parameters<typeof applyClubCameraAction>[1]>([
            ["ArrowLeft", "left"], ["ArrowRight", "right"], ["ArrowUp", "up"],
            ["ArrowDown", "down"], ["+", "zoom_in"], ["=", "zoom_in"],
            ["-", "zoom_out"], ["Home", "home"],
          ]);
          const action = actions.get(event.key);
          if (action === undefined) return;
          event.preventDefault();
          onCameraChange((prior) => applyClubCameraAction(prior, action));
        }}
      />
      <p role="status" aria-live="polite" className="text-xs text-slate-400">
        {cameraStatus(camera, source.status)}
      </p>
      <p className="text-xs text-slate-500">
        Drag or arrow keys orbit; wheel or +/- zoom; Home resets. Imported STL axes
        are permuted by stable extent and one sign is adjusted only to preserve
        handedness; units and physical front/back registration are unknown.
      </p>
    </div>
  );
}
