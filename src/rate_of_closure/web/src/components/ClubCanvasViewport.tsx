import type { RefObject } from "react";

import type { Vec3 } from "./clubCanvasGeometry";
import { CameraControlBar } from "./CameraControlBar";
import { pointerCoordinates, type PointerCoordinates } from "./pointerCoordinates";
import {
  applyCameraPreset,
  applyManualOverride,
  recenterCamera,
  safeTrackingZoom,
  setFaceOnSide,
  setTrackingEnabled,
  withCameraZoom,
  withManualOrbit,
  type CameraState,
} from "../model/cameraCommands";

interface Props {
  readonly camera: CameraState;
  readonly canvasRef: RefObject<HTMLCanvasElement>;
  readonly dragRef: { current: PointerCoordinates | null };
  readonly subjectRef: { current: Vec3 };
  readonly clearanceRadiusM: number;
  readonly baseHalfExtentM: number;
  readonly updateCamera: (transform: (current: CameraState) => CameraState) => void;
}

/** Camera controls and interaction adapter for the moving clubhead viewport. */
export function ClubCanvasViewport(props: Props) {
  return <>
    <CameraControlBar state={props.camera} subjectLabel="Clubhead"
      onPreset={(preset) => props.updateCamera((current) => applyCameraPreset(current, preset))}
      onFaceOnSide={(side) => props.updateCamera((current) => setFaceOnSide(current, side))}
      onTracking={(enabled) => props.updateCamera((current) => setTrackingEnabled(
        current, enabled, props.subjectRef.current,
      ))}
      onAutoFit={(enabled) => props.updateCamera((current) => ({
        ...current,
        autoFitEnabled: enabled,
        zoom: enabled ? safeTrackingZoom(
          current.zoom, props.clearanceRadiusM, props.baseHalfExtentM,
        ) : current.zoom,
      }))}
      onRecenter={() => props.updateCamera((current) => recenterCamera(
        current, props.subjectRef.current,
      ))} />
    <canvas ref={props.canvasRef} width={840} height={571}
      className="w-full cursor-grab touch-none rounded-xl border border-slate-800/80 bg-slate-950/80 shadow-lg shadow-black/30 active:cursor-grabbing"
      role="img"
      aria-label="Animated 3D clubhead rotating under the scenario's angular velocity. Drag to orbit; scroll to zoom."
      onPointerDown={(event) => {
        props.dragRef.current = pointerCoordinates(event.nativeEvent);
        event.currentTarget.setPointerCapture?.(event.pointerId);
        props.updateCamera(applyManualOverride);
      }}
      onPointerMove={(event) => {
        const drag = props.dragRef.current;
        if (!drag) return;
        const pointer = pointerCoordinates(event.nativeEvent);
        props.updateCamera((current) => withManualOrbit(
          current,
          current.yawRad - (pointer.x - drag.x) * 0.008,
          Math.max(-1.4, Math.min(1.4, current.pitchRad + (pointer.y - drag.y) * 0.008)),
        ));
        props.dragRef.current = pointer;
      }}
      onPointerUp={(event) => {
        props.dragRef.current = null;
        event.currentTarget.releasePointerCapture?.(event.pointerId);
      }}
      onPointerLeave={() => { props.dragRef.current = null; }}
      onWheel={(event) => {
        const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
        props.updateCamera((current) => withCameraZoom(
          current,
          current.autoFitEnabled
            ? safeTrackingZoom(
              current.zoom * factor, props.clearanceRadiusM, props.baseHalfExtentM,
            )
            : current.zoom * factor,
        ));
      }} />
    <p className="text-xs text-slate-500">Drag the view to orbit; scroll to zoom.</p>
  </>;
}
