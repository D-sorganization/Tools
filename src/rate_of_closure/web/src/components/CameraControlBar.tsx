/** Shared accessible controls for an isolated 3D viewport camera. */

import {
  type CameraState,
  type CameraViewId,
  type FaceOnSide,
} from "../model/cameraCommands";

interface Props {
  state: CameraState;
  subjectLabel: "Clubhead" | "Ball" | "Impact";
  onPreset: (preset: CameraViewId) => void;
  onFaceOnSide: (side: FaceOnSide) => void;
  onTracking: (enabled: boolean) => void;
  onAutoFit: (enabled: boolean) => void;
  onRecenter: () => void;
  trackingAvailable?: boolean;
}

const VIEW_BUTTONS: ReadonlyArray<{
  id: CameraViewId;
  label: string;
  title: string;
}> = [
  {
    id: "camera.view.face_on",
    label: "Face On",
    title: "App frame: lateral view toward the target line from the selected explicit side.",
  },
  {
    id: "camera.view.down_the_line",
    label: "Down the Line",
    title: "App frame: look from behind along +x downrange with +y vertical.",
  },
  {
    id: "camera.view.overhead",
    label: "Overhead",
    title: "App frame: look down along -y with +x downrange toward screen-up.",
  },
  {
    id: "camera.view.isometric",
    label: "Reset View",
    title: "App frame: restore the canonical isometric orientation without changing zoom or focus.",
  },
];

export function CameraControlBar({
  state,
  subjectLabel,
  onPreset,
  onFaceOnSide,
  onTracking,
  onAutoFit,
  onRecenter,
  trackingAvailable = true,
}: Props) {
  const status = state.trackingSuspended
    ? "Tracking suspended by manual orbit"
    : state.trackingEnabled ? `Tracking ${subjectLabel}` : "Tracking off";
  return (
    <div className="flex flex-wrap items-center gap-2 rounded border border-slate-700/80 bg-slate-900/50 px-2 py-1.5 text-xs"
      aria-label={`${subjectLabel} camera controls`}>
      {VIEW_BUTTONS.map(({ id, label, title }) => (
        <button key={id} type="button" title={title} data-camera-command={id}
          aria-pressed={state.presetId === id} onClick={() => onPreset(id)}
          className="rounded border border-slate-600 px-2 py-1 hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 aria-pressed:border-sky-400 aria-pressed:bg-sky-500/15">
          {label}
        </button>
      ))}
      <label className="flex items-center gap-1 text-slate-300"
        title="Choose the physical side of the target line used by Face On; no handedness is inferred.">
        Face-on side
        <select aria-label="Face-on camera side" value={state.faceOnSide}
          onChange={(event) => onFaceOnSide(event.target.value as FaceOnSide)}
          className="rounded border border-slate-700 bg-slate-900 px-1 py-1">
          <option value="right">Right of target</option>
          <option value="left">Left of target</option>
        </select>
      </label>
      <label className="flex items-center gap-1 text-slate-300"
        title={`Follow the ${subjectLabel.toLowerCase()} while preserving safe zoom.`}>
        <input type="checkbox" checked={state.trackingEnabled} disabled={!trackingAvailable}
          data-camera-command="camera.track_subject" aria-label={`Track ${subjectLabel}`}
          onChange={(event) => onTracking(event.target.checked)} />
        Track {subjectLabel}
      </label>
      <label className="flex items-center gap-1 text-slate-300"
        title="Opt in to reducing unsafe zoom only when the tracked subject would violate the 16% clearance.">
        <input type="checkbox" checked={state.autoFitEnabled}
          data-camera-command="camera.auto_fit" aria-label="Auto Fit camera"
          onChange={(event) => onAutoFit(event.target.checked)} />
        Auto Fit
      </label>
      <button type="button" data-camera-command="camera.recenter"
        aria-label={`Re-center ${subjectLabel}`} onClick={onRecenter}
        title={`Center the camera on the current ${subjectLabel.toLowerCase()} and resume tracking.`}
        className="rounded border border-slate-600 px-2 py-1 hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400">
        Re-center
      </button>
      <span role="status" aria-label="Camera tracking state"
        className={state.trackingSuspended ? "text-amber-300" : "text-slate-400"}>
        {status}
      </span>
    </div>
  );
}
