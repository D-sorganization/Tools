/** Accessible canonical camera controls for the animated clubhead canvas. */

import {
  type CameraState,
  type CameraViewId,
  type FaceOnSide,
} from "../model/cameraPresets";

interface Props {
  state: CameraState;
  activeViewId: CameraViewId | null;
  onView: (view: CameraViewId) => void;
  onFaceOnSide: (side: FaceOnSide) => void;
  onReset: () => void;
  onAutoFit: () => void;
}

const VIEW_BUTTONS: ReadonlyArray<{
  id: CameraViewId;
  label: string;
  title: string;
}> = [
  {
    id: "camera.view.isometric",
    label: "Isometric",
    title: "Canonical engineering isometric view; preserves target and zoom.",
  },
  {
    id: "camera.view.face_on",
    label: "Face On",
    title: "Lateral view from the explicitly selected side of the target line.",
  },
  {
    id: "camera.view.down_the_line",
    label: "Down the Line",
    title: "Look from behind exactly along +x downrange with +y vertical.",
  },
  {
    id: "camera.view.overhead",
    label: "Overhead",
    title: "Look exactly down along -y with +x downrange toward screen-up.",
  },
];

const BUTTON_CLASS = "rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 aria-pressed:border-sky-400 aria-pressed:bg-sky-500/15";

export function ClubCameraControls({
  state,
  activeViewId,
  onView,
  onFaceOnSide,
  onReset,
  onAutoFit,
}: Props) {
  return (
    <div role="group" aria-label="Clubhead camera controls"
      className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800/80 bg-slate-900/60 px-3 py-2 text-xs shadow-lg shadow-black/20 backdrop-blur">
      {VIEW_BUTTONS.map(({ id, label, title }) => (
        <button key={id} type="button" title={title} data-camera-command={id}
          aria-pressed={activeViewId === id} onClick={() => onView(id)}
          className={BUTTON_CLASS}>
          {label}
        </button>
      ))}
      <label className="flex items-center gap-1 text-slate-300"
        title="Choose the physical viewing side; golfer handedness is never inferred.">
        Face-on side
        <select aria-label="Face-on camera side" value={state.faceOnSide}
          data-camera-control="camera.face_on_side"
          onChange={(event) => onFaceOnSide(event.target.value as FaceOnSide)}
          className="rounded border border-slate-700 bg-slate-900 px-1 py-1">
          <option value="right">Right of target</option>
          <option value="left">Left of target</option>
        </select>
      </label>
      <button type="button" data-camera-command="camera.reset_view"
        title="Restore canonical isometric orientation without changing target or zoom."
        onClick={onReset} className={BUTTON_CLASS}>
        Reset View
      </button>
      <button type="button" data-camera-command="camera.auto_fit"
        title="Fit the complete current clubhead and shaft with 16% clearance."
        onClick={onAutoFit} className={BUTTON_CLASS}>
        Auto Fit
      </button>
    </div>
  );
}
