import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";
import { FIELD_GUIDANCE } from "../model/units";

interface Props {
  tilts: { yaw: number; side: number; forward: number };
  enabled: boolean;
  onChange: (tilts: { yaw: number; side: number; forward: number }) => void;
}

export function PlaneTiltControls({ tilts, enabled, onChange }: Props) {
  const field = (
    label: string,
    value: number,
    guidanceKey: string,
    update: (value: number) => Props["tilts"],
  ) => (
    <label className="mb-2 block text-sm" title={FIELD_GUIDANCE[guidanceKey]}>
      <span className="mb-1 flex justify-between text-slate-300">
        <span className="flex items-center">{label}<FieldInfo label={label} guidance={FIELD_GUIDANCE[guidanceKey]} /></span>
        <span className="text-slate-500">deg</span>
      </span>
      <DecimalInput value={value} aria-label={`${label} deg`} title={FIELD_GUIDANCE[guidanceKey]}
        disabled={!enabled}
        onCommit={(next) => onChange(update(next))}
        className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-45" />
    </label>
  );
  return <fieldset
    disabled={!enabled}
    aria-describedby="plane-orientation-applicability"
    className="mb-3 rounded-lg border border-slate-700/80 bg-slate-950/40 p-3 disabled:opacity-70"
  >
    <legend className="px-1 text-sm font-semibold text-slate-200">Plane Orientation</legend>
    {field("Plane Yaw", tilts.yaw, "planeYawDeg", (yaw) => ({ ...tilts, yaw }))}
    {field("Plane Side Tilt", tilts.side, "planeSideTiltDeg", (side) => ({ ...tilts, side }))}
    {field("Plane Forward Tilt", tilts.forward, "planeForwardTiltDeg", (forward) => ({ ...tilts, forward }))}
    <p id="plane-orientation-applicability" className="text-xs leading-relaxed text-slate-400">
      {enabled
        ? "Applies to articulated pendulum swing sources."
        : "Not applicable to Manual Constant-Twist Delivery; manual attack angle and club path define its direction."}
    </p>
  </fieldset>;
}
