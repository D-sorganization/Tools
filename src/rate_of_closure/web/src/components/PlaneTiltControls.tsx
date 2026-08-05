import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";
import { FIELD_GUIDANCE } from "../model/units";

interface Props {
  tilts: { yaw: number; side: number; forward: number };
  onChange: (tilts: { yaw: number; side: number; forward: number }) => void;
}

export function PlaneTiltControls({ tilts, onChange }: Props) {
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
        onCommit={(next) => onChange(update(next))}
        className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none" />
    </label>
  );
  return <>{field("Plane Yaw", tilts.yaw, "planeYawDeg", (yaw) => ({ ...tilts, yaw }))}
    {field("Plane Side Tilt", tilts.side, "planeSideTiltDeg", (side) => ({ ...tilts, side }))}
    {field("Plane Forward Tilt", tilts.forward, "planeForwardTiltDeg", (forward) => ({ ...tilts, forward }))}</>;
}
