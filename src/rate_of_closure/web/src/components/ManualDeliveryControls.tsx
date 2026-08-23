import {
  MANUAL_DELIVERY_BOUNDS,
  type ManualDelivery,
  type ShaftAxisDatum,
} from "../model/manualDelivery";
import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";

interface Props {
  readonly enabled: boolean;
  readonly value: ManualDelivery;
  readonly onChange: (value: ManualDelivery) => void;
}

const INPUT_CLASS =
  "no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 " +
  "text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed " +
  "disabled:opacity-45";

const GUIDANCE = {
  manualAttackAngleDeg:
    "App frame x = target, y = up, z = right. Positive is upward; negative descends toward the ground.",
  manualClubPathDeg:
    "Horizontal reference-point direction relative to the target line. Positive travels right of target; negative travels left.",
  manualForwardShaftLeanDeg:
    "Positive leans the shaft and head targetward at impact. The rigid pose uses Rz(-lean) and reduces delivered dynamic loft by the same angle.",
  shaftAxisDatum:
    "Tracked Reference preserves the legacy shaft line through the tracked head point. Generated Club Hosel uses the selected generic head profile's hosel anchor.",
} as const;

export function ManualDeliveryControls({ enabled, value, onChange }: Props) {
  const field = (
    key: keyof typeof MANUAL_DELIVERY_BOUNDS,
    label: string,
  ) => {
    const [minimum, maximum] = MANUAL_DELIVERY_BOUNDS[key];
    const guidance = GUIDANCE[key];
    return (
      <label className="block text-sm" title={guidance}>
        <span className="mb-1 flex items-center justify-between gap-2 text-slate-300">
          <span className="flex items-center">
            {label}
            <FieldInfo label={label} guidance={guidance} />
          </span>
          <span className="text-slate-500">deg</span>
        </span>
        <DecimalInput
          aria-label={label}
          value={value[key]}
          min={minimum}
          max={maximum}
          disabled={!enabled}
          title={guidance}
          onCommit={(next) => onChange({ ...value, [key]: next })}
          className={INPUT_CLASS}
        />
      </label>
    );
  };

  return (
    <fieldset
      className="mb-4 rounded-lg border border-slate-700/80 bg-slate-950/40 p-3 disabled:opacity-70"
      aria-describedby="manual-delivery-frame-note"
    >
      <legend className="px-1 text-sm font-semibold text-slate-200">
        Manual Delivery
      </legend>
      <p id="manual-delivery-frame-note" className="mb-3 text-xs leading-relaxed text-slate-400">
        Reference-point delivery in the target/ground frame. These controls apply only to Manual Constant-Twist Delivery.
      </p>
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
        {field("manualAttackAngleDeg", "Manual Attack Angle")}
        {field("manualClubPathDeg", "Manual Club Path")}
        {field("manualForwardShaftLeanDeg", "Manual Forward Shaft Lean")}
      </div>
      <label className="mt-3 block text-sm" title={GUIDANCE.shaftAxisDatum}>
        <span className="mb-1 flex items-center text-slate-300">
          Shaft Axis Datum
          <FieldInfo label="Shaft Axis Datum" guidance={GUIDANCE.shaftAxisDatum} />
        </span>
        <select
          aria-label="Shaft Axis Datum"
          value={value.shaftAxisDatum}
          disabled={!enabled}
          title={GUIDANCE.shaftAxisDatum}
          onChange={(event) => onChange({
            ...value,
            shaftAxisDatum: event.target.value as ShaftAxisDatum,
          })}
          className={INPUT_CLASS}
        >
          <option value="tracked_reference">Tracked Reference (Legacy)</option>
          <option value="generated_hosel">Generated Club Hosel</option>
        </select>
      </label>
      {!enabled && (
        <p className="mt-2 text-xs text-slate-500">
          Select Manual Constant-Twist Delivery to edit these values.
        </p>
      )}
    </fieldset>
  );
}
