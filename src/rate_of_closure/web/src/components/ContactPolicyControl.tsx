import { type ContactMode } from "../model/contact";

export function ContactPolicyControl({
  value,
  onChange,
}: {
  value: ContactMode;
  onChange: (mode: ContactMode) => void;
}) {
  return (
    <>
      <label
        className="mb-3 block text-sm"
        title="Choose whether to inspect delivery at a selected instant or require the unmodified clubhead path to reach the fixed ball."
      >
        <span className="mb-1 block text-slate-300">Contact Policy</span>
        <select
          value={value}
          aria-label="Contact Policy"
          onChange={(event) => onChange(event.target.value as ContactMode)}
          className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          <option value="delivery_inspection">
            Delivery Inspection (Align at τ)
          </option>
          <option value="fixed_ball_contact">Fixed Ball (Hit or Miss)</option>
        </select>
      </label>
      <p className="mb-3 rounded-lg border border-slate-700/80 bg-slate-950/50 px-3 py-2 text-xs leading-relaxed text-slate-400">
        {value === "delivery_inspection"
          ? "Moves the complete swing so the selected clubhead reference point meets the ball. Use this to inspect delivery; it is not collision detection."
          : "Keeps the ball and swing fixed. A sampled reference-point/sphere check reports a hit or miss without inventing impact or flight data."}
      </p>
    </>
  );
}
