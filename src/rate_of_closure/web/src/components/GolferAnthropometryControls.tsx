import {
  DEFAULT_GOLFER_HEIGHT_M,
  DEFAULT_GOLFER_MASS_KG,
  STANDARD_ARM_LENGTH_FRACTION,
  STANDARD_ARM_MASS_FRACTION,
} from "../model/movementOptimizerSimulation";
import { type GolferAnthropometryTs } from "../model/simulationTypes";
import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";

interface Props {
  readonly enabled: boolean;
  readonly value: GolferAnthropometryTs;
  readonly onChange: (value: GolferAnthropometryTs) => void;
}

const INPUT_CLASS =
  "no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 " +
  "text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed " +
  "disabled:opacity-45";

const GUIDANCE = {
  heightM:
    "Golfer standing height in meters. Scales the upper-segment (lead arm) length via standard Dempster/Winter anthropometry (42% of height).",
  bodyMassKg:
    "Golfer total body mass in kilograms. Scales the upper-segment (lead arm) mass via standard Dempster/Winter anthropometry (10% of body mass).",
} as const;

export function GolferAnthropometryControls({ enabled, value, onChange }: Props) {
  const heightM = value.heightM ?? DEFAULT_GOLFER_HEIGHT_M;
  const bodyMassKg = value.bodyMassKg ?? DEFAULT_GOLFER_MASS_KG;
  const scaledArmLength = (heightM * STANDARD_ARM_LENGTH_FRACTION).toFixed(3);
  const scaledArmMass = (bodyMassKg * STANDARD_ARM_MASS_FRACTION).toFixed(2);

  return (
    <fieldset
      className="mb-4 rounded-lg border border-slate-700/80 bg-slate-950/40 p-3 disabled:opacity-70"
      aria-describedby="golfer-anthropometry-note"
    >
      <legend className="px-1 text-sm font-semibold text-slate-200">
        Golfer Anthropometry
      </legend>
      <p id="golfer-anthropometry-note" className="mb-3 text-xs leading-relaxed text-slate-400">
        Golfer physical dimensions determining linked-body segment lengths and inertias for reproducible swing deliveries.
      </p>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
        <label className="block text-sm" title={GUIDANCE.heightM}>
          <span className="mb-1 flex items-center justify-between gap-2 text-slate-300">
            <span className="flex items-center">
              Golfer Height
              <FieldInfo label="Golfer Height" guidance={GUIDANCE.heightM} />
            </span>
            <span className="text-slate-500">m</span>
          </span>
          <DecimalInput
            aria-label="Golfer Height"
            value={heightM}
            min={1.0}
            max={2.5}
            disabled={!enabled}
            title={GUIDANCE.heightM}
            onCommit={(next) =>
              onChange({
                ...value,
                heightM: next,
                armLengthM: undefined,
              })
            }
            className={INPUT_CLASS}
          />
        </label>

        <label className="block text-sm" title={GUIDANCE.bodyMassKg}>
          <span className="mb-1 flex items-center justify-between gap-2 text-slate-300">
            <span className="flex items-center">
              Golfer Body Mass
              <FieldInfo label="Golfer Body Mass" guidance={GUIDANCE.bodyMassKg} />
            </span>
            <span className="text-slate-500">kg</span>
          </span>
          <DecimalInput
            aria-label="Golfer Body Mass"
            value={bodyMassKg}
            min={30.0}
            max={250.0}
            disabled={!enabled}
            title={GUIDANCE.bodyMassKg}
            onCommit={(next) =>
              onChange({
                ...value,
                bodyMassKg: next,
                armMassKg: undefined,
              })
            }
            className={INPUT_CLASS}
          />
        </label>
      </div>
      <div className="mt-3 rounded border border-slate-800 bg-slate-900/50 px-2.5 py-2 text-xs text-slate-400">
        <span className="font-medium text-slate-300">Dempster/Winter Segments: </span>
        Lead Arm Length: <span className="font-semibold text-sky-400">{scaledArmLength} m</span>,
        Mass: <span className="font-semibold text-sky-400">{scaledArmMass} kg</span>
      </div>
      {!enabled && (
        <p className="mt-2 text-xs text-slate-500">
          Select Movement Optimizer to edit these values.
        </p>
      )}
    </fieldset>
  );
}
