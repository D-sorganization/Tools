import { useState } from "react";

import {
  SUGGESTED_MAX_TEE_HEIGHT_M,
  DRIVER_TEE_HEIGHT_M,
  resolveBallSetup,
  type BallSetup,
  type SupportMode,
} from "../model/ballSetup";
import { DecimalInput } from "./DecimalInput";

interface Props {
  setup: BallSetup;
  userOverridden: boolean;
  onChange: (setup: BallSetup) => void;
  onUseClubDefault: () => void;
}

const formatMm = (value: number) => String(Number((value * 1000).toFixed(3)));

export function BallSetupControl({
  setup,
  userOverridden,
  onChange,
  onUseClubDefault,
}: Props) {
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [lastTeeHeightM, setLastTeeHeightM] = useState(
    setup.supportMode === "tee" ? setup.teeHeightM : DRIVER_TEE_HEIGHT_M,
  );

  const selectMode = (supportMode: SupportMode) => {
    setError(null);
    if (setup.supportMode === "tee") setLastTeeHeightM(setup.teeHeightM);
    onChange(resolveBallSetup({
      supportMode,
      teeHeightM: supportMode === "tee" ? lastTeeHeightM : 0,
    }));
  };

  const commitHeightMm = (heightMm: number) => {
    try {
      const next = resolveBallSetup({ supportMode: "tee", teeHeightM: heightMm / 1000 });
      setError(null);
      setLastTeeHeightM(next.teeHeightM);
      onChange(next);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setRevision((value) => value + 1);
    }
  };

  return (
    <fieldset className="mb-4 rounded-lg border border-slate-700/80 bg-slate-950/40 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <legend className="text-sm font-semibold text-slate-200">Ball Support</legend>
        <span className="text-xs text-sky-300">
          {userOverridden ? "Explicit Override" : "Club Default"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label="Ball Support Mode">
        {(["ground", "tee"] as const).map((mode) => (
          <label
            key={mode}
            className={`cursor-pointer rounded-lg border px-3 py-2 text-center text-sm transition-colors ${
              setup.supportMode === mode
                ? "border-sky-400 bg-sky-500/15 text-sky-200"
                : "border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500"
            }`}
          >
            <input
              type="radio"
              name="ball-support-mode"
              value={mode}
              checked={setup.supportMode === mode}
              title={`Use ${mode === "ground" ? "Ground" : "Tee"} ball support`}
              onChange={() => selectMode(mode)}
              className="sr-only"
            />
            {mode === "ground" ? "Ground" : "Tee"}
          </label>
        ))}
      </div>
      <label className="mt-3 block text-sm">
        <span className="mb-1 flex justify-between text-slate-300">
          <span>Tee Height</span>
          <span className="text-slate-500">mm</span>
        </span>
        <DecimalInput
          key={revision}
          aria-label="Tee Height"
          value={setup.teeHeightM}
          format={formatMm}
          disabled={setup.supportMode === "ground"}
          onCommit={commitHeightMm}
          className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 disabled:cursor-not-allowed disabled:opacity-45"
        />
      </label>
      <p className="mt-2 text-xs text-slate-500">
        {setup.supportMode === "ground"
          ? "Ground mode places the ball on the ground; effective tee height is 0 mm."
          : `Clearance from the ground plane to the bottom of the ball. Common range: 0–${SUGGESTED_MAX_TEE_HEIGHT_M * 1000} mm; larger finite values are accepted.`}
      </p>
      {error && <p role="alert" className="mt-2 text-xs text-rose-300">{error}</p>}
      {userOverridden && (
        <button
          type="button"
          onClick={onUseClubDefault}
          className="mt-2 rounded border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:border-slate-500"
        >
          Use Club Default
        </button>
      )}
    </fieldset>
  );
}
