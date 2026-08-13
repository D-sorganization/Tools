import { DecimalInput } from "./DecimalInput";
import {
  MAX_RUNS,
  LOCALIZED_TORQUE_DURATION_S,
  keysForMode,
  localizedTorqueJointId,
  variableDef,
  variableLabel,
  type Distribution,
  type NoiseSpecTs,
  type VariationMode,
  type VariationPlanTs,
} from "../model/variation";
import type { VariationAnalysisExecution } from "../model/variationAnalysisPolicy";
import {
  BUTTON_CLASS,
  INPUT_CLASS,
  MODE_LABELS,
  PANEL_CLASS,
  defaultSpec,
} from "./variationUi";

const DISTRIBUTIONS: Distribution[] = ["normal", "uniform", "triangular"];

interface VariationSetupProps {
  plan: VariationPlanTs;
  onPlanChange: (plan: VariationPlanTs) => void;
  analysisExecution: VariationAnalysisExecution;
  onAnalysisExecutionChange: (value: VariationAnalysisExecution) => void;
  onConfigurationChange: () => void;
}

export function VariationSetup({
  plan,
  onPlanChange,
  analysisExecution,
  onAnalysisExecutionChange,
  onConfigurationChange,
}: VariationSetupProps): JSX.Element {
  const updatePlan = (updates: Partial<VariationPlanTs>) => {
    onPlanChange({ ...plan, ...updates });
    onConfigurationChange();
  };
  const setSpec = (index: number, updates: Partial<NoiseSpecTs>) => {
    updatePlan({
      noise: plan.noise.map((row, rowIndex) =>
        rowIndex === index ? { ...row, ...updates } : row,
      ),
    });
  };
  const changeMode = (mode: VariationMode) => {
    updatePlan({ mode, baseVariables: {}, noise: [defaultSpec(mode, new Set(), plan.ballSetup)], groups: [] });
  };
  const changeVariable = (index: number, variableKey: string) => {
    const previous = plan.noise[index];
    const previousId = previous.specId ?? previous.variableKey;
    const usedDefaultId = previous.specId === undefined || previous.specId === previous.variableKey;
    const nextId = usedDefaultId ? variableKey : previousId;
    const covarianceGrouped = (plan.groups ?? []).some(
      (group) => group.matrixKind === "covariance" && group.specIds.includes(previousId),
    );
    const localizedJoint = localizedTorqueJointId(variableKey);
    updatePlan({
      noise: plan.noise.map((spec, rowIndex) =>
        rowIndex === index
          ? {
              ...spec,
              variableKey,
              specId: nextId,
              scale: covarianceGrouped
                ? spec.scale
                : variableDef(variableKey)?.typicalScale ?? spec.scale,
              timeWindowS: localizedJoint === null ? null : [0, 0.1],
              pointIds: localizedJoint === null ? [] : [localizedJoint],
            }
          : spec,
      ),
      groups: (plan.groups ?? []).map((group) => ({
        ...group,
        specIds: group.specIds.map((specId) =>
          specId === previousId ? nextId : specId,
        ),
      })),
    });
  };

  const locusCount = plan.noise.filter(
    (spec) =>
      (spec.timeWindowS !== null && spec.timeWindowS !== undefined) ||
      (spec.pointIds?.length ?? 0) > 0,
  ).length;
  const contextualCount = plan.noise.filter(
    (spec) => localizedTorqueJointId(spec.variableKey) !== null,
  ).length;
  const retainedLocusCount = locusCount - contextualCount;
  const usedVariables = new Set(plan.noise.map((spec) => spec.variableKey));
  const legalKeys = keysForMode(plan.mode, plan.ballSetup);
  const canAdd = usedVariables.size < legalKeys.length;

  return (
    <>
      <div className={PANEL_CLASS}>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Study Setup
        </h2>
        <label className="mb-3 block text-sm" title="Select the pipeline slice each run exercises.">
          <span className="mb-1 block text-slate-300">Pipeline</span>
          <select
            value={plan.mode}
            onChange={(event) => changeMode(event.target.value as VariationMode)}
            className={INPUT_CLASS}
          >
            {(Object.keys(MODE_LABELS) as VariationMode[]).map((mode) => (
              <option key={mode} value={mode}>{MODE_LABELS[mode]}</option>
            ))}
          </select>
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-sm" title={`Browser-capped at ${MAX_RUNS} runs.`}>
            <span className="mb-1 block text-slate-300">Runs (≤ {MAX_RUNS})</span>
            <DecimalInput
              min={2}
              max={MAX_RUNS}
              value={plan.nRuns}
              aria-label="Runs"
              onCommit={(value) => updatePlan({ nRuns: Math.round(value) })}
              className={INPUT_CLASS}
            />
          </label>
          <label className="block text-sm" title="Master RNG seed for reproducible samples.">
            <span className="mb-1 block text-slate-300">Seed</span>
            <DecimalInput
              min={0}
              value={plan.seed}
              aria-label="Seed"
              onCommit={(value) => updatePlan({ seed: Math.round(value) })}
              className={INPUT_CLASS}
            />
          </label>
        </div>
        <label className="mt-3 block text-sm" title="Choose which analyses execute when Run is pressed.">
          <span className="mb-1 block text-slate-300">Analysis Execution</span>
          <select
            aria-label="Analysis execution"
            value={analysisExecution}
            onChange={(event) =>
              onAnalysisExecutionChange(event.target.value as VariationAnalysisExecution)
            }
            className={INPUT_CLASS}
          >
            <option value="all_together">All Enabled Together</option>
            <option value="individual">Each Enabled Individually (OAT)</option>
            <option value="both">Both</option>
          </select>
        </label>
        <p className="mt-2 text-xs text-slate-500">
          This execution choice is UI policy and is not stored in the physical variation plan.
        </p>
      </div>

      <div className={PANEL_CLASS}>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Varied Variables (Noise)
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          Grouped correlation or covariance plans sample normal inputs jointly; correlation
          controls co-movement while each noise scale retains its marginal standard deviation.
        </p>
        <p className={`mb-3 text-xs ${plan.ballSetup?.supportMode === "tee" ? "text-sky-300" : "text-amber-300"}`}>
          {plan.ballSetup?.supportMode === "tee"
            ? "Tee Height is available as a numeric variation input for the active Tee setup."
            : "Tee Height is excluded in Ground mode. Select Tee in Simulation to enable it."}
        </p>
        {(plan.groups?.length ?? 0) > 0 && (
          <p className="mb-3 text-xs text-sky-300">
            This plan contains {plan.groups!.length} grouped correlation or covariance definition(s).
          </p>
        )}
        <p className={`mb-3 text-xs ${contextualCount > 0 ? "text-amber-300" : "text-slate-500"}`}>
          Localized torque loci execute additively on the fixed 1.5 s TypeScript-reference
          double-pendulum swing at every RK4 substep.
        </p>
        {retainedLocusCount > 0 && (
          <p className="mb-3 text-xs text-amber-300">
            Imported locus metadata is retained losslessly but cannot yet execute through the
            scalar browser path.
          </p>
        )}
        {plan.noise.map((spec, index) => {
          const definition = variableDef(spec.variableKey);
          const specId = spec.specId ?? spec.variableKey;
          const grouped = (plan.groups ?? []).some((group) => group.specIds.includes(specId));
          const covarianceGrouped = (plan.groups ?? []).some(
            (group) => group.matrixKind === "covariance" && group.specIds.includes(specId),
          );
          const localizedJoint = localizedTorqueJointId(spec.variableKey);
          const localizedError = localizedLocusError(spec);
          return (
            <div
              key={spec.specId ?? `${spec.variableKey}-${index}`}
              className="mb-3 rounded-lg border border-slate-800 bg-slate-950/40 p-3"
            >
              <div className="mb-2 flex gap-2">
                <select
                  aria-label={`Variable ${index + 1}`}
                  value={spec.variableKey}
                  onChange={(event) => changeVariable(index, event.target.value)}
                  title={`${spec.variableKey} — ${definition?.guidance ?? ""}`}
                  className={`${INPUT_CLASS} flex-1`}
                >
                  {legalKeys.map((key) => (
                    <option key={key} value={key}>{variableLabel(key)}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => updatePlan({
                    noise: plan.noise.length > 1
                      ? plan.noise.filter((_row, rowIndex) => rowIndex !== index)
                      : plan.noise,
                    groups: [],
                  })}
                  title="Remove this noise row; correlation groups are cleared."
                  className={BUTTON_CLASS}
                >
                  ✕
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                <select
                  aria-label={`${variableLabel(spec.variableKey)} distribution`}
                  disabled={grouped}
                  value={spec.distribution}
                  onChange={(event) =>
                    setSpec(index, { distribution: event.target.value as Distribution })
                  }
                  title={grouped
                    ? "Grouped correlation and covariance specs must use a normal distribution."
                    : "Normal uses scale as standard deviation; uniform and triangular use half-width."}
                  className={INPUT_CLASS}
                >
                  {DISTRIBUTIONS.map((distribution) => (
                    <option key={distribution} value={distribution}>{distribution}</option>
                  ))}
                </select>
                <DecimalInput
                  step="any"
                  disabled={covarianceGrouped}
                  value={spec.scale}
                  aria-label={`${variableLabel(spec.variableKey)} noise scale`}
                  onCommit={(value) => setSpec(index, { scale: value })}
                  title={covarianceGrouped
                    ? "Covariance-group scale is fixed by the covariance matrix diagonal."
                    : `Noise scale [${definition?.unit ?? ""}]. ${definition?.guidance ?? ""}`}
                  className={INPUT_CLASS}
                />
                {(["lower", "upper"] as const).map((bound) => (
                  <input
                    key={bound}
                    aria-label={`${variableLabel(spec.variableKey)} ${bound} bound`}
                    type="number"
                    step="any"
                    placeholder={`${bound === "lower" ? "min" : "max"} (opt.)`}
                    value={spec[bound] ?? ""}
                    onChange={(event) =>
                      setSpec(index, {
                        [bound]: event.target.value === "" ? null : Number(event.target.value),
                      })
                    }
                    title={`Optional absolute ${bound} clipping bound.`}
                    className={INPUT_CLASS}
                  />
                ))}
              </div>
              {localizedJoint !== null && (
                <div className="mt-2 grid grid-cols-1 gap-2 text-sm sm:grid-cols-3">
                  <DecimalInput
                    min={0}
                    max={LOCALIZED_TORQUE_DURATION_S}
                    step="any"
                    value={spec.timeWindowS?.[0] ?? 0}
                    aria-label={`${variableLabel(spec.variableKey)} window start`}
                    onCommit={(value) => setSpec(index, {
                      timeWindowS: [value, spec.timeWindowS?.[1] ?? 0.1],
                    })}
                    title="Inclusive start [s] of the required half-open [start, end) torque window."
                    className={INPUT_CLASS}
                  />
                  <DecimalInput
                    min={0}
                    max={LOCALIZED_TORQUE_DURATION_S}
                    step="any"
                    value={spec.timeWindowS?.[1] ?? 0.1}
                    aria-label={`${variableLabel(spec.variableKey)} window end`}
                    onCommit={(value) => setSpec(index, {
                      timeWindowS: [spec.timeWindowS?.[0] ?? 0, value],
                    })}
                    title="Exclusive end [s] of the required half-open [start, end) torque window."
                    className={INPUT_CLASS}
                  />
                  <select
                    aria-label={`${variableLabel(spec.variableKey)} topological joint`}
                    value={localizedJoint}
                    disabled
                    title="Stable topological joint.* torque ID; it is distinct from spatial swing.* trace point IDs."
                    className={INPUT_CLASS}
                  >
                    <option value={localizedJoint}>{localizedJoint}</option>
                  </select>
                  {localizedError !== null && (
                    <p role="alert" className="text-xs text-amber-300 sm:col-span-3">
                      {localizedError}
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
        <button
          type="button"
          disabled={!canAdd}
          onClick={() => updatePlan({
            noise: [...plan.noise, defaultSpec(plan.mode, usedVariables, plan.ballSetup)],
          })}
          title="Add another unique varied variable."
          className={BUTTON_CLASS}
        >
          Add Variable
        </button>
      </div>
    </>
  );
}

const localizedLocusError = (spec: NoiseSpecTs): string | null => {
  if (localizedTorqueJointId(spec.variableKey) === null) return null;
  const window = spec.timeWindowS;
  if (window === undefined || window === null) {
    return "A finite half-open torque time window is required.";
  }
  if (!(0 <= window[0] && window[0] < window[1] && window[1] <= LOCALIZED_TORQUE_DURATION_S)) {
    return `Torque window must satisfy 0 ≤ start < end ≤ ${LOCALIZED_TORQUE_DURATION_S} s.`;
  }
  return null;
};
