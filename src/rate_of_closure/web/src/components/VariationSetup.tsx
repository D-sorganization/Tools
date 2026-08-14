import { DecimalInput } from "./DecimalInput";
import {
  MAX_RUNS,
  outputsForMode,
  keysForMode,
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
  selectedOutputMetrics: readonly string[];
  onSelectedOutputMetricsChange: (metrics: readonly string[]) => void;
  chipStudyEnabled: boolean;
  onChipStudyEnabledChange: (value: boolean) => void;
  chipTargetCarryYd: number;
  onChipTargetCarryYdChange: (value: number) => void;
  onConfigurationChange: () => void;
}

export function VariationSetup({
  plan,
  onPlanChange,
  analysisExecution,
  onAnalysisExecutionChange,
  selectedOutputMetrics,
  onSelectedOutputMetricsChange,
  chipStudyEnabled,
  onChipStudyEnabledChange,
  chipTargetCarryYd,
  onChipTargetCarryYdChange,
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
    updatePlan({
      mode,
      baseVariables: {},
      noise: [defaultSpec(mode, new Set(), plan.ballSetup)],
      groups: [],
    });
  };
  const changeVariable = (index: number, variableKey: string) => {
    const previous = plan.noise[index];
    const previousId = previous.specId ?? previous.variableKey;
    const usedDefaultId =
      previous.specId === undefined || previous.specId === previous.variableKey;
    const nextId = usedDefaultId ? variableKey : previousId;
    const covarianceGrouped = (plan.groups ?? []).some(
      (group) =>
        group.matrixKind === "covariance" && group.specIds.includes(previousId),
    );
    updatePlan({
      noise: plan.noise.map((spec, rowIndex) =>
        rowIndex === index
          ? {
              ...spec,
              variableKey,
              specId: nextId,
              scale: covarianceGrouped
                ? spec.scale
                : (variableDef(variableKey)?.typicalScale ?? spec.scale),
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

  const localizedCount = plan.noise.filter(
    (spec) =>
      (spec.timeWindowS !== null && spec.timeWindowS !== undefined) ||
      (spec.pointIds?.length ?? 0) > 0,
  ).length;
  const usedVariables = new Set(plan.noise.map((spec) => spec.variableKey));
  const legalKeys = keysForMode(plan.mode, plan.ballSetup);
  const canAdd = usedVariables.size < legalKeys.length;

  return (
    <>
      <div className={PANEL_CLASS}>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Study Setup
        </h2>
        <label
          className="mb-3 block text-sm"
          title="Select the pipeline slice each run exercises."
        >
          <span className="mb-1 block text-slate-300">Pipeline</span>
          <select
            value={plan.mode}
            onChange={(event) =>
              changeMode(event.target.value as VariationMode)
            }
            className={INPUT_CLASS}
          >
            {(Object.keys(MODE_LABELS) as VariationMode[]).map((mode) => (
              <option key={mode} value={mode}>
                {MODE_LABELS[mode]}
              </option>
            ))}
          </select>
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label
            className="block text-sm"
            title={`Browser-capped at ${MAX_RUNS} runs.`}
          >
            <span className="mb-1 block text-slate-300">
              Runs (≤ {MAX_RUNS})
            </span>
            <DecimalInput
              min={2}
              max={MAX_RUNS}
              value={plan.nRuns}
              aria-label="Runs"
              onCommit={(value) => updatePlan({ nRuns: Math.round(value) })}
              className={INPUT_CLASS}
            />
          </label>
          <label
            className="block text-sm"
            title="Master RNG seed for reproducible samples."
          >
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
        <label
          className="mt-3 block text-sm"
          title="Choose which analyses execute when Run is pressed."
        >
          <span className="mb-1 block text-slate-300">Analysis Execution</span>
          <select
            aria-label="Analysis execution"
            value={analysisExecution}
            onChange={(event) =>
              onAnalysisExecutionChange(
                event.target.value as VariationAnalysisExecution,
              )
            }
            className={INPUT_CLASS}
          >
            <option value="all_together">All Enabled Together</option>
            <option value="individual">Each Enabled Individually (OAT)</option>
            <option value="both">Both</option>
          </select>
        </label>
        <p className="mt-2 text-xs text-slate-500">
          Execution and output focus are saved workspace policy; the physical
          plan remains reusable.
        </p>
        <fieldset className="mt-3" aria-label="Selected output metrics">
          <legend className="mb-2 text-sm text-slate-300">
            Saved Output Focus
          </legend>
          <div className="grid max-h-36 grid-cols-2 gap-1 overflow-y-auto rounded border border-slate-800 p-2">
            {outputsForMode(plan.mode).map((metric) => {
              const checked = selectedOutputMetrics.includes(metric);
              return (
                <label
                  key={metric}
                  className="flex items-center gap-2 text-xs text-slate-300"
                >
                  <input
                    type="checkbox"
                    aria-label={metric}
                    checked={checked}
                    onChange={() => {
                      const next = checked
                        ? selectedOutputMetrics.filter(
                            (item) => item !== metric,
                          )
                        : [...selectedOutputMetrics, metric];
                      if (next.length > 0) onSelectedOutputMetricsChange(next);
                    }}
                  />
                  <span>{metric}</span>
                </label>
              );
            })}
          </div>
        </fieldset>
        {plan.mode === "swing" && (
          <div className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-950/20 p-3">
              className="flex cursor-pointer items-start gap-3"
              title="Use a representative 56-degree wedge on the ground and retain every contact cohort for the declared chip objective."
                aria-label="Analyze wedge chip forgiveness"
                checked={chipStudyEnabled}
                onChange={(event) => {
                  onChipStudyEnabledChange(event.target.checked);
                  onConfigurationChange();
                className="mt-1"
              <span>
                <span className="block text-sm font-semibold text-emerald-200">
                  Analyze Wedge Chip Forgiveness
                </span>
                <span className="mt-1 block text-xs leading-5 text-slate-400">
                  Ground-mode 56° wedge · all misses and failures retained · CVaR and Wilson confidence evidence.
                </span>
              </span>
            <label className="mt-3 block text-xs text-slate-300">
              <span className="mb-1 block">Chip Target Carry</span>
              <span className="flex items-center gap-2">
                <DecimalInput
                  min={1}
                  max={200}
                  value={chipTargetCarryYd}
                  aria-label="Chip target carry yards"
                  onCommit={(value) => {
                    onChipTargetCarryYdChange(value);
                    onConfigurationChange();
                  className={INPUT_CLASS}
                <span>yd</span>
              </span>
        )}
      </div>

      <div className={PANEL_CLASS}>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Varied Variables (Noise)
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          Grouped correlation or covariance plans sample normal inputs jointly;
          correlation controls co-movement while each noise scale retains its
          marginal standard deviation.
        </p>
        <p
          className={`mb-3 text-xs ${plan.ballSetup?.supportMode === "tee" ? "text-sky-300" : "text-amber-300"}`}
        >
          {plan.ballSetup?.supportMode === "tee"
            ? "Tee Height is available as a numeric variation input for the active Tee setup."
            : "Tee Height is excluded in Ground mode. Select Tee in Simulation to enable it."}
        </p>
        {(plan.groups?.length ?? 0) > 0 && (
          <p className="mb-3 text-xs text-sky-300">
            This plan contains {plan.groups!.length} grouped correlation or
            covariance definition(s).
          </p>
        )}
        <p
          className={`mb-3 text-xs ${localizedCount > 0 ? "text-amber-300" : "text-slate-500"}`}
        >
          Localized specs using time windows or point IDs are retained in files
          and the library, but cannot yet execute in the scalar browser path.
        </p>
        {plan.noise.map((spec, index) => {
          const definition = variableDef(spec.variableKey);
          const specId = spec.specId ?? spec.variableKey;
          const grouped = (plan.groups ?? []).some((group) =>
            group.specIds.includes(specId),
          );
          const covarianceGrouped = (plan.groups ?? []).some(
            (group) =>
              group.matrixKind === "covariance" &&
              group.specIds.includes(specId),
          );
          return (
            <div
              key={spec.specId ?? `${spec.variableKey}-${index}`}
              className="mb-3 rounded-lg border border-slate-800 bg-slate-950/40 p-3"
            >
              <div className="mb-2 flex gap-2">
                <select
                  aria-label={`Variable ${index + 1}`}
                  value={spec.variableKey}
                  onChange={(event) =>
                    changeVariable(index, event.target.value)
                  }
                  title={`${spec.variableKey} — ${definition?.guidance ?? ""}`}
                  className={`${INPUT_CLASS} flex-1`}
                >
                  {legalKeys.map((key) => (
                    <option key={key} value={key}>
                      {variableLabel(key)}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() =>
                    updatePlan({
                      noise:
                        plan.noise.length > 1
                          ? plan.noise.filter(
                              (_row, rowIndex) => rowIndex !== index,
                            )
                          : plan.noise,
                      groups: [],
                    })
                  }
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
                    setSpec(index, {
                      distribution: event.target.value as Distribution,
                    })
                  }
                  title={
                    grouped
                      ? "Grouped correlation and covariance specs must use a normal distribution."
                      : "Normal uses scale as standard deviation; uniform and triangular use half-width."
                  }
                  className={INPUT_CLASS}
                >
                  {DISTRIBUTIONS.map((distribution) => (
                    <option key={distribution} value={distribution}>
                      {distribution}
                    </option>
                  ))}
                </select>
                <DecimalInput
                  step="any"
                  disabled={covarianceGrouped}
                  value={spec.scale}
                  aria-label={`${variableLabel(spec.variableKey)} noise scale`}
                  onCommit={(value) => setSpec(index, { scale: value })}
                  title={
                    covarianceGrouped
                      ? "Covariance-group scale is fixed by the covariance matrix diagonal."
                      : `Noise scale [${definition?.unit ?? ""}]. ${definition?.guidance ?? ""}`
                  }
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
                        [bound]:
                          event.target.value === ""
                            ? null
                            : Number(event.target.value),
                      })
                    }
                    title={`Optional absolute ${bound} clipping bound.`}
                    className={INPUT_CLASS}
                  />
                ))}
              </div>
            </div>
          );
        })}
        <button
          type="button"
          disabled={!canAdd}
          onClick={() =>
            updatePlan({
              noise: [
                ...plan.noise,
                defaultSpec(plan.mode, usedVariables, plan.ballSetup),
              ],
            })
          }
          title="Add another unique varied variable."
          className={BUTTON_CLASS}
        >
          Add Variable
        </button>
      </div>
    </>
  );
}
