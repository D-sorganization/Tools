import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";
import { localizedTorqueSourceLabel } from "./localizedTorquePresentation";
import { PANEL_CLASS } from "./variationUi";

export function VariationLocalizedSources({
  ensemble,
}: { readonly ensemble: SwingVariationResultTs }): JSX.Element | null {
  const firstCommands = ensemble.runs.find(
    (trial) => trial.localizedTorqueCommands.length > 0,
  )?.localizedTorqueCommands ?? [];
  if (firstCommands.length === 0) return null;
  return (
    <section aria-label="Localized torque result sources" className={PANEL_CLASS}>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Localized Torque Result Sources
      </h2>
      <p className="mb-3 text-xs text-slate-500">
        Filter labels identify commanded-torque sources. Topological joint.* IDs are not
        spatial swing.* trace points; exported values use N*m and half-open [start, end) windows.
      </p>
      <ul className="flex flex-wrap gap-2" aria-label="Localized torque source filters">
        {firstCommands.map((command) => (
          <li key={command.specId}>
            <span className="inline-flex rounded-full border border-sky-500/40 bg-sky-950/40 px-3 py-1 text-xs text-sky-200">
              {localizedTorqueSourceLabel(ensemble, command.variableKey)}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
