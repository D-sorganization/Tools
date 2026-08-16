import { variableLabel } from "../model/variation";
import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";

/** Describe a localized source without confusing topological and spatial IDs. */
export function localizedTorqueSourceLabel(
  ensemble: SwingVariationResultTs,
  variableKey: string,
): string {
  const command = ensemble.runs
    .flatMap((trial) => trial.localizedTorqueCommands)
    .find((candidate) => candidate.variableKey === variableKey);
  if (!command) return variableKey;
  return [
    variableLabel(command.variableKey), command.specId, command.jointId,
    `[${command.timeWindowS[0]}, ${command.timeWindowS[1]}) s`,
    command.unit, command.provenance,
  ].join(" · ");
}
