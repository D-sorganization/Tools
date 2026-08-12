import {
  buildMorrisFactorRows,
  type MorrisFactorDraft,
} from "../model/morrisAuthorityRequest";
import type { SupportMode } from "../model/ballSetup";
import { INPUT_CLASS } from "./variationUi";
import { DecimalInput } from "./DecimalInput";

interface MorrisFactorEditorProps {
  readonly drafts: readonly MorrisFactorDraft[];
  readonly supportMode: SupportMode;
  readonly disabled: boolean;
  readonly onChange: (drafts: readonly MorrisFactorDraft[]) => void;
}

export function MorrisFactorEditor(props: MorrisFactorEditorProps) {
  const rows = buildMorrisFactorRows(props.drafts, props.supportMode);
  const update = (index: number, change: Partial<MorrisFactorDraft>) => {
    props.onChange(props.drafts.map((draft, draftIndex) => (
      draftIndex === index ? Object.freeze({ ...draft, ...change }) : draft
    )));
  };
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-xs">
        <thead className="text-slate-400">
          <tr><th className="p-2">Use</th><th>Factor</th><th>Lower</th><th>Upper</th><th>Unit</th><th>Validation</th></tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.variableKey} data-variable-key={row.variableKey}
              className="border-t border-slate-800 align-top">
              <td className="p-2"><input type="checkbox" checked={row.enabled}
                disabled={props.disabled || !row.applicable}
                aria-label={`Use ${row.label}`}
                title={`Include ${row.label} in the Morris design`}
                onChange={(event) => update(index, { enabled: event.target.checked })} /></td>
              <td className="p-2"><span className="font-medium text-slate-200">{row.label}</span>
                <span className="mt-1 block max-w-xs text-slate-500">{row.guidance}</span></td>
              <td className="p-2"><DecimalInput className={INPUT_CLASS} value={row.lower ?? 0}
                aria-label={`${row.label} lower bound`} disabled={props.disabled || !row.enabled}
                aria-invalid={row.validationError !== null}
                aria-describedby={`morris-factor-error-${row.specId}`}
                title={`Minimum ${row.label} value sampled by the authority`}
                onCommit={(lower) => update(index, { lower })} /></td>
              <td className="p-2"><DecimalInput className={INPUT_CLASS} value={row.upper ?? 0}
                aria-label={`${row.label} upper bound`} disabled={props.disabled || !row.enabled}
                aria-invalid={row.validationError !== null}
                aria-describedby={`morris-factor-error-${row.specId}`}
                title={`Maximum ${row.label} value sampled by the authority`}
                onCommit={(upper) => update(index, { upper })} /></td>
              <td className="p-2 text-slate-400">{row.unit}</td>
              <td id={`morris-factor-error-${row.specId}`} className="p-2 text-amber-300">
                {row.validationError ?? (!row.applicable ? "Requires tee support" : "Valid")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
