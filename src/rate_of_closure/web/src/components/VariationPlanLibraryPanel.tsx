import type { NamedVariationPlan } from "../model/variationPlanLibrary";
import { BUTTON_CLASS, INPUT_CLASS, PANEL_CLASS } from "./variationUi";

interface VariationPlanLibraryPanelProps {
  plans: NamedVariationPlan[];
  selectedId: string;
  name: string;
  onSelectedIdChange: (id: string) => void;
  onNameChange: (name: string) => void;
  onSave: () => void;
  onLoad: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}

export function VariationPlanLibraryPanel({
  plans,
  selectedId,
  name,
  onSelectedIdChange,
  onNameChange,
  onSave,
  onLoad,
  onDuplicate,
  onDelete,
}: VariationPlanLibraryPanelProps): JSX.Element {
  const hasSelection = plans.some((plan) => plan.id === selectedId);
  return (
    <div className={PANEL_CLASS}>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Named Plan Library
      </h2>
      <label className="mb-2 block text-sm" title="Persistent plan name stored in this browser.">
        <span className="mb-1 block text-slate-300">Plan Name</span>
        <input
          aria-label="Plan name"
          value={name}
          onChange={(event) => onNameChange(event.target.value)}
          className={INPUT_CLASS}
        />
      </label>
      <label className="mb-3 block text-sm" title="Select a persistent named variation plan.">
        <span className="mb-1 block text-slate-300">Saved Plans</span>
        <select
          aria-label="Saved plan library"
          value={selectedId}
          onChange={(event) => onSelectedIdChange(event.target.value)}
          className={INPUT_CLASS}
        >
          <option value="">No Plan Selected</option>
          {plans.map((plan) => (
            <option key={plan.id} value={plan.id}>{plan.name}</option>
          ))}
        </select>
      </label>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onSave}
          disabled={name.trim().length === 0}
          title="Save the complete current v2 plan to local browser storage."
          className={BUTTON_CLASS}
        >
          Save Named Plan
        </button>
        <button
          type="button"
          onClick={onLoad}
          disabled={!hasSelection}
          title="Load the selected plan including base values, flight model, groups, and locus metadata."
          className={BUTTON_CLASS}
        >
          Load Selected Plan
        </button>
        <button
          type="button"
          onClick={onDuplicate}
          disabled={!hasSelection}
          title="Create an independent copy of the selected named plan."
          className={BUTTON_CLASS}
        >
          Duplicate Selected Plan
        </button>
        <button
          type="button"
          onClick={onDelete}
          disabled={!hasSelection}
          title="Delete the selected plan from local browser storage."
          className={BUTTON_CLASS}
        >
          Delete Selected Plan
        </button>
      </div>
    </div>
  );
}
