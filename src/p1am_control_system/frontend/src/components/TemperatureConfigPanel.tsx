import React from "react";
import type { TemperatureConfig } from "../types";

/**
 * "Calibration & limits" editor for the temperature controller.
 *
 * Split out of TemperatureControl.tsx unchanged. The draft lives in the parent
 * (which also owns the PUT and the server's authoritative response), so this
 * file is the form and nothing else — edits are reported through `onDraftChange`
 * and only `onSave` commits.
 */
interface Props {
  draft: TemperatureConfig;
  busy: boolean;
  onDraftChange: (next: TemperatureConfig) => void;
  onSave: () => void;
}

export const TemperatureConfigPanel: React.FC<Props> = ({
  draft,
  busy,
  onDraftChange,
  onSave,
}) => (
  <details className="tc-card tc-config">
    <summary>Calibration &amp; limits</summary>

    <p className="tc-hint" style={{ marginTop: "0.8rem" }}>
      The thermocouple module reports temperature directly. These limits set the
      safe operating envelope — the server clamps setpoints to the band and
      latches the heater OFF at the high-high cutoff.
    </p>

    <div className="tc-config-grid">
      <ConfigField
        label="Deadband (°C, ± half-band)"
        value={draft.deadband_c}
        onChange={(v) => onDraftChange({ ...draft, deadband_c: v })}
      />
      <ConfigField
        label="Min ON time (s)"
        value={draft.min_on_time_s}
        onChange={(v) => onDraftChange({ ...draft, min_on_time_s: v })}
      />
      <ConfigField
        label="Min OFF time (s)"
        value={draft.min_off_time_s}
        onChange={(v) => onDraftChange({ ...draft, min_off_time_s: v })}
      />
      <ConfigField
        label="High-high cutoff (°C)"
        value={draft.hh_limit_c}
        onChange={(v) => onDraftChange({ ...draft, hh_limit_c: v })}
      />
      <ConfigField
        label="Max setpoint (°C)"
        value={draft.setpoint_max_c}
        onChange={(v) => onDraftChange({ ...draft, setpoint_max_c: v })}
      />
      <ConfigField
        label="Type-K tag"
        value={draft.type_k.tag}
        stringMode
        onChange={(v) =>
          onDraftChange({
            ...draft,
            type_k: { ...draft.type_k, tag: v as unknown as string },
          })
        }
      />
      <ConfigField
        label="Type-K full scale (°C)"
        value={draft.type_k.full_scale_c}
        onChange={(v) =>
          onDraftChange({
            ...draft,
            type_k: { ...draft.type_k, full_scale_c: v },
          })
        }
      />
      <ConfigField
        label="Type-R tag"
        value={draft.type_r.tag}
        stringMode
        onChange={(v) =>
          onDraftChange({
            ...draft,
            type_r: { ...draft.type_r, tag: v as unknown as string },
          })
        }
      />
      <ConfigField
        label="Type-R full scale (°C)"
        value={draft.type_r.full_scale_c}
        onChange={(v) =>
          onDraftChange({
            ...draft,
            type_r: { ...draft.type_r, full_scale_c: v },
          })
        }
      />
      <ConfigField
        label="Heater label"
        value={draft.heater_label}
        stringMode
        onChange={(v) =>
          onDraftChange({ ...draft, heater_label: v as unknown as string })
        }
      />
    </div>

    <div style={{ marginTop: "1rem" }}>
      <button
        className={`btn ${busy ? "tc-disabled" : ""}`}
        onClick={onSave}
        disabled={busy}
      >
        Save Configuration
      </button>
    </div>
  </details>
);

interface ConfigFieldProps {
  label: string;
  value: number | string;
  stringMode?: boolean;
  onChange: (v: number) => void;
}
const ConfigField: React.FC<ConfigFieldProps> = ({
  label,
  value,
  stringMode,
  onChange,
}) => (
  <label className="tc-field">
    <span>{label}</span>
    <input
      type={stringMode ? "text" : "number"}
      value={value}
      onChange={(e) =>
        onChange(
          (stringMode ? e.target.value : Number.parseFloat(e.target.value)) as number,
        )
      }
    />
  </label>
);
