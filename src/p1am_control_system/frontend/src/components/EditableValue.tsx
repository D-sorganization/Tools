import React, { useState, useRef, useEffect, useCallback } from "react";
import { decideCommit } from "../lib/editableValue";
import "./EditableValue.css";

/**
 * A value that displays like text but edits in place.
 *
 * Click (or focus + Enter/Space) turns the reading into an input; Enter or blur
 * commits, Escape cancels. Parsing/clamping/"did it change" is delegated to the
 * pure {@link decideCommit} helper (DRY/DbC), so this component is only the UI
 * shell: it never writes an invalid or unchanged value, and a live `value` prop
 * update while not editing simply re-renders the reading.
 *
 * Reused everywhere the HMI shows an operator-changeable number (setpoints,
 * cutoffs, interlock limits, config), so all inline edits behave identically.
 */
export interface EditableValueProps {
  /** Current committed value. */
  value: number;
  /** Called with the new (clamped) value only when it is valid AND changed. */
  onCommit: (next: number) => void | Promise<void>;
  /** Accessible name for the control (required — DbC). */
  label: string;
  /** Display formatter for the reading (default: `String`). */
  format?: (v: number) => string;
  /** Custom parser passed through to {@link decideCommit}. */
  parse?: (raw: string) => number;
  /** Unit suffix shown after the reading, e.g. "°C". */
  unit?: string;
  /** Inclusive clamp bounds. */
  min?: number;
  max?: number;
  /** Native input step (edit mode only). */
  step?: number;
  /** When true the value renders as plain, non-editable text. */
  disabled?: boolean;
  /** Tooltip / title on the reading. */
  title?: string;
  className?: string;
}

/** Plain-number string for the input box (no unit, no display formatting). */
function toInputText(value: number): string {
  return Number.isFinite(value) ? String(value) : "";
}

export const EditableValue: React.FC<EditableValueProps> = ({
  value,
  onCommit,
  label,
  format = String,
  parse,
  unit,
  min,
  max,
  step,
  disabled = false,
  title,
  className,
}) => {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [invalid, setInvalid] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Select-all on entering edit so the operator can just type the new value.
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const beginEdit = useCallback(() => {
    if (disabled || busy) return;
    setDraft(toInputText(value));
    setInvalid(false);
    setEditing(true);
  }, [disabled, busy, value]);

  const cancel = useCallback(() => {
    setEditing(false);
    setInvalid(false);
  }, []);

  const commit = useCallback(async () => {
    const decision = decideCommit(draft, value, { min, max, parse });
    if (!decision.ok) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    if (decision.changed) {
      setBusy(true);
      try {
        await onCommit(decision.value);
      } finally {
        setBusy(false);
      }
    }
    setEditing(false);
  }, [draft, value, min, max, parse, onCommit]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        void commit();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancel();
      }
    },
    [commit, cancel],
  );

  // Blur commits when valid, otherwise discards the edit (don't trap the user).
  const onBlur = useCallback(() => {
    const decision = decideCommit(draft, value, { min, max, parse });
    if (decision.ok) {
      void commit();
    } else {
      cancel();
    }
  }, [draft, value, min, max, parse, commit, cancel]);

  const reading = `${format(value)}${unit ? ` ${unit}` : ""}`;

  if (disabled) {
    return (
      <span className={`editable-value editable-value--disabled ${className ?? ""}`} title={title}>
        {reading}
      </span>
    );
  }

  if (!editing) {
    return (
      <button
        type="button"
        className={`editable-value editable-value--reading ${className ?? ""}`}
        onClick={beginEdit}
        aria-label={`${label}: ${reading}. Click to edit.`}
        title={title ?? "Click to edit"}
        disabled={busy}
      >
        <span className="editable-value__text">{reading}</span>
        <span className="editable-value__pencil" aria-hidden="true">
          ✎
        </span>
      </button>
    );
  }

  return (
    <span className={`editable-value editable-value--editing ${className ?? ""}`}>
      <input
        ref={inputRef}
        type="number"
        className={`editable-value__input${invalid ? " editable-value__input--invalid" : ""}`}
        value={draft}
        min={min}
        max={max}
        step={step}
        aria-label={label}
        aria-invalid={invalid}
        onChange={(e) => {
          setDraft(e.target.value);
          if (invalid) setInvalid(false);
        }}
        onKeyDown={onKeyDown}
        onBlur={onBlur}
      />
      {unit ? <span className="editable-value__unit">{unit}</span> : null}
    </span>
  );
};
