import { useEffect, useRef, useState, type InputHTMLAttributes } from "react";

const DECIMAL_DRAFT = /^-?(?:\d+(?:\.\d*)?|\.\d*)?$/;

type NativeProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "type" | "value" | "defaultValue" | "onChange" | "onBlur"
>;

export interface DecimalInputProps extends NativeProps {
  value: number;
  onCommit: (value: number) => void;
  min?: number;
  max?: number;
  format?: (value: number) => string;
}

function bounded(value: number, min?: number, max?: number): number {
  return Math.min(max ?? Number.POSITIVE_INFINITY, Math.max(min ?? Number.NEGATIVE_INFINITY, value));
}

/**
 * Decimal editor that keeps transient text separate from committed model state.
 *
 * Native controlled number fields cannot represent intermediate edits such as
 * an empty value or a lone minus sign. Keeping a draft fixes replacement and
 * negative-number entry while preserving a numeric model boundary.
 */
export function DecimalInput({
  value,
  onCommit,
  min,
  max,
  format = String,
  className = "",
  onFocus,
  onKeyDown,
  ...props
}: DecimalInputProps) {
  const [draft, setDraft] = useState(format(value));
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setDraft(format(value));
  }, [format, value]);

  const commit = () => {
    const parsed = Number(draft);
    if (draft.trim() === "" || !Number.isFinite(parsed)) {
      setDraft(format(value));
      return;
    }
    const next = bounded(parsed, min, max);
    setDraft(format(next));
    onCommit(next);
  };

  return (
    <input
      {...props}
      type="text"
      inputMode="decimal"
      value={draft}
      min={min}
      max={max}
      className={`numeric-input ${className}`.trim()}
      onChange={(event) => {
        if (DECIMAL_DRAFT.test(event.target.value)) setDraft(event.target.value);
      }}
      onFocus={(event) => {
        focused.current = true;
        event.currentTarget.select();
        onFocus?.(event);
      }}
      onBlur={() => {
        focused.current = false;
        commit();
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur();
        if (event.key === "Escape") {
          setDraft(format(value));
          event.currentTarget.blur();
        }
        onKeyDown?.(event);
      }}
    />
  );
}
