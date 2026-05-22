/**
 * Reusable unit selector dropdown (DRY).
 * Renders a compact dropdown for selecting units on any quantity.
 */
import React, { useId } from 'react';

interface UnitSelectorProps<T extends string> {
    label: string;
    value: T;
    options: readonly T[];
    onChange: (unit: T) => void;
}

export function UnitSelector<T extends string>({
    label, value, options, onChange,
}: UnitSelectorProps<T>): React.ReactElement {
    const id = useId();
    return (
        <div className="unit-row">
            <label className="unit-label" htmlFor={id}>{label}</label>
            <select
                id={id}
                className="unit-select"
                value={value}
                onChange={e => onChange(e.target.value as T)}
            >
                {options.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                ))}
            </select>
        </div>
    );
}
