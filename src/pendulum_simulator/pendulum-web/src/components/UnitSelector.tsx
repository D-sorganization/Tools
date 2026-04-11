/**
 * Reusable unit selector dropdown (DRY).
 * Renders a compact dropdown for selecting units on any quantity.
 */
import React from 'react';

interface UnitSelectorProps<T extends string> {
    label: string;
    value: T;
    options: readonly T[];
    onChange: (unit: T) => void;
}

export function UnitSelector<T extends string>({
    label, value, options, onChange,
}: UnitSelectorProps<T>): React.ReactElement {
    return (
        <div className="unit-row">
            <span className="unit-label">{label}</span>
            <select
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
