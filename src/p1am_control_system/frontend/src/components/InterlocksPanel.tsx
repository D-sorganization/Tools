import React, { useState, useId } from "react";
import { InterlockConfig } from "../App";
import { limitInputValue, parseLimitInput } from "../lib/limits";
import { ShieldAlert, Save } from "lucide-react";

interface InterlocksPanelProps {
  interlocks: InterlockConfig[];
  onChange: (tagId: number, field: keyof InterlockConfig, value: number | null) => void;
  onDeploy: () => void;
  deploying: boolean;
}

const InterlocksPanelImpl: React.FC<InterlocksPanelProps> = ({
  interlocks,
  onChange,
  onDeploy,
  deploying,
}) => {
  const [selectedTag, setSelectedTag] = useState<number>(0);
  const baseId = useId();

  const activeInterlock = interlocks[selectedTag];

  return (
    <div className="panel flex flex-col h-full">
      <div className="panel-header">
        <h2>
          <ShieldAlert size={16} className="text-red-400" />
          Alarms & Interlocks Config
        </h2>
        <button
          onClick={onDeploy}
          disabled={deploying}
          className="btn btn-primary"
          style={{ padding: "0.25rem 0.75rem", fontSize: "0.8rem" }}
        >
          {deploying ? "Deploying..." : (
            <span style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Save size={14} /> Deploy Config
            </span>
          )}
        </button>
      </div>
      <div className="p-4 flex-1 overflow-auto">
        <div className="flex gap-4 mb-6">
          <div className="flex-1">
            <label htmlFor={`${baseId}-select-tag`} className="block text-sm font-semibold text-gray-400 mb-2">
              Select Tag to Configure
            </label>
            <select
              id={`${baseId}-select-tag`}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white"
              value={selectedTag}
              onChange={(e) => setSelectedTag(Number(e.target.value))}
            >
              {interlocks.map((_, i) => (
                <option key={i} value={i}>
                  Tag {i}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-red-900/10 border border-red-500/30 rounded-xl p-4">
            <h3 className="text-red-400 font-bold mb-3 flex items-center gap-2">
              High-High (HH) Alarm
            </h3>
            <label htmlFor={`${baseId}-hihi-limit`} className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              id={`${baseId}-hihi-limit`}
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={limitInputValue(activeInterlock.hihi_limit)}
              placeholder="disabled"
              onChange={(e) => onChange(selectedTag, "hihi_limit", parseLimitInput(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers critical E-Stop condition.</p>
          </div>

          <div className="bg-yellow-900/10 border border-yellow-500/30 rounded-xl p-4">
            <h3 className="text-yellow-400 font-bold mb-3 flex items-center gap-2">
              High (H) Alarm
            </h3>
            <label htmlFor={`${baseId}-high-limit`} className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              id={`${baseId}-high-limit`}
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={limitInputValue(activeInterlock.high_limit)}
              placeholder="disabled"
              onChange={(e) => onChange(selectedTag, "high_limit", parseLimitInput(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers warning alarm, requires ACK.</p>
          </div>

          <div className="bg-yellow-900/10 border border-yellow-500/30 rounded-xl p-4">
            <h3 className="text-yellow-400 font-bold mb-3 flex items-center gap-2">
              Low (L) Alarm
            </h3>
            <label htmlFor={`${baseId}-low-limit`} className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              id={`${baseId}-low-limit`}
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={limitInputValue(activeInterlock.low_limit)}
              placeholder="disabled"
              onChange={(e) => onChange(selectedTag, "low_limit", parseLimitInput(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers warning alarm, requires ACK.</p>
          </div>

          <div className="bg-red-900/10 border border-red-500/30 rounded-xl p-4">
            <h3 className="text-red-400 font-bold mb-3 flex items-center gap-2">
              Low-Low (LL) Alarm
            </h3>
            <label htmlFor={`${baseId}-lolo-limit`} className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              id={`${baseId}-lolo-limit`}
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={limitInputValue(activeInterlock.lolo_limit)}
              placeholder="disabled"
              onChange={(e) => onChange(selectedTag, "lolo_limit", parseLimitInput(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers critical E-Stop condition.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Memoized so the interlocks editor skips re-render when its `interlocks` /
 * `onChange` / `onDeploy` / `deploying` props are unchanged.
 */
export const InterlocksPanel = React.memo(InterlocksPanelImpl);
