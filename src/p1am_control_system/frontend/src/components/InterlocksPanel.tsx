import React, { useState } from "react";
import { InterlockConfig } from "../App";
import { ShieldAlert, Save } from "lucide-react";

interface InterlocksPanelProps {
  interlocks: InterlockConfig[];
  onChange: (tagId: number, field: keyof InterlockConfig, value: number) => void;
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
          {deploying ? "Creating Draft..." : (
            <span style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Save size={14} /> Create Protected Draft
            </span>
          )}
        </button>
      </div>
      <div className="p-4 flex-1 overflow-auto">
        <div className="flex gap-4 mb-6">
          <div className="flex-1">
            <label className="block text-sm font-semibold text-gray-400 mb-2">
              Select Tag to Configure
            </label>
            <select
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
            <label className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={activeInterlock.hihi_limit}
              onChange={(e) => onChange(selectedTag, "hihi_limit", parseFloat(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers critical E-Stop condition.</p>
          </div>

          <div className="bg-yellow-900/10 border border-yellow-500/30 rounded-xl p-4">
            <h3 className="text-yellow-400 font-bold mb-3 flex items-center gap-2">
              High (H) Alarm
            </h3>
            <label className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={activeInterlock.high_limit}
              onChange={(e) => onChange(selectedTag, "high_limit", parseFloat(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers warning alarm, requires ACK.</p>
          </div>

          <div className="bg-yellow-900/10 border border-yellow-500/30 rounded-xl p-4">
            <h3 className="text-yellow-400 font-bold mb-3 flex items-center gap-2">
              Low (L) Alarm
            </h3>
            <label className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={activeInterlock.low_limit}
              onChange={(e) => onChange(selectedTag, "low_limit", parseFloat(e.target.value))}
            />
            <p className="text-xs text-gray-500">Triggers warning alarm, requires ACK.</p>
          </div>

          <div className="bg-red-900/10 border border-red-500/30 rounded-xl p-4">
            <h3 className="text-red-400 font-bold mb-3 flex items-center gap-2">
              Low-Low (LL) Alarm
            </h3>
            <label className="block text-xs text-gray-400 mb-1">Trigger Limit</label>
            <input
              type="number"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2 text-white mb-2"
              value={activeInterlock.lolo_limit}
              onChange={(e) => onChange(selectedTag, "lolo_limit", parseFloat(e.target.value))}
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
