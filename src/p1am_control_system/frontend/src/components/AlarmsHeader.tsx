import React from "react";
import { AlertOctagon, CheckCircle, BellRing } from "lucide-react";
import { ActiveAlarm } from "../App";

interface AlarmsHeaderProps {
  activeAlarms: ActiveAlarm[];
  onAcknowledgeAll: () => void;
}

export const AlarmsHeader: React.FC<AlarmsHeaderProps> = ({
  activeAlarms,
  onAcknowledgeAll,
}) => {
  // ⚡ Bolt Optimization: Use a single-pass loop instead of chained filter() and reduce()
  // to avoid intermediate array allocations and minimize garbage collection overhead.
  let unacknowledgedCount = 0;
  let highestSeverity = 0;
  for (let i = 0; i < activeAlarms.length; i++) {
    const a = activeAlarms[i];
    if (!a.acknowledged) {
      unacknowledgedCount++;
    }
    if (a.severity > highestSeverity) {
      highestSeverity = a.severity;
    }
  }

  let headerColorClass = "bg-gray-800/50 border-gray-700/50";
  let textColorClass = "text-gray-300";

  if (unacknowledgedCount > 0) {
    if (highestSeverity >= 2) {
      headerColorClass = "bg-red-900/30 border-red-500/50 animate-pulse";
      textColorClass = "text-red-400";
    } else if (highestSeverity === 1) {
      headerColorClass = "bg-yellow-900/30 border-yellow-500/50 animate-pulse";
      textColorClass = "text-yellow-400";
    }
  } else if (activeAlarms.length > 0) {
    if (highestSeverity >= 2) {
      headerColorClass = "bg-red-900/20 border-red-800/50";
      textColorClass = "text-red-500/70";
    } else {
      headerColorClass = "bg-yellow-900/20 border-yellow-800/50";
      textColorClass = "text-yellow-500/70";
    }
  }

  return (
    <div
      className={`flex items-center justify-between px-4 py-3 border rounded-xl mb-4 transition-colors duration-300 ${headerColorClass}`}
    >
      <div className="flex items-center gap-3">
        {activeAlarms.length > 0 ? (
          <AlertOctagon className={`w-6 h-6 ${textColorClass}`} />
        ) : (
          <CheckCircle className="w-6 h-6 text-emerald-500" />
        )}
        <div>
          <h2 className="font-semibold text-gray-100 flex items-center gap-2">
            System Status
            {unacknowledgedCount > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 text-xs border border-red-500/20">
                {unacknowledgedCount} Unacknowledged
              </span>
            )}
          </h2>
          <p className={`text-sm ${textColorClass}`}>
            {activeAlarms.length === 0
              ? "All systems normal. No active alarms."
              : `${activeAlarms.length} active alarm(s) detected.`}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {unacknowledgedCount > 0 && (
          <button
            onClick={onAcknowledgeAll}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200 shadow-lg hover:scale-105 active:scale-95 ${
              highestSeverity >= 2
                ? "bg-red-500 hover:bg-red-400 text-white shadow-red-500/20"
                : "bg-yellow-500 hover:bg-yellow-400 text-black shadow-yellow-500/20"
            }`}
          >
            <BellRing className="w-4 h-4" />
            Acknowledge All
          </button>
        )}
      </div>
    </div>
  );
};
