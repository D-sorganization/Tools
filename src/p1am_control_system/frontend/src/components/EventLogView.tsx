import React from "react";
import { EventLogEntry } from "../App";
import { List } from "lucide-react";

interface EventLogViewProps {
  events: EventLogEntry[];
}

const EventLogViewImpl: React.FC<EventLogViewProps> = ({ events }) => {
  return (
    <div className="panel flex flex-col h-full">
      <div className="panel-header">
        <h2>
          <List size={16} className="text-blue-400" />
          Event & Alarm Log
        </h2>
      </div>
      <div className="flex-1 overflow-auto p-0">
        <table className="w-full text-left border-collapse text-sm">
          <thead className="bg-gray-800/80 sticky top-0 backdrop-blur-sm z-10 text-gray-400">
            <tr>
              <th className="p-3 font-medium border-b border-gray-700">Timestamp</th>
              <th className="p-3 font-medium border-b border-gray-700">Type</th>
              <th className="p-3 font-medium border-b border-gray-700">Description</th>
              <th className="p-3 font-medium border-b border-gray-700">Severity</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr>
                <td colSpan={4} className="p-8 text-center text-gray-500 italic">
                  No events recorded.
                </td>
              </tr>
            ) : (
              events.map((event, i) => {
                let rowColor = "hover:bg-gray-800/40";
                let typeColor = "text-gray-300";

                if (event.severity >= 2) {
                  rowColor = "bg-red-900/10 hover:bg-red-900/20";
                  typeColor = "text-red-400 font-semibold";
                } else if (event.severity === 1) {
                  rowColor = "bg-yellow-900/10 hover:bg-yellow-900/20";
                  typeColor = "text-yellow-400 font-semibold";
                } else if (event.event_type === "ACKNOWLEDGE") {
                  typeColor = "text-emerald-400 font-semibold";
                }

                return (
                  <tr key={i} className={`border-b border-gray-800/50 transition-colors ${rowColor}`}>
                    <td className="p-3 text-gray-400 font-mono text-xs whitespace-nowrap">
                      {new Date(event.timestamp).toLocaleString()}
                    </td>
                    <td className={`p-3 text-xs tracking-wider ${typeColor}`}>
                      {event.event_type}
                    </td>
                    <td className="p-3 text-gray-300">
                      {event.description}
                    </td>
                    <td className="p-3">
                      {event.severity > 0 && (
                        <span
                          className={`px-2 py-1 rounded text-xs font-bold ${
                            event.severity >= 2
                              ? "bg-red-500/20 text-red-400"
                              : "bg-yellow-500/20 text-yellow-400"
                          }`}
                        >
                          Level {event.severity}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/** Memoized so the event log skips re-render when its `events` prop is stable. */
export const EventLogView = React.memo(EventLogViewImpl);
