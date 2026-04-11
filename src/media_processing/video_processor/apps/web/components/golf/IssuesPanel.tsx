"use client";

import { SwingIssue, SwingPhase } from "@/lib/golf/types";
import { useState } from "react";

interface IssuesPanelProps {
  issues: SwingIssue[];
}

export default function IssuesPanel({ issues }: IssuesPanelProps) {
  const [expandedIssue, setExpandedIssue] = useState<string | null>(null);

  const getSeverityStyles = (severity: SwingIssue["severity"]) => {
    switch (severity) {
      case "major":
        return {
          bg: "bg-red-50",
          border: "border-red-200",
          icon: "bg-red-500",
          text: "text-red-700",
          badge: "bg-red-100 text-red-700",
        };
      case "moderate":
        return {
          bg: "bg-yellow-50",
          border: "border-yellow-200",
          icon: "bg-yellow-500",
          text: "text-yellow-700",
          badge: "bg-yellow-100 text-yellow-700",
        };
      case "minor":
        return {
          bg: "bg-green-50",
          border: "border-green-200",
          icon: "bg-green-500",
          text: "text-green-700",
          badge: "bg-green-100 text-green-700",
        };
    }
  };

  const getPhaseLabel = (phase: SwingPhase): string => {
    const labels: Record<SwingPhase, string> = {
      [SwingPhase.ADDRESS]: "Address",
      [SwingPhase.TAKEAWAY]: "Takeaway",
      [SwingPhase.BACKSWING]: "Backswing",
      [SwingPhase.TOP_OF_BACKSWING]: "Top of Backswing",
      [SwingPhase.TRANSITION]: "Transition",
      [SwingPhase.DOWNSWING]: "Downswing",
      [SwingPhase.IMPACT]: "Impact",
      [SwingPhase.FOLLOW_THROUGH]: "Follow Through",
      [SwingPhase.FINISH]: "Finish",
      [SwingPhase.UNKNOWN]: "Unknown",
    };
    return labels[phase];
  };

  // Sort issues by severity
  const sortedIssues = [...issues].sort((a, b) => {
    const severityOrder = { major: 0, moderate: 1, minor: 2 };
    return severityOrder[a.severity] - severityOrder[b.severity];
  });

  const majorCount = issues.filter((i) => i.severity === "major").length;
  const moderateCount = issues.filter((i) => i.severity === "moderate").length;
  const minorCount = issues.filter((i) => i.severity === "minor").length;

  if (issues.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Swing Issues
        </h3>
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-green-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
          <p className="text-lg font-medium text-gray-900">
            No Issues Detected
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Your swing mechanics look great! Keep up the good work.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Swing Issues</h3>
        <div className="flex items-center space-x-2">
          {majorCount > 0 && (
            <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-700 rounded-full">
              {majorCount} Major
            </span>
          )}
          {moderateCount > 0 && (
            <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-700 rounded-full">
              {moderateCount} Moderate
            </span>
          )}
          {minorCount > 0 && (
            <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-700 rounded-full">
              {minorCount} Minor
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3">
        {sortedIssues.map((issue) => {
          const styles = getSeverityStyles(issue.severity);
          const isExpanded = expandedIssue === issue.id;

          return (
            <div
              key={issue.id}
              className={`rounded-lg border ${styles.border} ${styles.bg} overflow-hidden transition-all duration-200`}
            >
              <button
                onClick={() => setExpandedIssue(isExpanded ? null : issue.id)}
                className="w-full p-4 text-left flex items-start justify-between hover:bg-opacity-80"
              >
                <div className="flex items-start space-x-3">
                  <div
                    className={`w-2 h-2 rounded-full ${styles.icon} mt-2 flex-shrink-0`}
                  />
                  <div>
                    <p className={`font-medium ${styles.text}`}>{issue.name}</p>
                    <p className="text-sm text-gray-600 mt-1">
                      {issue.description}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2 ml-4">
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${styles.badge}`}
                  >
                    {getPhaseLabel(issue.phase)}
                  </span>
                  <svg
                    className={`w-5 h-5 text-gray-400 transition-transform ${
                      isExpanded ? "rotate-180" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </div>
              </button>

              {isExpanded && (
                <div className="px-4 pb-4 border-t border-opacity-50 border-current">
                  <div className="pt-4 space-y-3">
                    {/* Measurement details */}
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Measured Value</span>
                      <span className="font-medium">
                        {issue.measurementValue.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Expected Range</span>
                      <span className="font-medium">
                        {issue.expectedRange[0]} - {issue.expectedRange[1]}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Detected at Frame</span>
                      <span className="font-medium">{issue.detectedAt}</span>
                    </div>

                    {/* Drill recommendation */}
                    {issue.drillRecommendation && (
                      <div className="mt-4 p-3 bg-white bg-opacity-50 rounded-lg">
                        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
                          Recommended Drill
                        </p>
                        <p className="text-sm text-gray-700">
                          {issue.drillRecommendation}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
