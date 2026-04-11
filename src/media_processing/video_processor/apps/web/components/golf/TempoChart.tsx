"use client";

import { TempoMetrics } from "@/lib/golf/types";

interface TempoChartProps {
  tempo: TempoMetrics;
}

export default function TempoChart({ tempo }: TempoChartProps) {
  const getTempoQuality = (): {
    label: string;
    color: string;
    description: string;
  } => {
    const ratio = tempo.tempoRatio;
    if (ratio >= 2.5 && ratio <= 3.5) {
      return {
        label: "Excellent",
        color: "text-green-600",
        description:
          "Your tempo is in the ideal 3:1 range used by most professional golfers.",
      };
    }
    if (ratio >= 2 && ratio <= 4) {
      return {
        label: "Good",
        color: "text-yellow-600",
        description:
          "Your tempo is close to ideal. Minor adjustments could improve consistency.",
      };
    }
    if (ratio < 2) {
      return {
        label: "Too Quick",
        color: "text-red-600",
        description:
          "Your downswing is too fast relative to your backswing. Try slowing down the transition.",
      };
    }
    return {
      label: "Too Slow",
      color: "text-orange-600",
      description:
        "Your backswing is too slow. This can cause timing issues and loss of power.",
    };
  };

  const getRhythmColor = (): string => {
    switch (tempo.rhythm) {
      case "smooth":
        return "bg-green-500";
      case "quick":
        return "bg-yellow-500";
      case "slow":
        return "bg-orange-500";
      case "uneven":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const quality = getTempoQuality();
  const backswingPercent =
    (tempo.backswingDuration / tempo.totalSwingDuration) * 100;
  const downswingPercent =
    (tempo.downswingDuration / tempo.totalSwingDuration) * 100;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Tempo Analysis
      </h3>

      {/* Main Tempo Display */}
      <div className="flex items-center justify-center space-x-8 mb-6">
        <div className="text-center">
          <p className="text-4xl font-bold text-blue-600">
            {(tempo.backswingDuration / 1000).toFixed(2)}s
          </p>
          <p className="text-sm text-gray-500">Backswing</p>
        </div>

        <div className="flex flex-col items-center">
          <div className="w-px h-8 bg-gray-300" />
          <div className="px-4 py-2 bg-gray-100 rounded-full">
            <span className="text-lg font-bold text-gray-900">
              {tempo.tempoRatio.toFixed(1)} : 1
            </span>
          </div>
          <div className="w-px h-8 bg-gray-300" />
        </div>

        <div className="text-center">
          <p className="text-4xl font-bold text-purple-600">
            {(tempo.downswingDuration / 1000).toFixed(2)}s
          </p>
          <p className="text-sm text-gray-500">Downswing</p>
        </div>
      </div>

      {/* Tempo Bar Visualization */}
      <div className="mb-6">
        <div className="flex h-6 rounded-lg overflow-hidden">
          <div
            className="bg-blue-500 flex items-center justify-center text-white text-xs font-medium"
            style={{ width: `${backswingPercent}%` }}
          >
            {backswingPercent.toFixed(0)}%
          </div>
          <div
            className="bg-purple-500 flex items-center justify-center text-white text-xs font-medium"
            style={{ width: `${downswingPercent}%` }}
          >
            {downswingPercent.toFixed(0)}%
          </div>
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>Address</span>
          <span>Top</span>
          <span>Impact</span>
        </div>
      </div>

      {/* Quality Assessment */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            Tempo Quality
          </span>
          <span className={`font-bold ${quality.color}`}>{quality.label}</span>
        </div>
        <p className="text-sm text-gray-600">{quality.description}</p>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-2xl font-bold text-gray-900">
            {(tempo.totalSwingDuration / 1000).toFixed(2)}s
          </p>
          <p className="text-sm text-gray-500">Total Duration</p>
        </div>

        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-2xl font-bold text-gray-900">
            {tempo.transitionPause.toFixed(0)}ms
          </p>
          <p className="text-sm text-gray-500">Transition Pause</p>
        </div>

        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${getRhythmColor()}`} />
            <p className="text-lg font-bold text-gray-900 capitalize">
              {tempo.rhythm}
            </p>
          </div>
          <p className="text-sm text-gray-500">Rhythm</p>
        </div>
      </div>

      {/* Pro Comparison */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-3">
          Compare to Tour Pros
        </h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Rory McIlroy</span>
            <span className="font-medium">3.1:1</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Tiger Woods</span>
            <span className="font-medium">3.0:1</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Ernie Els</span>
            <span className="font-medium">4.2:1</span>
          </div>
          <div className="flex items-center justify-between text-sm font-bold">
            <span className="text-blue-600">Your Tempo</span>
            <span className="text-blue-600">
              {tempo.tempoRatio.toFixed(1)}:1
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
