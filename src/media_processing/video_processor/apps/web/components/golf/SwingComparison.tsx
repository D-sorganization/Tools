"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  SwingAnalysis,
  SwingSession,
  PoseFrame,
  Landmark,
  PoseLandmark,
} from "@/lib/golf/types";
import {
  getAllSessions,
  getAnalysisWithFrames,
  compareSwings,
} from "@/lib/golf/persistence";
import { SwingComparison as SwingComparisonType } from "@/lib/golf/types";

interface SwingComparisonProps {
  currentAnalysis: SwingAnalysis;
  onClose: () => void;
}

export default function SwingComparisonComponent({
  currentAnalysis,
  onClose,
}: SwingComparisonProps) {
  const [sessions, setSessions] = useState<SwingSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null,
  );
  const [comparison, setComparison] = useState<SwingComparisonType | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [overlayMode, setOverlayMode] = useState<"side-by-side" | "overlay">(
    "side-by-side",
  );
  const [syncedFrame, setSyncedFrame] = useState(0);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Load available sessions
  useEffect(() => {
    getAllSessions(20)
      .then((allSessions) => {
        // Filter out current session
        const otherSessions = allSessions.filter(
          (s) => s.id !== currentAnalysis.sessionId,
        );
        setSessions(otherSessions);
      })
      .finally(() => setLoading(false));
  }, [currentAnalysis.sessionId]);

  // Generate comparison when session selected
  useEffect(() => {
    if (!selectedSessionId) {
      setComparison(null);
      return;
    }

    compareSwings(currentAnalysis.sessionId, selectedSessionId).then(
      (result) => {
        setComparison(result ?? null);
      },
    );
  }, [selectedSessionId, currentAnalysis.sessionId]);

  // Draw overlay visualization
  const drawOverlay = useCallback(() => {
    if (!canvasRef.current || !comparison || overlayMode !== "overlay") return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Get frames at synced position
    const frame1 = comparison.swing1.poseFrames[syncedFrame];
    const frame2 =
      comparison.swing2.poseFrames[
        Math.min(syncedFrame, comparison.swing2.poseFrames.length - 1)
      ];

    if (!frame1 || !frame2) return;

    // Draw first swing (current) in blue
    drawPose(ctx, frame1.landmarks, "#3B82F6", 1, width, height);

    // Draw second swing (comparison) in orange with opacity
    drawPose(ctx, frame2.landmarks, "#F97316", overlayOpacity, width, height);

    // Draw legend
    ctx.font = "14px sans-serif";
    ctx.fillStyle = "#3B82F6";
    ctx.fillRect(10, 10, 20, 20);
    ctx.fillStyle = "#1F2937";
    ctx.fillText("Current Swing", 35, 25);

    ctx.fillStyle = "#F97316";
    ctx.globalAlpha = overlayOpacity;
    ctx.fillRect(10, 35, 20, 20);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#1F2937";
    ctx.fillText("Comparison Swing", 35, 50);
  }, [comparison, syncedFrame, overlayMode, overlayOpacity]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  // Draw pose skeleton
  const drawPose = (
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark[],
    color: string,
    opacity: number,
    width: number,
    height: number,
  ) => {
    ctx.globalAlpha = opacity;

    // Define connections (simplified skeleton)
    const connections: [PoseLandmark, PoseLandmark][] = [
      // Torso
      [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER],
      [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP],
      [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP],
      [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP],
      // Left arm
      [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW],
      [PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST],
      // Right arm
      [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW],
      [PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST],
      // Left leg
      [PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE],
      [PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE],
      // Right leg
      [PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE],
      [PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE],
    ];

    // Draw connections
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];
      if (startPoint && endPoint) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.stroke();
      }
    });

    // Draw joints
    ctx.fillStyle = color;
    landmarks.forEach((landmark) => {
      if (landmark.visibility && landmark.visibility > 0.5) {
        ctx.beginPath();
        ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    ctx.globalAlpha = 1;
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl p-8 max-w-4xl w-full mx-4">
          <p className="text-center text-gray-500">Loading sessions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-900">Swing Comparison</h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {sessions.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500">
                No other sessions available for comparison.
              </p>
              <p className="text-sm text-gray-400 mt-2">
                Record more swings to enable comparison.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Session Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select a swing to compare
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {sessions.map((session) => (
                    <button
                      key={session.id}
                      onClick={() => setSelectedSessionId(session.id)}
                      className={`p-4 border rounded-lg text-left transition-all ${
                        selectedSessionId === session.id
                          ? "border-blue-500 bg-blue-50 ring-2 ring-blue-200"
                          : "border-gray-200 hover:border-gray-300"
                      }`}
                    >
                      <p className="font-medium text-gray-900 truncate">
                        {session.videoFileName}
                      </p>
                      <p className="text-sm text-gray-500">
                        {new Date(session.timestamp).toLocaleDateString()}
                      </p>
                      <p className="text-lg font-bold text-blue-600 mt-1">
                        Score:{" "}
                        {Math.round(session.analysis?.scores?.overall || 0)}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Comparison View */}
              {comparison && (
                <>
                  {/* View Mode Toggle */}
                  <div className="flex items-center justify-center space-x-4">
                    <button
                      onClick={() => setOverlayMode("side-by-side")}
                      className={`px-4 py-2 rounded-lg font-medium ${
                        overlayMode === "side-by-side"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                      }`}
                    >
                      Side by Side
                    </button>
                    <button
                      onClick={() => setOverlayMode("overlay")}
                      className={`px-4 py-2 rounded-lg font-medium ${
                        overlayMode === "overlay"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                      }`}
                    >
                      Overlay
                    </button>
                  </div>

                  {/* Overlay Controls */}
                  {overlayMode === "overlay" && (
                    <div className="bg-gray-50 rounded-lg p-4 space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Frame: {syncedFrame}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max={Math.max(
                            comparison.swing1.poseFrames.length - 1,
                            comparison.swing2.poseFrames.length - 1,
                          )}
                          value={syncedFrame}
                          onChange={(e) =>
                            setSyncedFrame(parseInt(e.target.value))
                          }
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Comparison Opacity: {Math.round(overlayOpacity * 100)}
                          %
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={overlayOpacity}
                          onChange={(e) =>
                            setOverlayOpacity(parseFloat(e.target.value))
                          }
                          className="w-full"
                        />
                      </div>
                      <canvas
                        ref={canvasRef}
                        width={640}
                        height={480}
                        className="w-full bg-gray-900 rounded-lg"
                      />
                    </div>
                  )}

                  {/* Improvement Summary */}
                  <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm opacity-90">
                          Overall Improvement
                        </p>
                        <p className="text-4xl font-bold">
                          {comparison.overallImprovement.toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-6xl">
                        {comparison.overallImprovement >= 50 ? "📈" : "📉"}
                      </div>
                    </div>
                  </div>

                  {/* Metrics Comparison Table */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <table className="w-full">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">
                            Metric
                          </th>
                          <th className="px-4 py-3 text-center text-sm font-medium text-gray-700">
                            Previous
                          </th>
                          <th className="px-4 py-3 text-center text-sm font-medium text-gray-700">
                            Current
                          </th>
                          <th className="px-4 py-3 text-center text-sm font-medium text-gray-700">
                            Change
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {comparison.differences.map((diff, index) => (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-4 py-3 text-sm text-gray-900">
                              {diff.metric}
                            </td>
                            <td className="px-4 py-3 text-sm text-center text-gray-600">
                              {typeof diff.value1 === "number"
                                ? diff.value1.toFixed(2)
                                : diff.value1}
                            </td>
                            <td className="px-4 py-3 text-sm text-center text-gray-900 font-medium">
                              {typeof diff.value2 === "number"
                                ? diff.value2.toFixed(2)
                                : diff.value2}
                            </td>
                            <td className="px-4 py-3 text-sm text-center">
                              <span
                                className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                                  diff.improvement
                                    ? "bg-green-100 text-green-700"
                                    : diff.delta === 0
                                      ? "bg-gray-100 text-gray-700"
                                      : "bg-red-100 text-red-700"
                                }`}
                              >
                                {diff.delta > 0 ? "+" : ""}
                                {diff.delta.toFixed(2)}
                                {diff.improvement ? " ✓" : ""}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Side by Side Score Cards */}
                  {overlayMode === "side-by-side" && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 rounded-xl p-4">
                        <h4 className="text-sm font-medium text-gray-500 mb-3">
                          Previous Swing
                        </h4>
                        <div className="text-center">
                          <p className="text-5xl font-bold text-gray-400">
                            {Math.round(comparison.swing1.scores.overall)}
                          </p>
                          <p className="text-sm text-gray-500 mt-1">
                            Overall Score
                          </p>
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                          <div className="text-center">
                            <p className="font-medium text-gray-600">
                              {comparison.swing1.scores.tempo.toFixed(0)}
                            </p>
                            <p className="text-xs text-gray-500">Tempo</p>
                          </div>
                          <div className="text-center">
                            <p className="font-medium text-gray-600">
                              {comparison.swing1.scores.balance.toFixed(0)}
                            </p>
                            <p className="text-xs text-gray-500">Balance</p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-blue-50 rounded-xl p-4 border-2 border-blue-200">
                        <h4 className="text-sm font-medium text-blue-600 mb-3">
                          Current Swing
                        </h4>
                        <div className="text-center">
                          <p className="text-5xl font-bold text-blue-600">
                            {Math.round(comparison.swing2.scores.overall)}
                          </p>
                          <p className="text-sm text-blue-500 mt-1">
                            Overall Score
                          </p>
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                          <div className="text-center">
                            <p className="font-medium text-blue-600">
                              {comparison.swing2.scores.tempo.toFixed(0)}
                            </p>
                            <p className="text-xs text-blue-500">Tempo</p>
                          </div>
                          <div className="text-center">
                            <p className="font-medium text-blue-600">
                              {comparison.swing2.scores.balance.toFixed(0)}
                            </p>
                            <p className="text-xs text-blue-500">Balance</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
