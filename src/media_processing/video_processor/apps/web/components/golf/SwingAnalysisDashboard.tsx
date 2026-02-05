'use client';

import { useState, useCallback, useEffect } from 'react';
import { SwingAnalysis, PoseFrame, Landmark, SwingPhase } from '@/lib/golf/types';
import { analyzeSwing, quickAnalyze } from '@/lib/golf/swingAnalyzer';
import { saveSession, getAllSessions } from '@/lib/golf/persistence';
import { downloadHTMLReport, downloadCSVReport, downloadJSONExport } from '@/lib/golf/reportGenerator';
import { v4 as uuidv4 } from 'uuid';
import ScoreCard from './ScoreCard';
import TempoChart from './TempoChart';
import IssuesPanel from './IssuesPanel';
import MetricsPanel from './MetricsPanel';
import PhaseTimeline from './PhaseTimeline';
import RecommendationsPanel from './RecommendationsPanel';

interface SwingAnalysisDashboardProps {
  videoElement: HTMLVideoElement | null;
  fps: number;
  onAnalysisComplete?: (analysis: SwingAnalysis) => void;
  disabled?: boolean;
}

export default function SwingAnalysisDashboard({
  videoElement,
  fps,
  onAnalysisComplete,
  disabled = false,
}: SwingAnalysisDashboardProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<SwingAnalysis | null>(null);
  const [poseFrames, setPoseFrames] = useState<PoseFrame[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'details' | 'history'>('overview');

  // Handle pose detection results
  const handlePoseDetected = useCallback((landmarks: Landmark[]) => {
    if (!videoElement || isAnalyzing) return;

    const frame: PoseFrame = {
      frameNumber: Math.floor(videoElement.currentTime * fps),
      timestamp: videoElement.currentTime * 1000,
      landmarks,
      confidence: landmarks.reduce((acc, l) => acc + (l.visibility || 0), 0) / landmarks.length,
    };

    setPoseFrames((prev) => {
      // Avoid duplicates
      if (prev.some((f) => f.frameNumber === frame.frameNumber)) {
        return prev;
      }
      return [...prev, frame].sort((a, b) => a.frameNumber - b.frameNumber);
    });

    setCurrentFrame(frame.frameNumber);
  }, [videoElement, fps, isAnalyzing]);

  // Run full swing analysis
  const runAnalysis = useCallback(async () => {
    if (poseFrames.length < 10) {
      setError('Need at least 10 pose frames. Please play the video to capture poses.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = analyzeSwing(poseFrames, fps, videoElement?.src);
      setAnalysis(result);

      // Save to database
      const session = {
        id: result.sessionId,
        timestamp: Date.now(),
        videoFileName: videoElement?.src?.split('/').pop() || 'Unknown',
        videoDuration: videoElement?.duration || 0,
        swingCount: 1,
        analysis: result,
      };

      await saveSession(session, result);

      if (onAnalysisComplete) {
        onAnalysisComplete(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [poseFrames, fps, videoElement, onAnalysisComplete]);

  // Clear current analysis
  const clearAnalysis = useCallback(() => {
    setAnalysis(null);
    setPoseFrames([]);
    setError(null);
  }, []);

  // Export functions
  const handleExportHTML = useCallback(() => {
    if (!analysis) return;
    const session = {
      id: analysis.sessionId,
      timestamp: Date.now(),
      videoFileName: videoElement?.src?.split('/').pop() || 'Unknown',
      videoDuration: videoElement?.duration || 0,
      swingCount: 1,
      analysis,
    };
    downloadHTMLReport(session, analysis);
  }, [analysis, videoElement]);

  const handleExportCSV = useCallback(() => {
    if (!analysis) return;
    downloadCSVReport(analysis);
  }, [analysis]);

  const handleExportJSON = useCallback(() => {
    if (!analysis) return;
    const session = {
      id: analysis.sessionId,
      timestamp: Date.now(),
      videoFileName: videoElement?.src?.split('/').pop() || 'Unknown',
      videoDuration: videoElement?.duration || 0,
      swingCount: 1,
      analysis,
    };
    downloadJSONExport(session, analysis);
  }, [analysis, videoElement]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Swing Analysis</h2>
          <p className="text-sm text-gray-500">
            {poseFrames.length} frames captured
            {analysis && ` • Overall Score: ${Math.round(analysis.scores.overall)}`}
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {analysis && (
            <div className="flex items-center space-x-2">
              <button
                onClick={handleExportHTML}
                className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
              >
                Export HTML
              </button>
              <button
                onClick={handleExportCSV}
                className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
              >
                Export CSV
              </button>
              <button
                onClick={handleExportJSON}
                className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
              >
                Export JSON
              </button>
            </div>
          )}
          <button
            onClick={analysis ? clearAnalysis : runAnalysis}
            disabled={disabled || isAnalyzing || (!analysis && poseFrames.length < 10)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              analysis
                ? 'text-red-700 bg-red-50 hover:bg-red-100'
                : 'text-white bg-blue-600 hover:bg-blue-700'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isAnalyzing ? 'Analyzing...' : analysis ? 'Clear Analysis' : 'Analyze Swing'}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Tabs */}
      {analysis && (
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {(['overview', 'details', 'history'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-3 px-1 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>
      )}

      {/* Content */}
      {analysis ? (
        <div className="space-y-6">
          {activeTab === 'overview' && (
            <>
              {/* Score Card */}
              <ScoreCard scores={analysis.scores} />

              {/* Phase Timeline */}
              <PhaseTimeline
                phases={analysis.phases}
                totalDuration={analysis.tempo.totalSwingDuration}
                currentFrame={currentFrame}
              />

              {/* Tempo Chart */}
              <TempoChart tempo={analysis.tempo} />

              {/* Issues Summary */}
              <IssuesPanel issues={analysis.issues} />

              {/* Recommendations */}
              <RecommendationsPanel recommendations={analysis.recommendations} />
            </>
          )}

          {activeTab === 'details' && (
            <MetricsPanel
              analysis={analysis}
              keyPositions={analysis.keyPositions}
              balance={analysis.balance}
              plane={analysis.plane}
              posture={analysis.posture}
            />
          )}

          {activeTab === 'history' && (
            <SessionHistoryPanel currentSessionId={analysis.sessionId} />
          )}
        </div>
      ) : (
        /* Pre-analysis state */
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <svg
            className="w-16 h-16 mx-auto text-gray-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Analyze</h3>
          <p className="text-sm text-gray-500 max-w-md mx-auto">
            Enable pose detection and play through your swing video. Once enough frames are
            captured, click "Analyze Swing" to generate a comprehensive analysis report.
          </p>
          <div className="mt-6 flex items-center justify-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${poseFrames.length >= 10 ? 'bg-green-500' : 'bg-gray-300'}`} />
              <span>10+ frames required</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-gray-300" />
              <span>{poseFrames.length} captured</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Session History Panel Component
function SessionHistoryPanel({ currentSessionId }: { currentSessionId: string }) {
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAllSessions(10)
      .then(setSessions)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="text-center py-8 text-gray-500">
        Loading session history...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium text-gray-900">Recent Sessions</h3>
      {sessions.length === 0 ? (
        <p className="text-sm text-gray-500">No previous sessions found.</p>
      ) : (
        <div className="space-y-3">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`p-4 border rounded-lg ${
                session.id === currentSessionId
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{session.videoFileName}</p>
                  <p className="text-sm text-gray-500">
                    {new Date(session.timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-blue-600">
                    {Math.round(session.analysis?.scores?.overall || 0)}
                  </p>
                  <p className="text-xs text-gray-500">Score</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
